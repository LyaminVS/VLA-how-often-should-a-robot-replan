"""Bounded horizontal force intervention; the pretrained policy is unchanged.

Forces are applied at physics substeps and permanently released on first finger
contact. The moving reference is a protocol input, not an observation given to VLA.
"""

import math

from vla_replan.backend import LiberoEnvironment


def reference_motion(elapsed_s, period_s, amplitude_m, start_delay_s, ramp_s=1.0):
    """Displacement and velocity of a smoothly introduced sinusoidal reference."""
    tau = elapsed_s - start_delay_s
    if period_s is None or tau <= 0:
        return 0.0, 0.0
    omega = 2 * math.pi / period_s
    u = min(tau / ramp_s, 1.0)
    ramp = u * u * (3 - 2 * u)
    derivative = 6 * u * (1 - u) / ramp_s if u < 1 else 0.0
    return (
        amplitude_m * ramp * math.sin(omega * tau),
        amplitude_m * (derivative * math.sin(omega * tau) + ramp * omega * math.cos(omega * tau)),
    )


class MovingTargetEnvironment(LiberoEnvironment):
    def __init__(self, base_config, motion_spec):
        super().__init__(base_config)
        self.motion_spec = motion_spec
        self.condition = "static"
        self._domain = None
        self._original_pre_action = None
        self._ready = False

    def set_condition(self, condition):
        if condition not in self.motion_spec.periods_s:
            raise ValueError(f"Unknown motion condition: {condition}")
        self.condition = condition

    def _detach(self):
        if self._domain is not None and self._original_pre_action is not None:
            if self._ready:
                self._domain.sim.data.xfrc_applied[self._body_id, :] = 0
            self._domain._pre_action = self._original_pre_action
        self._original_pre_action = None
        self._domain = None
        self._ready = False

    def reset(self, episode):
        self._detach()
        observation, metadata = super().reset(episode)
        domain = self.env._env.env
        candidates = [name for name in domain.objects_dict if "alphabet_soup" in name]
        if len(candidates) != 1:
            raise ValueError(f"Expected one alphabet soup object, got {candidates}")
        self._domain = domain
        self._object_name = candidates[0]
        obj = domain.objects_dict[self._object_name]
        self._body_id = domain.sim.model.body_name2id(obj.root_body)
        joint_id = domain.sim.model.joint_name2id(obj.joints[-1])
        if int(domain.sim.model.jnt_type[joint_id]) != 0:
            raise ValueError("Moving target must have a free joint")
        self._dof_address = int(domain.sim.model.jnt_dofadr[joint_id])
        descendants = {self._body_id}
        for body in range(domain.sim.model.nbody):
            parent = body
            while parent > 0:
                if parent == self._body_id:
                    descendants.add(body)
                    break
                parent = int(domain.sim.model.body_parentid[parent])
        self._object_geoms = {
            geom for geom in range(domain.sim.model.ngeom)
            if int(domain.sim.model.geom_bodyid[geom]) in descendants
        }
        self._finger_geoms = set()
        for robot in domain.robots:
            important = robot.gripper.important_geoms
            for key, names in important.items():
                if "finger" in key:
                    self._finger_geoms.update(domain.sim.model.geom_name2id(name) for name in names)
        if not self._finger_geoms or not self._object_geoms:
            raise ValueError("Cannot identify target and finger collision geoms")
        self._origin = [float(v) for v in domain.sim.data.body_xpos[self._body_id]]
        self._initial_axis_z = [float(domain.sim.data.body_xmat[self._body_id][i]) for i in (2, 5, 8)]
        self._max_precontact_tilt = 0.0
        self._start_time = float(domain.sim.data.time)
        self._first_contact_time = None
        self._first_contact_step = None
        self._control_step = 0
        self._last_force = 0.0
        self._last_saturated = False
        self._active_samples = self._saturated_samples = 0
        self._error_sum = self._squared_error_sum = 0.0
        self._force_peak = self._force_after_contact_peak = 0.0
        self._x_min = self._x_max = self._origin[0]
        self._y_min = self._y_max = self._origin[1]
        self._z_min = self._z_max = self._origin[2]
        self._ready = True
        self._original_pre_action = domain._pre_action
        domain._pre_action = self._pre_action
        self._detect_contact()
        metadata.update(
            motion_object_name=self._object_name,
            motion_object_mass_kg=float(domain.sim.model.body_mass[self._body_id]),
            motion_physics_timestep_s=float(domain.sim.model.opt.timestep),
            motion_object_initial_position_m=self._origin,
            motion_target_geom_names=[domain.sim.model.geom_id2name(i) for i in sorted(self._object_geoms)],
            motion_finger_geom_names=[domain.sim.model.geom_id2name(i) for i in sorted(self._finger_geoms)],
        )
        return observation, metadata

    def _elapsed(self):
        return max(0.0, float(self._domain.sim.data.time) - self._start_time)

    def _detect_contact(self):
        if self._first_contact_time is not None:
            return
        data = self._domain.sim.data
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            if ((a in self._object_geoms and b in self._finger_geoms)
                    or (b in self._object_geoms and a in self._finger_geoms)) and contact.dist <= 0:
                self._first_contact_time = self._elapsed()
                # First control observation boundary at or after the physical contact.
                self._first_contact_step = int(math.ceil(self._first_contact_time * self.config.control_hz - 1e-8))
                data.xfrc_applied[self._body_id, :] = 0
                self._last_force = 0.0
                self._last_saturated = False
                return

    def _reference(self):
        displacement, velocity = reference_motion(
            self._elapsed(), self.motion_spec.periods_s[self.condition],
            self.motion_spec.amplitude_m, self.motion_spec.start_delay_s, self.motion_spec.ramp_s,
        )
        return [self._origin[0] + displacement, self._origin[1], self._origin[2]], velocity

    def _pre_action(self, action, policy_step=False):
        self._original_pre_action(action, policy_step=policy_step)
        self._detect_contact()
        data = self._domain.sim.data
        data.xfrc_applied[self._body_id, :] = 0.0
        position = data.body_xpos[self._body_id]
        target, velocity = self._reference()
        active = (self.condition != "static" and self._elapsed() >= self.motion_spec.start_delay_s
                  and self._first_contact_time is None)
        force = 0.0
        saturated = False
        if active:
            error = target[0] - float(position[0])
            actual_velocity = float(data.qvel[self._dof_address])
            requested = (self.motion_spec.kp_n_per_m * error
                         + self.motion_spec.kd_ns_per_m * (velocity - actual_velocity))
            limit = self.motion_spec.max_force_n
            force = max(-limit, min(limit, requested))
            saturated = abs(requested) > limit
            data.xfrc_applied[self._body_id, 0] = force
            # Equivalent moment of the same horizontal force applied below COM.
            # A low application point reduces the tipping caused by table friction.
            data.xfrc_applied[self._body_id, 4] = self.motion_spec.force_application_offset_z_m * force
            self._active_samples += 1
            self._saturated_samples += int(saturated)
            self._error_sum += abs(error)
            self._squared_error_sum += error * error
        if self._first_contact_time is None:
            axis_z = [float(data.body_xmat[self._body_id][i]) for i in (2, 5, 8)]
            cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(axis_z, self._initial_axis_z))))
            self._max_precontact_tilt = max(self._max_precontact_tilt, math.degrees(math.acos(cosine)))
            self._x_min, self._x_max = min(self._x_min, float(position[0])), max(self._x_max, float(position[0]))
            self._y_min, self._y_max = min(self._y_min, float(position[1])), max(self._y_max, float(position[1]))
            self._z_min, self._z_max = min(self._z_min, float(position[2])), max(self._z_max, float(position[2]))
        self._last_force, self._last_saturated = force, saturated
        self._force_peak = max(self._force_peak, abs(force))
        if self._first_contact_time is not None:
            self._force_after_contact_peak = max(self._force_after_contact_peak, abs(force))

    def step(self, action):
        result = super().step(action)
        self._control_step += 1
        self._detect_contact()
        return result

    def telemetry(self):
        if not self._ready:
            return {}
        data = self._domain.sim.data
        target, reference_velocity = self._reference()
        position = [float(v) for v in data.body_xpos[self._body_id]]
        contact = self._first_contact_time
        samples = self._active_samples
        return {
            "condition": self.condition,
            "elapsed_sim_time_s": self._elapsed(),
            "object_pos": position,
            "object_initial_pos": self._origin,
            "target_pos": target,
            "object_velocity_x_m_s": float(data.qvel[self._dof_address]),
            "reference_velocity_x_m_s": reference_velocity,
            "motion_active": self.condition != "static" and contact is None
                             and self._elapsed() >= self.motion_spec.start_delay_s,
            "first_contact_sim_time": contact,
            "first_contact_step": self._first_contact_step,
            "tracking_error_m": abs(position[0] - target[0]),
            "horizontal_force_n": self._last_force,
            "applied_torque_y_nm": self.motion_spec.force_application_offset_z_m * self._last_force,
            "force_saturated": self._last_saturated,
            "active_physics_samples": samples,
            "active_sim_time_s": samples * float(self._domain.sim.model.opt.timestep),
            "tracking_mae_m": self._error_sum / samples if samples else None,
            "tracking_rmse_m": math.sqrt(self._squared_error_sum / samples) if samples else None,
            "force_saturation_fraction": self._saturated_samples / samples if samples else None,
            "peak_force_n": self._force_peak,
            "max_force_after_contact_n": self._force_after_contact_peak,
            "precontact_x_range_m": self._x_max - self._x_min,
            "precontact_y_range_m": self._y_max - self._y_min,
            "precontact_z_range_m": self._z_max - self._z_min,
            "max_precontact_tilt_deg": self._max_precontact_tilt,
            "contact_is_grasp": False,
        }

    def close(self):
        self._detach()
        super().close()
