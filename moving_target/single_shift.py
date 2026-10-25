"""One registered relocation, followed by entirely unmodified object physics.

This is an idealized position disturbance, not a force-driven actuator. It does
not hold the target, change its velocity, return it, or stop at finger contact.
"""
from moving_target.environment import MovingTargetEnvironment
from moving_target.config import MotionSpec


class SingleShiftEnvironment(MovingTargetEnvironment):
    def __init__(self, config, shift_cm=0.0, shift_time_s=1.025):
        super().__init__(config, MotionSpec())
        self.shift_cm = float(shift_cm)
        self.shift_time_s = float(shift_time_s)

    def reset(self, episode):
        observation, metadata = super().reset(episode)
        obj = self._domain.objects_dict[self._object_name]
        joint = self._domain.sim.model.joint_name2id(obj.joints[-1])
        self._qadr = int(self._domain.sim.model.jnt_qposadr[joint])
        self._shift_event = None
        self._pose_writes = 0
        metadata.update(intervention_kind="single_world_x_position_shift",
                        shift_cm=self.shift_cm, shift_time_s=self.shift_time_s)
        return observation, metadata

    def object_contacts(self):
        sim = self._domain.sim
        contacts = []
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if a in self._object_geoms or b in self._object_geoms:
                other = b if a in self._object_geoms else a
                contacts.append({"other_geom": sim.model.geom_id2name(other),
                                 "other_body": sim.model.body_id2name(int(sim.model.geom_bodyid[other])),
                                 "distance_m": float(c.dist)})
        return contacts

    def _pre_action(self, action, policy_step=False):
        self._original_pre_action(action, policy_step=policy_step)
        self._detect_contact()
        sim = self._domain.sim
        sim.data.xfrc_applied[self._body_id, :] = 0.0
        if self._shift_event is None and self._elapsed() + 1e-10 >= self.shift_time_s:
            before = sim.data.qpos[self._qadr:self._qadr + 7].copy()
            velocity = sim.data.qvel[self._dof_address:self._dof_address + 6].copy()
            contacts = self.object_contacts()
            if self.shift_cm != 0:
                sim.data.qpos[self._qadr] += self.shift_cm / 100.0
                sim.forward()
                self._pose_writes += 1
            self._shift_event = {
                "actual_sim_time_s": self._elapsed(),
                "requested_time_s": self.shift_time_s,
                "shift_cm": self.shift_cm,
                "pose_before": before.tolist(),
                "pose_after": sim.data.qpos[self._qadr:self._qadr + 7].tolist(),
                "velocity_before": velocity.tolist(),
                "velocity_after": sim.data.qvel[self._dof_address:self._dof_address + 6].tolist(),
                "finger_contact_before_shift": self._first_contact_time is not None,
                "contacts_before": contacts,
                "contacts_after": self.object_contacts(),
            }
            self._detect_contact()

    def telemetry(self):
        if not self._ready:
            return {}
        data = self._domain.sim.data
        return {
            "intervention_kind": "single_world_x_position_shift",
            "elapsed_sim_time_s": self._elapsed(),
            "object_pos": [float(v) for v in data.body_xpos[self._body_id]],
            "object_initial_pos": self._origin,
            "object_qpos": data.qpos[self._qadr:self._qadr + 7].tolist(),
            "object_qvel": data.qvel[self._dof_address:self._dof_address + 6].tolist(),
            "shift_cm": self.shift_cm,
            "shift_time_s": self.shift_time_s,
            "shift_applied": self._shift_event is not None,
            "pose_writes": self._pose_writes,
            "shift_event": self._shift_event,
            "first_contact_sim_time": self._first_contact_time,
            "first_contact_step": self._first_contact_step,
            "contact_is_grasp": False,
        }
