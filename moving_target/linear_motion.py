"""One-way velocity disturbance, smoothly introduced and released on grasp.

Idealized conveyor-like drive: only free-joint world-x velocity is prescribed.
Positions, orientation and the other five velocities remain under physics.
"""
import math
from moving_target.single_shift import SingleShiftEnvironment


def velocity_reference(elapsed, speed_cm_s, onset=.5, ramp=.5):
    if ramp <= 0 or not all(math.isfinite(x) for x in (elapsed, speed_cm_s, onset, ramp)):
        raise ValueError('Finite motion parameters and positive ramp required')
    u = max(0., min(1., (elapsed-onset)/ramp))
    return speed_cm_s / 100 * u*u*(3-2*u)


class LinearMotionEnvironment(SingleShiftEnvironment):
    def __init__(self, config, speed_cm_s=0., onset_s=.5, ramp_s=.5):
        velocity_reference(0., speed_cm_s, onset_s, ramp_s)
        super().__init__(config)
        self.speed_cm_s = float(speed_cm_s)
        self.onset_s, self.ramp_s = onset_s, ramp_s

    def reset(self, episode):
        observation, metadata = super().reset(episode)
        self._grasp_time = None
        self._velocity_writes = 0
        self._last_command = 0.
        self._min_x = self._max_x = self._origin[0]
        metadata.pop('shift_cm', None)
        metadata.pop('shift_time_s', None)
        metadata.update(intervention_kind='smooth_one_way_velocity_until_bilateral_grasp',
                        speed_cm_s=self.speed_cm_s, onset_s=self.onset_s, ramp_s=self.ramp_s)
        return observation, metadata

    def _check_bilateral_grasp(self):
        return self._domain._check_grasp(self._domain.robots[0].gripper,
                                        self._domain.objects_dict[self._object_name])

    def _pre_action(self, action, policy_step=False):
        self._original_pre_action(action, policy_step=policy_step)
        self._detect_contact()
        t = self._elapsed()
        if self._grasp_time is None and self._check_bilateral_grasp():
            self._grasp_time = t
        self._last_command = 0.
        if self.speed_cm_s and t >= self.onset_s and self._grasp_time is None:
            self._last_command = velocity_reference(t, self.speed_cm_s, self.onset_s, self.ramp_s)
            self._domain.sim.data.qvel[self._dof_address] = self._last_command
            self._velocity_writes += 1
        x = float(self._domain.sim.data.body_xpos[self._body_id][0])
        self._min_x, self._max_x = min(self._min_x, x), max(self._max_x, x)

    def telemetry(self):
        if not self._ready:
            return {}
        data = self._domain.sim.data
        return dict(intervention_kind='smooth_one_way_velocity_until_bilateral_grasp',
            elapsed_sim_time_s=self._elapsed(), speed_cm_s=self.speed_cm_s,
            onset_s=self.onset_s, ramp_s=self.ramp_s,
            object_pos=data.body_xpos[self._body_id].tolist(), object_initial_pos=self._origin,
            object_qpos=data.qpos[self._qadr:self._qadr+7].tolist(),
            object_qvel=data.qvel[self._dof_address:self._dof_address+6].tolist(),
            commanded_x_velocity_m_s=self._last_command, velocity_writes=self._velocity_writes,
            pose_writes=0, first_contact_sim_time=self._first_contact_time,
            bilateral_grasp_sim_time=self._grasp_time, drive_released=self._grasp_time is not None,
            grasp_definition='Simultaneous contact with both fingerpads; latched release, not proof of stable lifted grasp',
            x_range_m=[self._min_x,self._max_x])
