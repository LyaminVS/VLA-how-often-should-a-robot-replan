"""Pilot environment: actual varied initial can positions and four drive directions."""
import hashlib
import numpy as np
from moving_target.linear_motion import LinearMotionEnvironment, velocity_reference
from vla_replan.backend import observation_hash


class PlanarMotionEnvironment(LinearMotionEnvironment):
    def __init__(self, config, offset_cm=(0,0), axis=0, speed_cm_s=0.):
        super().__init__(config,speed_cm_s)
        self.offset_cm=tuple(float(x) for x in offset_cm)
        self.axis=int(axis)
        if self.axis not in (0,1):raise ValueError('Axis must be world x or y')
        self._preparing=False

    def reset(self,episode):
        obs,meta=super().reset(episode)
        self._preparing=True
        sim=self._domain.sim
        original=sim.data.qpos[self._qadr:self._qadr+7].copy()
        sim.data.qpos[self._qadr:self._qadr+2]+=np.asarray(self.offset_cm)/100
        sim.data.qvel[self._dof_address:self._dof_address+6]=0
        sim.forward()
        self._validate_contacts()
        # Settle initial placement before the policy is given its first image.
        for _ in range(5):self.env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))
        sim.forward();self._validate_contacts()
        self._origin=sim.data.body_xpos[self._body_id].tolist()
        self._start_time=float(sim.data.time)
        self._first_contact_time=self._first_contact_step=self._grasp_time=None
        self._control_step=self._velocity_writes=0
        self._last_command=0.;self._min_x=self._max_x=self._origin[0]
        self._preparing=False
        raw=self.env._format_raw_obs(self._domain._get_observations(force_update=True))
        obs=self._observation(raw)
        meta.update(initial_observation_sha256=observation_hash(raw),
            initial_scene_sha256=hashlib.sha256(sim.get_state().flatten().tobytes()).hexdigest(),
            original_can_qpos=original.tolist(),initial_can_position_m=self._origin,
            initial_can_offset_cm=list(self.offset_cm),drive_axis='xy'[self.axis],speed_cm_s=self.speed_cm_s,
            initial_placement_pose_writes=1,rollout_pose_writes=0,
            intervention_kind='planar_one_way_velocity_until_bilateral_contact')
        return obs,meta

    def _validate_contacts(self):
        invalid=[c for c in self.object_contacts() if c['distance_m']<-.001 and c['other_body'] not in ('floor','table')]
        if invalid:raise ValueError('Initial target overlap: '+str(invalid))

    def _pre_action(self,action,policy_step=False):
        self._original_pre_action(action,policy_step=policy_step)
        if self._preparing:return
        self._detect_contact();t=self._elapsed()
        if self._grasp_time is None and self._check_bilateral_grasp():self._grasp_time=t
        self._last_command=0.
        if self.speed_cm_s and t>=self.onset_s and self._grasp_time is None:
            self._last_command=velocity_reference(t,self.speed_cm_s,self.onset_s,self.ramp_s)
            self._domain.sim.data.qvel[self._dof_address+self.axis]=self._last_command
            self._velocity_writes+=1

    def telemetry(self):
        m=super().telemetry()
        if m:
            m.update(intervention_kind='planar_one_way_velocity_until_bilateral_contact',axis='xy'[self.axis],initial_offset_cm=list(self.offset_cm),commanded_axis_velocity_m_s=self._last_command)
            m.pop('commanded_x_velocity_m_s',None)
        return m
