"""Controlled kinematic target motion, permanently released at finger contact.

This is an idealized moving-target intervention, not a force-controlled physical
actuator. Before contact, impose the prescribed free-joint pose and velocity at
physics substeps. After contact, never write the object joint again. The VLA
receives the ordinary rendered images and robot state only.
"""
import math
from .environment import MovingTargetEnvironment,reference_motion

class KinematicTargetEnvironment(MovingTargetEnvironment):
    def reset(self,episode):
        obs,metadata=super().reset(episode)
        # Parent setup is complete before the first externally stepped action.
        model=self._domain.sim.model;obj=self._domain.objects_dict[self._object_name]
        jid=model.joint_name2id(obj.joints[-1]);self._qadr=int(model.jnt_qposadr[jid]);self._initial_pose=self._domain.sim.data.qpos[self._qadr:self._qadr+7].copy()
        self._kinematic_writes=0;self._writes_after_contact=0;self._max_imposed_tracking_error=0.
        metadata['intervention_kind']='kinematic_target_until_finger_contact'
        return obs,metadata

    def _pre_action(self,action,policy_step=False):
        self._original_pre_action(action,policy_step=policy_step)
        self._detect_contact()
        sim=self._domain.sim;data=sim.data;data.xfrc_applied[self._body_id,:]=0.
        active=(self.condition!='static' and self._elapsed()>=self.motion_spec.start_delay_s and self._first_contact_time is None)
        target,velocity=self._reference()
        if active:
            assert self._first_contact_time is None
            data.qpos[self._qadr:self._qadr+7]=self._initial_pose
            data.qpos[self._qadr]=target[0]
            data.qvel[self._dof_address:self._dof_address+6]=0.
            data.qvel[self._dof_address]=velocity
            sim.forward()
            self._kinematic_writes+=1
            error=abs(float(data.body_xpos[self._body_id,0])-target[0])
            self._max_imposed_tracking_error=max(self._max_imposed_tracking_error,error)
            self._active_samples+=1;self._error_sum+=error;self._squared_error_sum+=error*error
            self._detect_contact()
        position=data.body_xpos[self._body_id]
        if self._first_contact_time is None:
            axis=[float(data.body_xmat[self._body_id][i]) for i in (2,5,8)]
            cosine=max(-1.,min(1.,sum(a*b for a,b in zip(axis,self._initial_axis_z))))
            self._max_precontact_tilt=max(self._max_precontact_tilt,math.degrees(math.acos(cosine)))
            self._x_min,self._x_max=min(self._x_min,float(position[0])),max(self._x_max,float(position[0]))
            self._y_min,self._y_max=min(self._y_min,float(position[1])),max(self._y_max,float(position[1]))
            self._z_min,self._z_max=min(self._z_min,float(position[2])),max(self._z_max,float(position[2]))
        self._last_force=0.;self._last_saturated=False

    def telemetry(self):
        row=super().telemetry()
        if row:
            row.update(intervention_kind='kinematic_target_until_finger_contact',kinematic_writes=getattr(self,'_kinematic_writes',0),kinematic_writes_after_contact=getattr(self,'_writes_after_contact',0),max_imposed_tracking_error_m=getattr(self,'_max_imposed_tracking_error',0.))
        return row
