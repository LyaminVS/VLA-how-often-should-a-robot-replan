"""No policy calls: verify intervention, release, and static-control invariants."""
from types import SimpleNamespace
import numpy as np
from moving_target.kinematic import KinematicTargetEnvironment

def fake(condition='fast',contact=False):
    env=object.__new__(KinematicTargetEnvironment)
    env.condition=condition;env.motion_spec=SimpleNamespace(start_delay_s=.5)
    env._first_contact_time=1. if contact else None
    env._original_pre_action=lambda action,policy_step=False:None
    env._detect_contact=lambda:None;env._elapsed=lambda:1.5
    env._reference=lambda:([.04,0.,.8],.1)
    env._body_id=0;env._qadr=0;env._dof_address=0
    env._initial_pose=np.array([0.,0.,.8,1.,0.,0.,0.]);env._initial_axis_z=[0.,0.,1.]
    env._kinematic_writes=env._writes_after_contact=env._active_samples=env._saturated_samples=0
    env._max_imposed_tracking_error=env._error_sum=env._squared_error_sum=env._max_precontact_tilt=0.
    env._x_min=env._x_max=env._y_min=env._y_max=0.;env._z_min=env._z_max=.8
    data=SimpleNamespace(qpos=env._initial_pose.copy(),qvel=np.ones(6),xfrc_applied=np.ones((1,6)),body_xpos=np.array([[0.,0.,.8]]),body_xmat=np.eye(3).reshape(1,9))
    def forward():data.body_xpos[0]=data.qpos[:3]
    env._domain=SimpleNamespace(sim=SimpleNamespace(data=data,forward=forward))
    return env,data

def test_active_imposes_pose_velocity_without_external_force():
    e,d=fake();e._pre_action(np.zeros(7));assert d.qpos[0]==.04;assert np.array_equal(d.qvel,[.1,0,0,0,0,0]);assert not d.xfrc_applied.any();assert e._kinematic_writes==1;assert e._max_imposed_tracking_error==0

def test_contact_permanently_releases_pose_and_velocity():
    e,d=fake();e._pre_action(np.zeros(7));e._first_contact_time=1.6;d.qpos[0]=.123;d.qvel[:]=2.;e._pre_action(np.zeros(7));assert d.qpos[0]==.123;assert (d.qvel==2.).all();assert e._kinematic_writes==1

def test_static_control_never_writes_joint():
    e,d=fake('static');before=d.qpos.copy();e._pre_action(np.zeros(7));assert np.array_equal(before,d.qpos);assert (d.qvel==1.).all();assert e._kinematic_writes==0
