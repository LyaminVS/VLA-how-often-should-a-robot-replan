from types import SimpleNamespace
import numpy as np
from moving_target.linear_motion import LinearMotionEnvironment, velocity_reference


def test_ramp_is_continuous_monotone_and_never_returns():
    t=np.linspace(0,14,14001)
    v=np.array([velocity_reference(x,2) for x in t])
    assert v.min()==0 and np.all(np.diff(v)>=-1e-15)
    assert v[-1]==.02 and velocity_reference(.5,2)==0
    assert np.max(np.diff(v))<.000061
    assert velocity_reference(3,-2)==-.02


def fake_env(speed):
    env=object.__new__(LinearMotionEnvironment)
    env.speed_cm_s=speed;env.onset_s=.5;env.ramp_s=.5
    env._grasp_time=None;env._velocity_writes=0;env._last_command=0
    env._body_id=0;env._dof_address=0;env._min_x=env._max_x=0
    env._original_pre_action=lambda *args,**kwargs:None
    env._detect_contact=lambda:None
    env._elapsed=lambda:1.5
    env._check_bilateral_grasp=lambda:False
    data=SimpleNamespace(qpos=np.arange(7,dtype=float),qvel=np.arange(6,dtype=float),body_xpos=np.zeros((1,3)))
    env._domain=SimpleNamespace(sim=SimpleNamespace(data=data))
    return env,data


def test_zero_is_free_physics():
    env,data=fake_env(0)
    before=data.qvel.copy();pos=data.qpos.copy()
    env._pre_action([0]*7)
    assert np.array_equal(before,data.qvel) and np.array_equal(pos,data.qpos)
    assert env._velocity_writes==0


def test_only_x_velocity_changes_until_bilateral_grasp_then_permanent_release():
    env,data=fake_env(2)
    pos=data.qpos.copy();vel=data.qvel.copy()
    env._pre_action([0]*7)
    assert data.qvel[0]==.02 and np.array_equal(data.qvel[1:],vel[1:])
    assert np.array_equal(pos,data.qpos)
    env._check_bilateral_grasp=lambda:True
    data.qvel[0]=.007
    env._pre_action([0]*7)
    assert env._grasp_time==1.5 and data.qvel[0]==.007
    env._check_bilateral_grasp=lambda:False
    env._pre_action([0]*7)
    assert data.qvel[0]==.007 and env._velocity_writes==1
