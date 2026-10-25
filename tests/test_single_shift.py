from types import SimpleNamespace
import numpy as np
from moving_target.single_shift import SingleShiftEnvironment


def fake(shift=4, contact=None):
    env = object.__new__(SingleShiftEnvironment)
    env.shift_cm = shift
    env.shift_time_s = 1.025
    env._shift_event = None
    env._pose_writes = 0
    env._body_id = env._qadr = env._dof_address = 0
    env._first_contact_time = contact
    env._original_pre_action = lambda action, policy_step=False: None
    env._detect_contact = lambda: None
    env.object_contacts = lambda: []
    env._elapsed = lambda: 1.026
    data = SimpleNamespace(qpos=np.array([.1, .2, .8, 1., 0., 0., 0.]),
                           qvel=np.arange(6, dtype=float), xfrc_applied=np.ones((1, 6)))
    env._domain = SimpleNamespace(sim=SimpleNamespace(data=data, forward=lambda: None))
    return env, data


def test_shift_is_exactly_one_additive_position_write_and_preserves_velocity():
    env, data = fake()
    before, velocity = data.qpos.copy(), data.qvel.copy()
    env._pre_action([0] * 7)
    assert np.allclose(data.qpos, before + [.04, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(data.qvel, velocity)
    assert env._pose_writes == 1
    data.qpos[0] = .7  # Ordinary physics/robot motion after the intervention.
    env._pre_action([0] * 7)
    assert data.qpos[0] == .7 and env._pose_writes == 1


def test_zero_shift_never_writes_pose_or_velocity():
    env, data = fake(0)
    before, velocity = data.qpos.copy(), data.qvel.copy()
    env._pre_action([0] * 7)
    assert np.array_equal(data.qpos, before) and np.array_equal(data.qvel, velocity)
    assert env._pose_writes == 0 and env._shift_event is not None


def test_no_early_shift():
    env, data = fake()
    env._elapsed = lambda: 1.024
    before = data.qpos.copy()
    env._pre_action([0] * 7)
    assert np.array_equal(data.qpos, before) and env._shift_event is None


def test_negative_direction_and_contact_do_not_change_event_schedule():
    env, data = fake(-8, contact=.9)
    env._pre_action([0] * 7)
    assert np.isclose(data.qpos[0], .02)
    assert env._shift_event['finger_contact_before_shift'] is True
