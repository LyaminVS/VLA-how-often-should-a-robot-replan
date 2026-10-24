"""Intervention invariants at physics substeps, independent of LIBERO assets."""

from types import SimpleNamespace

import numpy as np
import pytest

from moving_target.config import MotionSpec
from moving_target.environment import MovingTargetEnvironment, reference_motion
from vla_replan.config import Episode
from vla_replan.core import Observation, run_episode


@pytest.fixture
def intervention():
    env = object.__new__(MovingTargetEnvironment)
    env.motion_spec = MotionSpec()
    env.config = SimpleNamespace(control_hz=20)
    env.condition = "fast"
    data = SimpleNamespace(
        time=1.5, body_xpos=np.zeros((1, 3)), qvel=np.zeros(6),
        body_xmat=np.asarray([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=float),
        xfrc_applied=np.zeros((1, 6)), ncon=0, contact=[],
    )
    env._domain = SimpleNamespace(sim=SimpleNamespace(data=data, model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002))))
    env._original_pre_action = lambda action, policy_step=False: None
    env._body_id = env._dof_address = 0
    env._origin = [0.0, 0.0, 0.0]
    env._initial_axis_z = [0.0, 0.0, 1.0]
    env._max_precontact_tilt = 0.0
    env._start_time = 0.0
    env._first_contact_time = env._first_contact_step = None
    env._control_step = 30
    env._last_force = 0.0
    env._last_saturated = False
    env._active_samples = env._saturated_samples = 0
    env._error_sum = env._squared_error_sum = 0.0
    env._force_peak = env._force_after_contact_peak = 0.0
    env._x_min = env._x_max = env._y_min = env._y_max = env._z_min = env._z_max = 0.0
    env._object_geoms = {1}
    env._finger_geoms = {2}
    env._ready = True
    return env


def test_horizontal_force_is_capped_and_point_of_application_torque_is_consistent(intervention):
    data = intervention._domain.sim.data
    for time, expected in ((1.5, 1.0), (3.5, -1.0)):
        data.time = time
        intervention._pre_action([0] * 7)
        assert data.xfrc_applied[0, 0] == pytest.approx(expected)
        assert data.xfrc_applied[0, 4] == pytest.approx(intervention.motion_spec.force_application_offset_z_m * expected)
        assert np.all(data.xfrc_applied[0, [1, 2, 3, 5]] == 0)
        assert intervention.telemetry()["force_saturated"]


def test_static_condition_has_no_force_or_active_physics_samples(intervention):
    intervention.set_condition("static")
    data = intervention._domain.sim.data
    data.body_xpos[0, 0] = 1.0
    for time in (0.0, 0.5, 1.5, 4.0):
        data.time = time
        intervention._pre_action([0] * 7)
        assert np.all(data.xfrc_applied == 0)
    assert intervention.telemetry()["active_physics_samples"] == 0
    assert intervention.telemetry()["tracking_mae_m"] is None


def test_no_force_before_scheduled_simulation_time(intervention):
    intervention._domain.sim.data.time = 0.49
    intervention._pre_action([0] * 7)
    assert intervention.telemetry()["horizontal_force_n"] == 0.0
    assert not intervention.telemetry()["motion_active"]


def test_finger_contact_disables_force_permanently_at_next_physics_substep(intervention):
    data = intervention._domain.sim.data
    intervention._pre_action([0] * 7)
    assert data.xfrc_applied[0, 0] != 0.0
    data.ncon = 1
    data.contact = [SimpleNamespace(geom1=2, geom2=1, dist=-0.0001)]
    intervention._pre_action([0] * 7)
    assert np.all(data.xfrc_applied == 0)
    assert intervention.telemetry()["first_contact_sim_time"] == 1.5
    assert intervention.telemetry()["first_contact_step"] == 30
    data.ncon = 0
    data.contact = []
    data.time = 2.0
    intervention._pre_action([0] * 7)
    result = intervention.telemetry()
    assert result["horizontal_force_n"] == 0.0
    assert result["applied_torque_y_nm"] == 0.0
    assert not result["motion_active"]
    assert result["max_force_after_contact_n"] == 0.0
    assert result["contact_is_grasp"] is False


def test_contact_step_is_first_control_boundary_after_substep_contact(intervention):
    data = intervention._domain.sim.data
    data.time = 1.502
    data.ncon = 1
    data.contact = [SimpleNamespace(geom1=2, geom2=1, dist=-0.0001)]
    intervention._pre_action([0] * 7)
    assert intervention.telemetry()["first_contact_step"] == 31
    assert np.all(data.xfrc_applied == 0)


def test_table_contact_does_not_trigger_finger_contact(intervention):
    data = intervention._domain.sim.data
    data.ncon = 1
    data.contact = [SimpleNamespace(geom1=1, geom2=99, dist=-0.0001)]
    intervention._pre_action([0] * 7)
    assert intervention.telemetry()["first_contact_step"] is None
    assert intervention.telemetry()["motion_active"]


def test_sinusoid_is_continuous_at_onset_and_end_of_ramp():
    def reference(time):
        return reference_motion(time, 4.0, 0.02, 0.5, 1.0)

    assert reference(0.49) == (0.0, 0.0)
    assert reference(0.5) == (0.0, 0.0)
    assert reference(0.5 + 1e-8) == pytest.approx((0.0, 0.0), abs=1e-10)
    assert reference(1.5 - 1e-8) == pytest.approx(reference(1.5 + 1e-8), abs=1e-8)
    t, eps = 0.9, 1e-6
    finite_difference = (reference(t + eps)[0] - reference(t - eps)[0]) / (2 * eps)
    assert finite_difference == pytest.approx(reference(t)[1], rel=1e-7)
    assert reference_motion(4, None, 0.02, 0.5, 1.0) == (0.0, 0.0)


def test_motion_changes_do_not_create_extra_replanning_calls():
    class Environment:
        def reset(self, episode):
            self.step_index = 0
            return Observation(None, [0, 0, 0], [0, 0]), {}

        def step(self, action):
            self.step_index += 1
            return Observation(None, [float(self.step_index), 0, 0], [0, 0]), False, False

    class Policy:
        def reset(self):
            pass

        def predict(self, observation, seed):
            return [[0] * 7 for _ in range(50)]

    env = Environment()

    def record(frame, row, outcome):
        row["motion"] = {"motion_active": 2 <= env.step_index < 8}

    episode = Episode("libero_object", "test", 0, 11, 5, 12)
    summary, rows, chunks = run_episode(env, Policy(), episode, record=record)
    assert [chunk["step"] for chunk in chunks] == [0, 5, 10]
    assert summary["model_calls"] == 3
    assert [row["step"] for row in rows if row["replanned"]] == [0, 5, 10]
    assert not rows[2]["replanned"] and not rows[8]["replanned"]
