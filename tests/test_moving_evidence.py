"""Tamper tests for synthetic logs; these fixtures are never experiment results."""

from copy import deepcopy

import pytest

from scripts.verify_moving_target import (
    compare_prefixes,
    parsed_utc,
    pre_motion_prefix,
    verify_action_log,
    verify_force_timing,
)
from vla_replan.config import Episode
from vla_replan.core import Observation, run_episode


@pytest.fixture
def synthetic_log():
    class Environment:
        def reset(self, episode):
            self.index = 0
            return Observation(None, [0, 0, 0], [0, 0]), {}

        def step(self, action):
            self.index += 1
            return Observation(None, [0, 0, 0], [0, 0]), self.index == 12, False

    class Policy:
        def reset(self):
            pass

        def predict(self, observation, seed):
            return [[1.5 if index % 3 == 0 else 0.2, -2.0, 0.0, 0.0, 0.0, 0.0,
                     1.0 if index % 2 == 0 else -1.0] for index in range(50)]

    summary, rows, chunks = run_episode(Environment(), Policy(), Episode("libero_object", "synthetic_test_only", 10, 11, 5, 12))
    summary["condition"] = "static"
    return summary, rows, chunks


def test_executor_fixture_has_exact_saved_action_prefixes(synthetic_log):
    summary, rows, chunks = synthetic_log
    result = verify_action_log(summary, rows, chunks)
    assert result["steps"] == 12
    assert result["model_calls"] == 3
    assert result["predicted_action_values"] == 3 * 50 * 7
    assert result["max_executed_action_error"] == 0.0
    assert summary["clipped_action_values"] > 0
    assert summary["gripper_close_commands"] > 0


def test_changed_executed_action_is_rejected(synthetic_log):
    summary, rows, chunks = synthetic_log
    rows[4]["action"][2] += 1e-12
    with pytest.raises(ValueError, match="clipped predicted prefix"):
        verify_action_log(summary, rows, chunks)


def test_nonfinite_unused_tail_action_is_rejected(synthetic_log):
    summary, rows, chunks = synthetic_log
    chunks[0]["actions"][49][6] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        verify_action_log(summary, rows, chunks)


def test_truncated_chunk_is_rejected(synthetic_log):
    summary, rows, chunks = synthetic_log
    chunks[1]["actions"].pop()
    with pytest.raises(ValueError, match="chunk horizon"):
        verify_action_log(summary, rows, chunks)


def test_wrong_sampling_seed_is_rejected(synthetic_log):
    summary, rows, chunks = synthetic_log
    chunks[1]["sampling_seed"] += 1
    with pytest.raises(ValueError, match="sampling seed"):
        verify_action_log(summary, rows, chunks)


def test_changed_inference_time_total_is_detected(synthetic_log):
    summary, rows, chunks = synthetic_log
    summary["inference_s"] += 0.1
    with pytest.raises(ValueError, match="Inference-time total"):
        verify_action_log(summary, rows, chunks)


def test_inference_time_cannot_appear_without_replanning(synthetic_log):
    summary, rows, chunks = synthetic_log
    rows[1]["inference_s"] = 0.1
    with pytest.raises(ValueError, match="Step inference time"):
        verify_action_log(summary, rows, chunks)


@pytest.mark.parametrize("field,fragment", [
    ("clipped_action_values", "Clipped action count"),
    ("gripper_close_commands", "Gripper close-command"),
    ("model_calls", "Model call total"),
])
def test_incorrect_summary_action_counts_are_rejected(synthetic_log, field, fragment):
    summary, rows, chunks = synthetic_log
    summary[field] += 1
    with pytest.raises(ValueError, match=fragment):
        verify_action_log(summary, rows, chunks)


def test_early_success_without_immediate_stop_is_rejected(synthetic_log):
    summary, rows, chunks = synthetic_log
    rows[0]["success_after_action"] = True
    with pytest.raises(ValueError, match="continued after success"):
        verify_action_log(summary, rows, chunks)


def test_wrong_summary_success_is_rejected(synthetic_log):
    summary, rows, chunks = synthetic_log
    summary["success"] = False
    with pytest.raises(ValueError, match="Summary success"):
        verify_action_log(summary, rows, chunks)


def prefixes(log):
    summary, rows, chunks = log
    result = {}
    for condition in ("static", "slow", "fast"):
        local = deepcopy(summary)
        local["condition"] = condition
        local["episode_id"] += "__motion_" + condition
        result[condition] = deepcopy(pre_motion_prefix(local, rows, chunks, 0.5, 20))
    return result


def test_controlled_prefix_compares_all_ten_actions_and_whole_predictions(synthetic_log):
    treatments = prefixes(synthetic_log)
    result = compare_prefixes(treatments)
    assert len(treatments["static"]["actions"]) == 10
    assert [chunk["step"] for chunk in treatments["static"]["predictions"]] == [0, 5]
    assert result["compared_action_vectors"] == 20
    assert result["compared_prediction_chunks"] == 4
    assert result["max_action_prefix_error"] == result["max_prediction_prefix_error"] == 0.0


def test_prediction_difference_in_unused_pre_onset_tail_is_detected(synthetic_log):
    treatments = prefixes(synthetic_log)
    treatments["slow"]["predictions"][0]["actions"][49][0] += 1e-9
    with pytest.raises(ValueError, match="full predictions differ"):
        compare_prefixes(treatments)


def test_exact_prefix_comparison_does_not_ignore_small_action_differences(synthetic_log):
    treatments = prefixes(synthetic_log)
    treatments["fast"]["actions"][9][0] += 1e-12
    with pytest.raises(ValueError, match="executed actions differ"):
        compare_prefixes(treatments)


def test_after_onset_changes_do_not_contaminate_controlled_prefix(synthetic_log):
    treatments = prefixes(synthetic_log)
    summary, rows, chunks = deepcopy(synthetic_log)
    summary["condition"] = "fast"
    rows[10]["action"][0] -= 0.2
    chunks[2]["actions"][0][0] -= 0.2
    treatments["fast"] = pre_motion_prefix(summary, rows, chunks, 0.5, 20)
    assert compare_prefixes(treatments)["max_action_prefix_error"] == 0.0


@pytest.fixture
def force_log():
    rows = []
    for time in (0.0, 0.45, 0.5, 0.55):
        contact = 0.55 if time >= 0.55 else None
        active = time >= 0.5 and contact is None
        force = 0.1 if active else 0.0
        rows.append({"sim_time_s": time, "motion": {
            "horizontal_force_n": force, "applied_torque_y_nm": -0.03 * force,
            "motion_active": active, "first_contact_sim_time": contact,
        }})
    summary = {"condition": "fast", "sim_time_s": 0.6, "motion_final": {
        **rows[-1]["motion"], "max_force_after_contact_n": 0.0,
    }}
    return summary, rows


def test_force_onset_and_permanent_contact_release_are_consistent(force_log):
    verify_force_timing(*force_log, 0.5, -0.03, 1.0)


@pytest.mark.parametrize("index", [0, 1, 3])
def test_nonzero_force_before_onset_or_after_contact_is_detected(force_log, index):
    summary, rows = force_log
    rows[index]["motion"]["horizontal_force_n"] = 0.01
    rows[index]["motion"]["applied_torque_y_nm"] = -0.0003
    with pytest.raises(ValueError, match="Nonzero intervention"):
        verify_force_timing(summary, rows, 0.5, -0.03, 1.0)


def test_static_force_is_rejected(force_log):
    summary, rows = force_log
    summary["condition"] = "static"
    with pytest.raises(ValueError, match="Nonzero intervention"):
        verify_force_timing(summary, rows, 0.5, -0.03, 1.0)


def test_lost_contact_latch_is_detected(force_log):
    summary, rows = force_log
    summary["motion_final"]["first_contact_sim_time"] = None
    with pytest.raises(ValueError, match="latch disappeared"):
        verify_force_timing(summary, rows, 0.5, -0.03, 1.0)


def test_physics_counters_catch_force_between_control_samples(force_log):
    summary, rows = force_log
    summary["motion_final"]["max_force_after_contact_n"] = 1e-12
    with pytest.raises(ValueError, match="Physics-substep"):
        verify_force_timing(summary, rows, 0.5, -0.03, 1.0)


def test_freeze_timestamp_requires_timezone_and_orders_as_utc():
    assert parsed_utc("2026-10-24T10:00:00+03:00", "test") < parsed_utc("2026-10-24T08:00:00+00:00", "test")
    with pytest.raises(ValueError, match="timezone"):
        parsed_utc("2026-10-24T10:00:00", "test")
