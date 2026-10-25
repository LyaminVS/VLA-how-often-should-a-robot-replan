"""The revision target is the next unexecuted old action at the same time index."""

import copy
import math

import pytest

from scripts.analyze_plan_revision import extract_episode, variance_translation


def diagnostic(horizon=5, *, target=2, scale=1):
    estimates = [[[0.0] * 7 for _ in range(horizon)] for _ in range(2)]
    # XYZ variance=4; all seven coordinates include rotation/gripper: 4+9+16=29.
    for call, sign in enumerate((-1.0, 1.0)):
        estimates[call][target] = [sign * 2 * scale, 0, 0, sign * 3 * scale, 0, 0, sign * 4 * scale]
    variance = [0.0] * horizon
    variance[target] = 29.0 * scale * scale
    return {"clean_estimates": estimates, "variance": variance}


def evidence(*, k=2, bootstrap=False):
    horizon = 5
    old_actions = [[10.0 + index, 0, 0, 0, 0, 0, -1.0] for index in range(horizon)]
    old_actions[2] = [2.0, -2.0, 0, 0, 0, 0, -1.0]
    new_actions = [[-2.0, 2.0, 0, 0, 0, 0, 1.0] for _ in range(horizon)]
    chunks = [
        {"step": 0, "sampling_seed": 11, "executed_steps": k,
         "decision": {"chosen_k": k, "bootstrap": bootstrap, "threshold": None, "cap_applied": False},
         "actions": old_actions, "diagnostics": diagnostic()},
        {"step": k, "sampling_seed": 22, "executed_steps": 2,
         "decision": {"chosen_k": 2, "bootstrap": False, "threshold": None, "cap_applied": False},
         "actions": new_actions, "diagnostics": diagnostic(target=0, scale=10)},
    ]
    steps = [{"step": index, "replanned": index in (0, k),
              "chunk_index": index if index < k else index - k,
              "plan_revision_translation_l2": None} for index in range(k + 2)]
    if k < horizon:
        steps[k]["plan_revision_translation_l2"] = math.sqrt(math.fsum(
            (new_actions[0][coordinate] - old_actions[k][coordinate]) ** 2 for coordinate in range(3)))
    summary = {"episode_id": "test-episode", "suite": "test-suite", "task": "test-task",
               "state": 0, "policy_seed": 11, "method": "fixed_2", "success": False,
               "model_calls": 2, "steps": len(steps), "max_steps": 10}
    return summary, steps, chunks


def extract(value):
    return extract_episode(*value, "supplemental", horizon=5, tail_length=2)


def test_target_is_old_k_not_k_minus_one_and_variance_is_from_old_call():
    events, coverage = extract(evidence())
    assert len(events) == 1
    event = events[0]
    assert (event["old_call_index"], event["old_step"], event["new_step"], event["k"]) == (0, 0, 2, 2)
    assert event["variance_total"] == 29.0
    assert event["variance_translation"] == 4.0
    assert event["variance_total"] != 2900.0  # The new call's variance at position zero.
    assert event["revision_translation"] == pytest.approx(math.sqrt(32.0))
    assert event["old_sampling_seed"] == 11 and event["new_sampling_seed"] == 22
    assert coverage["measurable_transitions"] == coverage["included_nonbootstrap"] == 1
    assert coverage["no_next_call"] == 1


def test_clipped_revision_is_a_separate_sensitivity_metric():
    event = extract(evidence())[0][0]
    assert event["revision_translation_clipped"] == pytest.approx(math.sqrt(8.0))
    assert event["revision_translation"] == 2 * event["revision_translation_clipped"]
    assert event["gripper_flip"] is True and event["gripper_zero_involved"] is False


@pytest.mark.parametrize("field", ["next_step", "executed_steps", "chosen_k"])
def test_boundary_mismatch_is_rejected(field):
    data = evidence()
    if field == "next_step":
        data[2][1]["step"] = 3
    elif field == "executed_steps":
        data[2][0]["executed_steps"] = 1
    else:
        data[2][0]["decision"]["chosen_k"] = 3
    with pytest.raises(ValueError, match="boundary"):
        extract(data)


def test_logged_revision_must_agree_with_raw_chunks():
    data = evidence()
    data[1][2]["plan_revision_translation_l2"] += 0.01
    with pytest.raises(ValueError, match="revision"):
        extract(data)


def test_logged_variance_must_replay_from_saved_clean_estimates():
    data = evidence()
    data[2][0]["diagnostics"]["variance"][2] = 30.0
    with pytest.raises(ValueError, match="variance"):
        extract(data)


def test_exhausted_horizon_is_missing_not_zero_or_last_available_action():
    events, coverage = extract(evidence(k=5))
    assert events == []
    assert coverage["horizon_exhausted"] == 1
    assert coverage["measurable_transitions"] == 0
    assert coverage["no_next_call"] == 1
    assert coverage["mean_revision_all_measurable"] is None


def test_horizon_exhaustion_rejects_fabricated_zero_logged_revision():
    data = evidence(k=5)
    data[1][5]["plan_revision_translation_l2"] = 0.0
    with pytest.raises(ValueError, match="exhausted"):
        extract(data)


def test_terminal_chunk_without_next_prediction_does_not_create_revision():
    summary, steps, chunks = evidence()
    summary.update(model_calls=1, steps=2, success=True)
    chunks = chunks[:1]
    chunks[0]["decision"]["chosen_k"] = 4  # Success truncates a selected prefix.
    events, coverage = extract((summary, steps[:2], chunks))
    assert events == []
    assert coverage["model_calls"] == coverage["no_next_call"] == 1
    assert coverage["horizon_exhausted"] == coverage["bootstrap_excluded"] == 0


def test_first_prediction_has_no_previous_plan():
    data = evidence()
    data[1][0]["plan_revision_translation_l2"] = 0.0
    with pytest.raises(ValueError, match="first prediction"):
        extract(data)


def test_bootstrap_retained_for_audit_but_marked_for_exclusion():
    events, coverage = extract(evidence(bootstrap=True))
    assert len(events) == 1 and events[0]["bootstrap"] is True
    assert coverage["bootstrap_excluded"] == 1
    assert coverage["included_nonbootstrap"] == 0
    assert coverage["mean_revision_all_measurable"] is not None
    assert coverage["mean_revision_nonbootstrap"] is None


def test_instrumented_fixed_first_chunk_is_eligible():
    events, coverage = extract(evidence(bootstrap=False))
    assert events[0]["old_call_index"] == 0 and events[0]["bootstrap"] is False
    assert coverage["bootstrap_excluded"] == 0 and coverage["included_nonbootstrap"] == 1
    assert events[0]["previous_revision"] is None
    assert events[0]["previous_revision_missing"] is True
    assert events[0]["step_fraction"] == 0
    assert events[0]["old_command_gripper"] == -1.0


def three_call_evidence():
    summary, steps, chunks = evidence()
    last = copy.deepcopy(chunks[1])
    last.update(step=4, sampling_seed=33)
    last["actions"][0][0] = 1.0
    chunks.append(last)
    steps.extend([
        {"step": 4, "replanned": True, "chunk_index": 0, "plan_revision_translation_l2": 3.0},
        {"step": 5, "replanned": False, "chunk_index": 1, "plan_revision_translation_l2": None},
    ])
    summary.update(model_calls=3, steps=6)
    return summary, steps, chunks


def test_previous_revision_uses_old_call_time_not_future_target_time():
    events, coverage = extract(three_call_evidence())
    assert len(events) == 2
    later = events[1]
    assert (later["old_step"], later["new_step"]) == (2, 4)
    assert later["previous_revision"] == pytest.approx(math.sqrt(32.0))
    assert later["previous_revision_missing"] is False
    assert later["revision_translation"] == 3.0
    assert later["step_fraction"] == 0.2
    assert later["old_command_gripper"] == 1.0
    assert coverage["included_nonbootstrap"] == 2


def test_future_target_changes_cannot_leak_into_previous_revision_feature():
    data = three_call_evidence()
    original = extract(data)[0][1]
    data[2][2]["actions"][0][0] = 2.0
    data[1][4]["plan_revision_translation_l2"] = 4.0
    changed = extract(data)[0][1]
    assert original["revision_translation"] == 3.0 and changed["revision_translation"] == 4.0
    for key in ("variance_total", "variance_translation", "previous_revision", "previous_revision_missing",
                "step_fraction", "old_command_gripper", "old_translation_norm", "k"):
        assert changed[key] == original[key]


def test_bootstrap_flag_is_from_old_not_new_call():
    data = evidence(bootstrap=False)
    data[2][1]["decision"]["bootstrap"] = True
    events, coverage = extract(data)
    assert events[0]["bootstrap"] is False
    assert coverage["included_nonbootstrap"] == 1


@pytest.mark.parametrize("old_gripper,new_gripper,flip,zero", [
    (0.0, -1.0, False, True), (0.0, 1.0, True, True),
    (1.0, 0.0, True, True), (-1.0, -0.1, False, False),
    (1.0, -1.0, True, False),
])
def test_gripper_command_threshold_and_zero_are_explicit(old_gripper, new_gripper, flip, zero):
    data = evidence()
    data[2][0]["actions"][2][6] = old_gripper
    data[2][1]["actions"][0][6] = new_gripper
    event = extract(data)[0][0]
    assert event["gripper_flip"] is flip
    assert event["gripper_zero_involved"] is zero
    assert event["gripper_old"] == old_gripper and event["gripper_new"] == new_gripper


def test_extractor_does_not_mutate_verified_evidence():
    data = evidence()
    before = copy.deepcopy(data)
    extract(data)
    assert data == before


def test_population_translation_variance_excludes_rotation_and_gripper():
    clean = diagnostic()["clean_estimates"]
    assert variance_translation(clean, 2, tail_length=2, horizon=5) == 4.0
    # Population variance uses denominator 2 here; sample variance would give 8.
    assert variance_translation(clean, 1, tail_length=2, horizon=5) == 0.0


@pytest.mark.parametrize("index", [-1, 5, True, 1.5])
def test_translation_variance_rejects_invalid_future_index(index):
    with pytest.raises(ValueError, match="index"):
        variance_translation(diagnostic()["clean_estimates"], index, tail_length=2, horizon=5)


def test_translation_variance_rejects_missing_or_nonfinite_estimates():
    clean = diagnostic()["clean_estimates"]
    with pytest.raises(ValueError, match="tail"):
        variance_translation(clean[:1], 2, tail_length=2, horizon=5)
    clean[0][2][0] = math.nan
    with pytest.raises(ValueError):
        variance_translation(clean, 2, tail_length=2, horizon=5)


def test_unknown_cohort_cannot_be_silently_pooled():
    with pytest.raises(ValueError, match="cohort"):
        extract_episode(*evidence(), "combined", horizon=5, tail_length=2)


def test_replan_marker_is_required_at_comparison_time():
    data = evidence()
    data[1][2]["replanned"] = False
    with pytest.raises(ValueError, match="marker"):
        extract(data)


def test_coverage_reconciles_call_inventory():
    _, coverage = extract(evidence())
    assert coverage["model_calls"] == (coverage["no_next_call"] + coverage["horizon_exhausted"]
                                       + coverage["bootstrap_excluded"] + coverage["included_nonbootstrap"])
