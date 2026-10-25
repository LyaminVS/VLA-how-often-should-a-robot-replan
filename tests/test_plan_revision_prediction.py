"""Leakage, grouping and weighting checks for offline prediction comparisons."""

import copy
import json
import math

import numpy as np
import pytest

from scripts.plan_revision_prediction import (
    MODEL_NAMES,
    compare_predictions,
    episode_weights,
    fit_ridge_predict,
    weighted_metrics,
)


def events(*, states=3, tasks=("task_a", "task_b"), methods=("dvac_cap_5", "dvac_cap_10"),
           seeds=(11, 22), cohort="capped", calls=4):
    result = []
    for task in tasks:
        for state in range(states):
            for method in methods:
                for seed in seeds:
                    for call in range(calls):
                        k = int(method.removeprefix("fixed_")) if method.startswith("fixed_") else 5
                        signal = (-0.8, 0.2, -0.4, 1.0)[(call + state + seed) % 4]
                        target = math.expm1(0.6 + 0.4 * signal)
                        result.append({
                            "cohort": cohort, "episode_id": f"{task}/{state}/{method}/{seed}",
                            "suite": "suite", "task": task, "state": state, "method": method,
                            "policy_seed": seed, "old_step": call * k, "new_step": (call + 1) * k,
                            "k": k, "bootstrap": False, "gripper_flip": False,
                            "variance_total": 10 ** signal, "variance_translation": 0.5 * 10 ** signal,
                            "revision_translation": target, "revision_translation_clipped": target / 2,
                            "old_translation_norm": 0.25, "step_fraction": call * k / 200,
                            "episode_max_steps": 200, "previous_revision": None,
                            "previous_revision_missing": True, "old_command_gripper": -1.0,
                        })
    return result


def test_held_out_physical_state_keeps_every_cap_seed_and_transition_together():
    rows = events()
    result, predictions = compare_predictions(rows, draws=100)
    group = result["groups"]["capped"]
    assert group["available"]
    assert len(group["folds"]) == 6
    assert len(predictions) == len(rows) * len(MODEL_NAMES)
    assert set(group["models"]) == set(MODEL_NAMES)
    for fold in group["folds"]:
        assert fold["test_state"] not in fold["train_states"]
        assert len(fold["train_states"]) == 5
        assert fold["test_support"]["episodes"] == 4
        assert fold["test_support"]["policy_seeds"] == [11, 22]
        assert fold["test_support"]["methods"] == ["dvac_cap_10", "dvac_cap_5"]
        assert not set(fold["test_episode_ids"]) & set(fold["train_episode_ids"])
        assert all(key["state"] != fold["test_state"]["state"] or key["task"] != fold["test_state"]["task"]
                   for key in fold["train_states"])
        baseline_features = fold["models"]["baseline"]["feature_names"]
        assert "log1p_k_squared" in baseline_features
        assert "method:dvac_cap_5" in baseline_features
        assert "log10_variance_total" not in baseline_features
        assert fold["models"]["baseline_plus_fm_total"]["feature_names"] == baseline_features + ["log10_variance_total"]
        assert fold["models"]["baseline"]["training_weight_mean"] == pytest.approx(1)
    comparison = group["comparisons"][0]
    assert comparison["mae_difference"]["value"] < 0
    assert comparison["percent_mae_improvement"]["value"] > 50
    assert comparison["state_wins"] + comparison["state_losses"] + comparison["state_ties"] == 6
    assert len(comparison["by_method"]) == len(comparison["by_task"]) == 2
    assert comparison["mae_difference"]["valid_bootstrap_draws"] == 100
    assert "no model refitting" in comparison["mae_difference"]["interval_note"]
    json.dumps(result, allow_nan=False)
    json.dumps(predictions, allow_nan=False)


def test_changing_held_out_targets_cannot_change_its_predictions_or_training_audit():
    original = events(states=2, methods=("dvac_cap_5",), seeds=(11,))
    changed = copy.deepcopy(original)
    for row in changed:
        if row["task"] == "task_a" and row["state"] == 0:
            row["revision_translation"] *= 7
    first, predictions_first = compare_predictions(original, draws=100)
    second, predictions_second = compare_predictions(changed, draws=100)
    target_fold = "capped|suite/task_a/state0"
    first_fold = next(fold for fold in first["groups"]["capped"]["folds"] if fold["fold_id"] == target_fold)
    second_fold = next(fold for fold in second["groups"]["capped"]["folds"] if fold["fold_id"] == target_fold)
    assert first_fold == second_fold
    assert [row["prediction"] for row in predictions_first if row["fold_id"] == target_fold] == [
        row["prediction"] for row in predictions_second if row["fold_id"] == target_fold]


def test_standardization_is_training_only_and_intercept_is_unpenalized():
    train = np.asarray([[0.0], [1.0], [2.0]])
    prediction, audit = fit_ridge_predict(train, np.array([0.0, 1.0, 2.0]), np.ones(3), np.array([[3.0]]))
    _, other_audit = fit_ridge_predict(train, np.array([0.0, 1.0, 2.0]), np.ones(3), np.array([[10000.0]]))
    assert audit == other_audit
    assert audit["train_feature_mean"] == [1.0]
    assert audit["train_feature_scale"][0] == pytest.approx(math.sqrt(2 / 3))
    assert prediction[0] == pytest.approx(2.5)
    constant, _ = fit_ridge_predict(np.ones((3, 2)), np.full(3, 7.0), np.ones(3), np.ones((1, 2)))
    assert constant[0] == pytest.approx(7)


def test_episode_weights_and_metrics_do_not_let_long_episodes_dominate():
    rows = [{"episode_id": "short"}] + [{"episode_id": "long"}] * 10
    weights = episode_weights(rows)
    assert weights[0] == pytest.approx(weights[1:].sum())
    normalized = episode_weights(rows, normalize_mean=True)
    assert normalized.mean() == pytest.approx(1)
    assert normalized[0] == pytest.approx(normalized[1:].sum())
    metrics = weighted_metrics([0] + [10] * 10, [2] + [10] * 10, weights)
    assert metrics["mae"] == pytest.approx(1)
    assert metrics["rmse"] == pytest.approx(math.sqrt(2))
    assert metrics["r2"] == pytest.approx(0.92)


def test_groups_stay_separate_and_only_old_adaptive_bootstrap_is_excluded():
    adaptive = events(states=2, methods=("dvac_cap_5",), seeds=(11,))
    for row in adaptive:
        row["bootstrap"] = row["old_step"] == 0
    fixed = events(states=2, methods=("fixed_5", "fixed_10"), seeds=(11,), cohort="supplemental")
    supplemental = events(states=2, methods=("dvac",), seeds=(11,), cohort="supplemental")
    result, predictions = compare_predictions(adaptive + fixed + supplemental, draws=100)
    assert result["groups"]["capped"]["excluded_bootstrap_events"] == 4
    assert result["groups"]["supplemental_fixed"]["excluded_bootstrap_events"] == 0
    assert result["groups"]["supplemental_fixed"]["nonbootstrap_support"]["events"] == len(fixed)
    assert result["groups"]["supplemental_dvac"]["nonbootstrap_support"]["events"] == len(supplemental)
    assert not any(row["group"] == "capped" and row["old_step"] == 0 for row in predictions)
    assert any(row["group"] == "supplemental_fixed" and row["old_step"] == 0 for row in predictions)


def test_one_physical_state_is_not_converted_to_independent_seed_or_cap_folds():
    result, predictions = compare_predictions(events(states=1, tasks=("task_a",)), draws=100)
    assert result["groups"]["capped"]["available"] is False
    assert result["groups"]["capped"]["nonbootstrap_support"]["episodes"] == 4
    assert predictions == []
    assert result["groups"]["supplemental_fixed"]["available"] is False


def test_zero_variance_and_constant_target_remain_finite_with_null_r_squared():
    rows = events(states=2, methods=("dvac_cap_5",), seeds=(11,))
    for row in rows:
        row.update(variance_total=0.0, variance_translation=0.0, revision_translation=0.0)
    result, predictions = compare_predictions(rows, draws=100)
    for metrics in result["groups"]["capped"]["models"].values():
        assert metrics == {"mae": 0.0, "rmse": 0.0, "r2": None}
    for comparison in result["groups"]["capped"]["comparisons"]:
        assert comparison["percent_mae_improvement"]["value"] is None
        assert comparison["percent_mae_improvement"]["interval_available"] is False
    assert all(row["prediction"] == 0 for row in predictions)


def test_seed_and_event_order_are_reproducible():
    rows = events(states=2, methods=("dvac_cap_5",), seeds=(11,))
    first = compare_predictions(rows, draws=100, seed=7)
    second = compare_predictions(list(reversed(rows)), draws=100, seed=7)
    assert first == second


@pytest.mark.parametrize("changes", [
    {"step_fraction": -1}, {"step_fraction": 0.3}, {"old_translation_norm": float("nan")},
    {"previous_revision_missing": 1}, {"previous_revision_missing": False, "previous_revision": None},
    {"previous_revision_missing": True, "previous_revision": 1.0}, {"old_command_gripper": float("inf")},
])
def test_invalid_or_inconsistent_prospective_predictors_fail(changes):
    rows = events(states=2, methods=("dvac_cap_5",), seeds=(11,))
    rows[0].update(changes)
    with pytest.raises(ValueError):
        compare_predictions(rows, draws=100)
