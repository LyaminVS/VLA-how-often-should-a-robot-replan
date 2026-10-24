"""Paired state resampling and reuse-aware reporting safeguards."""

import copy

import pytest

from scripts.analyze_capped import METHODS, cluster_statistics, frontier, paired_layout


def records(methods=("fixed_10", "dvac_cap_10"), states=(0, 1, 2), seeds=(11,)):
    result = []
    for task in ("task_a", "task_b"):
        for state in states:
            for seed in seeds:
                for method in methods:
                    result.append({"summary": {
                        "suite": "fixture_suite", "task": task, "state": state, "policy_seed": seed,
                        "method": method, "success": state % 2 == 0, "model_calls": 1 + state,
                        "steps": 10 * (1 + state), "eef_path_m": float(state), "gripper_close_commands": 1,
                    }})
    return result


def test_shared_state_bootstrap_preserves_exact_paired_equality():
    result = cluster_statistics(records(), methods=("fixed_10", "dvac_cap_10"), draws=100)
    contrast = result["contrasts"][0]
    assert contrast["designated_contrast"] is True
    assert contrast["success_difference"] == contrast["success_ci_low"] == contrast["success_ci_high"] == 0
    assert contrast["model_calls_difference"] == contrast["model_calls_ci_low"] == contrast["model_calls_ci_high"] == 0
    assert result["bootstrap"]["physical_settings"] == 6
    assert all(row["calls_per_100_actions_mean"] == 10 for row in result["methods"])


def test_duplicating_identical_policy_seeds_does_not_manufacture_state_precision():
    one = cluster_statistics(records(seeds=(11,)), methods=("fixed_10", "dvac_cap_10"), draws=100)
    two = cluster_statistics(records(seeds=(11, 22)), methods=("fixed_10", "dvac_cap_10"), draws=100)
    for original, duplicate in zip(one["methods"], two["methods"], strict=True):
        assert duplicate["episodes"] == 2 * original["episodes"]
        assert duplicate["physical_settings"] == original["physical_settings"]
        assert duplicate["success_ci_low"] == original["success_ci_low"]
        assert duplicate["success_ci_high"] == original["success_ci_high"]
        assert duplicate["model_calls_ci_low"] == original["model_calls_ci_low"]
        assert duplicate["model_calls_ci_high"] == original["model_calls_ci_high"]


def test_cost_ratio_uses_ratio_of_totals_not_average_per_episode_ratios():
    data = records(states=(0, 1))
    for record in data:
        row = record["summary"]
        row["model_calls"] = 1
        row["steps"] = 1 if row["state"] == 0 else 9
    result = cluster_statistics(data, methods=("fixed_10", "dvac_cap_10"), draws=100)
    assert all(row["calls_per_100_actions_mean"] == 20 for row in result["methods"])


def test_tasks_receive_equal_weight_even_with_different_numbers_of_states():
    data = records(states=(0, 1, 2, 3))
    data = [record for record in data if record["summary"]["task"] == "task_b" or record["summary"]["state"] < 2]
    for record in data:
        record["summary"]["success"] = record["summary"]["task"] == "task_a"
    result = cluster_statistics(data, methods=("fixed_10", "dvac_cap_10"), draws=100)
    assert all(row["success_mean"] == .5 for row in result["methods"])


@pytest.mark.parametrize("mutation,message", [
    (lambda data: data.pop(), "Incomplete paired"),
    (lambda data: data.append(copy.deepcopy(data[0])), "Duplicate"),
    (lambda data: data[0]["summary"].update(method="unexpected"), "unexpected"),
    (lambda data: data[0]["summary"].update(success=1), "boolean"),
    (lambda data: data[0]["summary"].update(model_calls=float("nan")), "Invalid metric"),
])
def test_incomplete_duplicate_or_invalid_measurements_are_rejected(mutation, message):
    data = records()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        paired_layout(data, ("fixed_10", "dvac_cap_10"))


def test_all_nine_methods_have_thirty_six_oriented_comparisons():
    result = cluster_statistics(records(methods=METHODS), draws=100)
    assert len(result["contrasts"]) == 36
    designated = {(row["method"], row["reference"]) for row in result["contrasts"] if row["designated_contrast"]}
    assert designated == {("dvac_cap_10", "fixed_10"), ("dvac_cap_10", "dvac_cap_50")}
    assert len(result["outcome_matrix"]) == 6
    assert all(method + "_success" in result["outcome_matrix"][0] for method in METHODS)


def test_frontier_requires_at_least_one_strict_improvement_and_is_descriptive():
    data = [{"method": "a", "success_mean": .8, "model_calls_mean": 10},
            {"method": "b", "success_mean": .8, "model_calls_mean": 10},
            {"method": "c", "success_mean": .7, "model_calls_mean": 12},
            {"method": "d", "success_mean": .9, "model_calls_mean": 20}]
    result = {row["method"]: row for row in frontier(data)}
    assert result["a"]["observed_nondominated"] is True
    assert result["b"]["observed_nondominated"] is True
    assert result["c"]["observed_nondominated"] is False
    assert result["d"]["observed_nondominated"] is True
    assert result["c"]["dominated_by_observed_means"] == "a,b"


def test_stats_do_not_emit_cross_cohort_timing_comparisons():
    result = cluster_statistics(records(methods=METHODS), draws=100)
    assert all(not any("wall" in key or "inference" in key for key in row) for row in result["methods"])
    assert all(not any("wall" in key or "inference" in key for key in row) for row in result["contrasts"])
