"""Paired disturbance comparisons and exclusion of incomplete method panels."""

import copy
import json

import pytest

from scripts.analyze_action_dropout import (
    METHODS, PROBABILITIES, TASKS, load_existing_export, panel_inventory, statistics,
)
from scripts.analyze_capped import sha256, write_csv


def records(methods=("fixed_5", "dvac_cap_5"), probabilities=PROBABILITIES):
    result = []
    for suite, task in TASKS:
        for state in range(10):
            for method in METHODS:
                selected_probabilities = probabilities if method in methods else (0.0,)
                for probability in selected_probabilities:
                    result.append({"summary": {
                        "episode_id": f"{suite}/{task}/{state}/{method}/{probability}",
                        "suite": suite, "task": task, "state": state, "policy_seed": 11,
                        "method": method, "dropout_p": probability,
                        "success": state < 7, "steps": 10 * (state + 1), "model_calls": state + 1,
                        "translation_drop_count": round(probability * 10 * (state + 1)),
                    }})
    return result


def test_balanced_method_panels_include_all_thirty_starts_at_every_probability():
    result = statistics(records(), draws=100)
    assert result["balanced_methods"] == ["fixed_5", "dvac_cap_5"]
    assert result["balanced_new_episodes"] == 120
    assert result["balanced_baseline_episodes"] == 60
    assert result["balanced_total_episodes"] == 180
    assert len(result["methods"]) == 6
    assert len(result["by_task"]) == 18
    assert len(result["degradation"]) == 4
    assert result["probabilities"] == [0.0, 0.1, 0.3]
    assert all(row["new_expected_episodes"] == 60 for row in result["method_panels"])
    assert all(row["episodes"] == row["physical_settings"] == 30 for row in result["methods"])
    assert all(row["episodes"] == 10 for row in result["by_task"])
    assert all(row["paired_physical_settings"] == 30 for row in result["degradation"])


def test_success_call_mean_excludes_failures_and_pools_successful_episodes():
    data = records(methods=("fixed_5",))
    success_counts = {task: index + 1 for index, (_, task) in enumerate(TASKS)}
    for record in data:
        row = record["summary"]
        row["success"] = row["state"] < success_counts[row["task"]]
    result = statistics(data, draws=100)
    for row in result["methods"]:
        assert row["model_calls_success_n"] == row["successes"] == 6
        assert row["model_calls_success_mean"] == pytest.approx(10 / 6)
        assert row["model_calls_mean"] == pytest.approx(5.5)
        assert row["episodes"] == 30
    for row in result["by_task"]:
        count = success_counts[row["task"]]
        assert row["model_calls_success_n"] == row["successes"] == count
        assert row["model_calls_success_mean"] == pytest.approx((count + 1) / 2)
        assert row["model_calls_mean"] == pytest.approx(5.5)
        assert row["episodes"] == 10


def test_success_call_mean_is_null_when_no_episode_succeeds():
    data = records(methods=("fixed_5",))
    for record in data:
        if record["summary"]["dropout_p"] == .3:
            record["summary"]["success"] = False
    result = statistics(data, draws=100)
    for row in result["methods"] + result["by_task"]:
        if row["dropout_p"] == .3:
            assert row["model_calls_success_n"] == row["successes"] == 0
            assert row["model_calls_success_mean"] is None
            assert row["model_calls_mean"] == pytest.approx(5.5)
        else:
            assert row["model_calls_success_n"] > 0
            assert row["model_calls_success_mean"] == pytest.approx(4)
    assert '"model_calls_success_mean": null' in json.dumps(result, allow_nan=False)


def test_identical_matched_outcomes_and_calls_have_exact_zero_difference_intervals():
    result = statistics(records(), draws=100)
    for row in result["degradation"]:
        assert row["success_difference"] == row["success_ci_low"] == row["success_ci_high"] == 0
        assert row["model_calls_difference"] == row["model_calls_ci_low"] == row["model_calls_ci_high"] == 0
        assert row["both_success"] == 21
        assert row["both_failure"] == 9
        assert row["method_only_success"] == row["reference_only_success"] == 0
    assert len(result["matched_method_contrasts"]) == 3
    assert all(row["success_difference"] == 0 for row in result["matched_method_contrasts"])


def test_missing_one_probability_cell_excludes_entire_method_from_comparisons():
    data = records()
    data.remove(next(row for row in data if row["summary"]["method"] == "dvac_cap_5"
                     and row["summary"]["dropout_p"] == .3))
    result = statistics(data, draws=100)
    assert result["balanced_methods"] == ["fixed_5"]
    assert result["completed_new_episodes"] == 119
    assert result["excluded_partial_new_episodes"] == 59
    assert all(row["method"] == "fixed_5" for row in result["methods"])
    partial = next(row for row in result["method_panels"] if row["method"] == "dvac_cap_5")
    assert partial["completed_by_probability"] == {"p10": 30, "p30": 29}
    assert len(partial["missing"]) == 1
    assert result["matched_method_contrasts"] == []


def test_baseline_only_is_inventory_and_does_not_fabricate_disturbance_results():
    result = statistics(records(methods=()), draws=100)
    assert result["balanced_methods"] == []
    assert result["methods"] == result["degradation"] == []
    assert result["completed_new_episodes"] == 0
    assert "No complete" in result["analysis_status"]


def test_degradation_direction_and_discordant_counts_use_own_baseline():
    data = records(methods=("fixed_5",))
    for record in data:
        row = record["summary"]
        if row["dropout_p"] == .1:
            row["success"] = row["state"] < 5
        elif row["dropout_p"] == .3:
            row["success"] = row["state"] < 9
    result = statistics(data, draws=100)
    comparisons = {row["dropout_p"]: row for row in result["degradation"]}
    assert comparisons[.1]["success_difference"] == pytest.approx(-.2)
    assert comparisons[.1]["reference_only_success"] == 6
    assert comparisons[.1]["method_only_success"] == 0
    assert comparisons[.3]["success_difference"] == pytest.approx(.2)
    assert comparisons[.3]["method_only_success"] == 6


def test_realized_dropout_and_calls_per_action_are_ratios_of_totals():
    data = records(methods=("fixed_5",))
    for record in data:
        row = record["summary"]
        row["steps"] = 1 if row["state"] < 5 else 9
        row["model_calls"] = 1
        row["translation_drop_count"] = 1 if row["dropout_p"] else 0
    result = statistics(data, draws=100)
    row = next(row for row in result["methods"] if row["dropout_p"] == .1)
    assert row["calls_per_100_actions_mean"] == 20
    assert row["realized_drop_fraction_mean"] == pytest.approx(.2)
    assert row["drop_fraction_per_episode_mean"] == pytest.approx((1 + 1 / 9) / 2)
    assert row["steps_mean"] == 5


def test_all_nine_method_panels_have_540_new_and_270_retained_episodes():
    result = statistics(records(methods=METHODS), draws=100)
    assert result["all_nine_methods_complete"] is True
    assert result["balanced_new_episodes"] == 540
    assert result["balanced_baseline_episodes"] == 270
    assert result["balanced_total_episodes"] == 810
    assert len(result["methods"]) == 27
    assert len(result["degradation"]) == 18
    assert len(result["matched_method_contrasts"]) == 12


def test_eight_home_methods_exclude_missing_k1_and_keep_matched_panels():
    methods = tuple(method for method in METHODS if method != "fixed_1")
    result = statistics(records(methods=methods), draws=100)
    assert result["balanced_methods"] == list(methods)
    assert result["balanced_new_episodes"] == 480
    assert result["balanced_baseline_episodes"] == 240
    assert result["balanced_total_episodes"] == 720
    assert result["retained_baseline_episodes"] == 270
    assert result["all_nine_methods_complete"] is False
    assert len(result["matched_method_contrasts"]) == 12
    assert all(row["method"] != "fixed_1" for row in result["methods"])


def test_probability_grid_is_explicit_and_can_follow_a_different_registered_config():
    probabilities = (0.0, .05, .2, .4)
    data = records(probabilities=probabilities)
    with pytest.raises(ValueError, match="probability"):
        statistics(data, draws=100)
    result = statistics(data, probabilities=probabilities, draws=100)
    assert result["probabilities"] == list(probabilities)
    assert result["balanced_new_episodes"] == 180
    assert len(result["methods"]) == 8
    assert len(result["degradation"]) == 6


@pytest.mark.parametrize("probabilities", [(0.0,), (.1, .3), (0.0, .1, .1), (0.0, .3, .1), (0.0, .125), (0.0, True)])
def test_invalid_analysis_probability_grids_fail(probabilities):
    with pytest.raises(ValueError, match="[Pp]robabilit"):
        statistics(records(), probabilities=probabilities, draws=100)


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "Duplicate"),
    (lambda rows: rows.pop(0), "270 retained"),
    (lambda rows: rows[0]["summary"].update(state=15), "states"),
    (lambda rows: rows[0]["summary"].update(policy_seed=22), "seed"),
    (lambda rows: rows[0]["summary"].update(success=1), "boolean"),
    (lambda rows: rows[0]["summary"].update(model_calls=0), "positive integer"),
    (lambda rows: rows[0]["summary"].update(steps=float("nan")), "positive integer"),
    (lambda rows: rows[0]["summary"].update(dropout_p=.4), "probability"),
    (lambda rows: rows[0]["summary"].update(translation_drop_count=1), "p=0"),
    (lambda rows: rows[0]["summary"].update(translation_drop_count=10000), "drop count"),
    (lambda rows: rows[0]["summary"].update(method="autohorizon_cap_5"), "method"),
])
def test_invalid_or_duplicate_measurements_fail_closed(mutation, message):
    data = records()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        panel_inventory(data)


def test_bootstrap_deterministic_and_preserves_state_pairing():
    first = statistics(records(), draws=100, seed=7)
    second = statistics(records(), draws=100, seed=7)
    assert first == second
    fixed = [row for row in first["methods"] if row["method"] == "fixed_5"]
    assert len({row["success_ci_low"] for row in fixed}) == 1
    assert len({row["success_ci_high"] for row in fixed}) == 1


def baseline_export(directory):
    rows = [r["summary"] for r in records(methods=())]
    write_csv(directory / "episodes.csv", rows)
    methods = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        methods.append({"method": method, "successes": sum(row["success"] for row in selected),
                        **{metric + "_mean": sum(float(row[metric]) for row in selected) / 30
                           for metric in ("success", "model_calls", "steps")}})
    data = {"complete": True, "episodes": 270, "verification": {"complete": True},
            "config_fingerprint": "test-config", "methods": methods}
    (directory / "analysis.json").write_text(json.dumps(data))
    refresh_hashes(directory)
    return data


def refresh_hashes(directory):
    (directory / "artifacts.json").write_text(json.dumps({
        "statistics_sha256": sha256(directory / "analysis.json"),
        "files": {name: sha256(directory / name) for name in ("analysis.json", "episodes.csv")},
    }))


def test_baseline_export_hash_and_totals_are_checked(tmp_path):
    data = baseline_export(tmp_path)
    rows, provenance = load_existing_export(tmp_path / "analysis.json")
    assert len(rows) == 270
    assert provenance["config_fingerprint"] == "test-config"
    data["methods"][0]["successes"] += 1
    (tmp_path / "analysis.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_existing_export(tmp_path / "analysis.json")
    refresh_hashes(tmp_path)
    with pytest.raises(ValueError, match="totals disagree"):
        load_existing_export(tmp_path / "analysis.json")
