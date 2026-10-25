"""Paired AutoHorizon comparison and immutable retained-result safeguards."""

import copy
import json

import pytest

from scripts.analyze_autohorizon import (
    AUTO_METHODS, EXISTING_METHODS, METHODS, PRIMARY_CONTRASTS, TASKS,
    comparison_statistics, horizon_behavior, load_existing, require_complete_layout,
)
from scripts.analyze_capped import cluster_statistics, sha256, write_csv


def fixture_records(methods=METHODS):
    records = []
    for suite, task in sorted(TASKS):
        for state in range(10):
            for method in methods:
                cap = int(method.rsplit("_", 1)[1])
                selected = min(cap, 3)
                summary = {
                    "episode_id": f"{suite}/{task}/{state}/{method}",
                    "suite": suite, "task": task, "state": state, "policy_seed": 11,
                    "method": method, "success": state % 3 != 0,
                    "model_calls": 2, "steps": 2 * selected,
                    "eef_path_m": .1 * state, "gripper_close_commands": 1,
                    "execution_cap": cap if not method.startswith("fixed_") else None,
                    "k": cap if method.startswith("fixed_") else None,
                    "inference_s": 1.0, "selector_s": .1, "simulation_s": 2.0,
                    "compute_s": 3.1, "episode_wall_s": 4.0,
                }
                chunks = [{"step": index * selected, "executed_steps": selected,
                           "decision": {"chosen_k": selected, "raw_chosen_k": 3,
                                        "execution_cap": cap, "bootstrap": False}}
                          for index in range(2)]
                records.append({"summary": summary, "chunks": chunks})
    return records


def split(data):
    return ([r for r in data if r["summary"]["method"] in EXISTING_METHODS],
            [r for r in data if r["summary"]["method"] in AUTO_METHODS])


def test_all_thirteen_methods_are_paired_on_thirty_starts_and_eight_contrasts_are_designated():
    result = comparison_statistics(*split(fixture_records()), draws=100)
    assert len(result["methods"]) == 13
    assert len(result["contrasts"]) == 78
    assert len(result["outcome_matrix"]) == 30
    primary = [row for row in result["contrasts"] if row["designated_contrast"]]
    assert {(row["method"], row["reference"]) for row in primary} == set(PRIMARY_CONTRASTS)
    assert all(row["physical_settings"] == row["paired_seed_runs"] == 30 for row in primary)
    assert all(row["comparison_role"] == "exploratory" for row in result["contrasts"] if not row["designated_contrast"])
    assert result["bootstrap"]["physical_settings"] == 30


def test_exact_paired_equality_has_zero_difference_interval_without_duplicating_reference_runs():
    result = comparison_statistics(*split(fixture_records()), draws=100)
    for row in result["contrasts"]:
        assert row["success_difference"] == row["success_ci_low"] == row["success_ci_high"] == 0
        assert row["model_calls_difference"] == row["model_calls_ci_low"] == row["model_calls_ci_high"] == 0
    assert all(row["episodes"] == 30 for row in result["methods"])


def test_combining_methods_preserves_original_point_estimates_and_state_intervals():
    existing, new = split(fixture_records())
    old = cluster_statistics(existing, methods=EXISTING_METHODS, draws=100)
    combined = comparison_statistics(existing, new, draws=100)
    actual = {row["method"]: row for row in combined["methods"]}
    for expected in old["methods"]:
        assert actual[expected["method"]] == expected


def test_calls_per_action_is_ratio_of_totals():
    data = fixture_records()
    for record in data:
        row = record["summary"]
        row["model_calls"] = 1
        row["steps"] = 1 if row["state"] < 5 else 9
    result = comparison_statistics(*split(data), draws=100)
    assert all(row["calls_per_100_actions_mean"] == 20 for row in result["methods"])


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows.pop(), "Incomplete paired"),
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "Duplicate"),
    (lambda rows: rows[0]["summary"].update(state=15), "Incomplete paired|Expected"),
    (lambda rows: rows[0]["summary"].update(policy_seed=22), "seed|paired"),
    (lambda rows: rows[0]["summary"].update(method="autohorizon"), "unexpected"),
    (lambda rows: rows[0]["summary"].update(success=1), "boolean"),
    (lambda rows: rows[0]["summary"].update(model_calls=float("nan")), "Invalid metric"),
    (lambda rows: rows[0]["summary"].update(execution_cap=50), "execution cap"),
])
def test_invalid_or_incomplete_auto_grid_is_rejected(mutation, message):
    old, new = split(fixture_records())
    mutation(new)
    with pytest.raises(ValueError, match=message):
        comparison_statistics(old, new, draws=100)


def test_wrong_tasks_and_entirely_wrong_seed_grid_are_rejected():
    rows = fixture_records(AUTO_METHODS)
    for record in rows:
        record["summary"]["policy_seed"] = 22
    with pytest.raises(ValueError, match="policy seed 11"):
        require_complete_layout(rows, AUTO_METHODS)
    for record in rows:
        record["summary"]["policy_seed"] = 11
        record["summary"]["suite"] = "wrong_suite"
    with pytest.raises(ValueError, match="three tasks"):
        require_complete_layout(rows, AUTO_METHODS)


def test_timing_is_not_compared_across_instrumentation_cohorts():
    result = comparison_statistics(*split(fixture_records()), draws=100)
    assert not any("wall" in key or "inference" in key for row in result["methods"] for key in row)
    assert not any("wall" in key or "inference" in key for row in result["contrasts"] for key in row)


def test_selected_horizon_is_distinct_from_terminal_truncation_and_first_call_is_not_forced_to_one():
    rows = fixture_records(AUTO_METHODS)
    rows[0]["chunks"][-1]["executed_steps"] = 1
    rows[0]["summary"]["steps"] = 4
    behavior, histogram, decisions, timing = horizon_behavior(rows)
    first = behavior[0]
    assert first["first_call_chosen_k_mean"] == 3
    assert first["chosen_k_mean_per_call"] == 3
    assert first["executed_prefix_mean_per_call"] < 3
    assert first["cap_applied_fraction"] == 0
    assert sum(row["calls"] for row in histogram) == 240
    assert len(decisions) == 240
    assert all(row["episode_wall_s_mean_per_episode"] == 4 for row in timing)


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows[0]["chunks"][0]["decision"].update(bootstrap=True), "forced bootstrap"),
    (lambda rows: rows[0]["chunks"][0]["decision"].update(chosen_k=50), "prefix"),
    (lambda rows: rows[0]["chunks"][0]["decision"].update(raw_chosen_k=100), "prefix"),
    (lambda rows: rows[0]["chunks"][0].update(executed_steps=4), "prefix"),
    (lambda rows: rows[0]["summary"].update(steps=20), "episode length"),
    (lambda rows: rows[0]["summary"].update(model_calls=3), "Chunk count"),
    (lambda rows: rows[0]["summary"].update(episode_wall_s=float("inf")), "runtime metric"),
])
def test_horizon_and_timing_inconsistencies_fail_closed(mutation, message):
    rows = fixture_records(AUTO_METHODS)
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        horizon_behavior(rows)


def write_existing_fixture(directory):
    records = fixture_records(EXISTING_METHODS)
    stats = cluster_statistics(records, methods=EXISTING_METHODS, draws=100)
    stats.update(complete=True, episodes=270, reused_fixed_episodes=150, capped_episodes=120,
                 verification={"complete": True}, config_fingerprint="test-config")
    (directory / "analysis.json").write_text(json.dumps(stats))
    write_csv(directory / "episodes.csv", [row["summary"] for row in records])
    refresh_artifacts(directory)
    return records, stats


def refresh_artifacts(directory):
    (directory / "artifacts.json").write_text(json.dumps({
        "statistics_sha256": sha256(directory / "analysis.json"), "config_fingerprint": "test-config",
        "files": {name: sha256(directory / name) for name in ("analysis.json", "episodes.csv")},
    }))


def test_existing_full_episode_export_is_hash_bound_and_agrees_with_analysis(tmp_path):
    records, stats = write_existing_fixture(tmp_path)
    actual, result, provenance = load_existing(tmp_path / "analysis.json")
    assert len(actual) == len(records) == 270
    assert result == stats
    assert provenance["episodes_csv_sha256"] == sha256(tmp_path / "episodes.csv")
    with (tmp_path / "episodes.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_existing(tmp_path / "analysis.json")


def test_rehashed_but_inconsistent_retained_counts_are_rejected(tmp_path):
    _, stats = write_existing_fixture(tmp_path)
    stats["methods"][0]["successes"] += 1
    (tmp_path / "analysis.json").write_text(json.dumps(stats))
    refresh_artifacts(tmp_path)
    with pytest.raises(ValueError, match="disagree on outcomes"):
        load_existing(tmp_path / "analysis.json")


def test_rehashed_but_inconsistent_retained_means_are_rejected(tmp_path):
    _, stats = write_existing_fixture(tmp_path)
    stats["methods"][0]["model_calls_mean"] += 1
    (tmp_path / "analysis.json").write_text(json.dumps(stats))
    refresh_artifacts(tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        load_existing(tmp_path / "analysis.json")
