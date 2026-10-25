"""Repeated-call dependence, exact-K conditioning and missing-data safeguards."""

import copy

import pytest

from scripts.plan_revision_statistics import (
    analyze_statistics, association_estimates, eligible_strata, validate_events,
    weighted_correlation, weighted_midranks,
)


def events(*, states=3, calls=4, methods=("dvac_cap_5",), cohort="capped"):
    result = []
    for task in ("task_a", "task_b"):
        for state in range(states):
            for method in methods:
                for call in range(calls):
                    value = float(1 + state * calls + call)
                    result.append({
                        "cohort": cohort, "episode_id": f"{task}/{state}/{method}", "suite": "suite",
                        "task": task, "state": state, "policy_seed": 11, "method": method,
                        "old_step": call * 5, "new_step": (call + 1) * 5, "k": 5, "bootstrap": False,
                        "variance_total": value, "variance_translation": value / 2,
                        "revision_translation": 2 * value, "revision_translation_clipped": value / 10,
                        "gripper_flip": False,
                    })
    return result


def test_weighted_midranks_respect_ties_and_integer_replication():
    assert weighted_midranks([1, 2, 2, 3], [1, 1, 1, 1]).tolist() == [.125, .5, .5, .875]
    assert weighted_midranks([1, 2, 3], [1, 2, 1]).tolist() == [.125, .5, .875]
    assert weighted_correlation([1, 2, 3], [3, 2, 1], [1, 2, 1]) == pytest.approx(-1)


def test_perfect_monotonic_association_and_six_physical_clusters():
    result = analyze_statistics(events(), draws=100)
    group = result["groups"]["capped"]
    assert group["nonbootstrap_support"]["events"] == 24
    assert group["nonbootstrap_support"]["physical_settings"] == 6
    for association in group["associations"]:
        assert association["adjusted_support"]["eligible_strata"] == 2
        for metric in ("adjusted_rank_correlation", "marginal_spearman", "episode_weighted_marginal_spearman"):
            assert association[metric]["value"] == pytest.approx(1)
            assert association[metric]["ci_low"] == pytest.approx(1)
            assert association[metric]["valid_bootstrap_draws"] == 100


def test_bootstrap_keeps_duplicate_caps_in_same_physical_state_and_does_not_manufacture_precision():
    rows = events()
    for index, row in enumerate(rows):
        row["revision_translation"] = float(1 + (index * 7) % 11)
    repeated = copy.deepcopy(rows)
    for row in repeated:
        row["method"] = "dvac_cap_10"
        row["episode_id"] = row["episode_id"].replace("dvac_cap_5", "dvac_cap_10")
    one = analyze_statistics(rows, draws=100)
    two = analyze_statistics(rows + repeated, draws=100)
    a = one["groups"]["capped"]["associations"][0]
    b = two["groups"]["capped"]["associations"][0]
    assert b["adjusted_support"]["events"] == 2 * a["adjusted_support"]["events"]
    assert b["adjusted_support"]["physical_settings"] == a["adjusted_support"]["physical_settings"]
    for metric in ("adjusted_rank_correlation", "marginal_spearman"):
        assert b[metric]["ci_low"] == pytest.approx(a[metric]["ci_low"])
        assert b[metric]["ci_high"] == pytest.approx(a[metric]["ci_high"])


def test_exact_k_conditioning_removes_between_interval_association():
    rows = events(states=4, calls=4)
    # Strong positive relation between K strata, but negative relation within K.
    for row in rows:
        if row["state"] >= 2:
            row["k"] = 10
            row["new_step"] = row["old_step"] + 10
        offset = 100 if row["k"] == 10 else 0
        position = row["old_step"] / 5 + (row["state"] % 2) * 4
        row["variance_total"] = offset + position + 1
        row["revision_translation"] = offset + 9 - position
    eligible, table = eligible_strata(rows, "variance_total", "revision_translation")
    stats = association_estimates(rows, "variance_total", "revision_translation", eligible)
    assert len(eligible) == 4
    assert stats["marginal_spearman"] > .4
    assert stats["adjusted_rank_correlation"] == pytest.approx(-1)
    assert all(row["physical_settings"] == 2 for row in table)


def test_all_calls_in_each_episode_have_equal_total_weight_even_when_episode_length_differs():
    rows = events(states=3, calls=4)
    for index, row in enumerate(rows):
        row["revision_translation"] = float(1 + (index * 7) % 11)
    eligible, _ = eligible_strata(rows, "variance_total", "revision_translation")
    base = association_estimates(rows, "variance_total", "revision_translation", eligible)
    # Repeat every event in one episode without adding another independent start.
    repeats = []
    for row in rows:
        if row["episode_id"] == "task_a/0/dvac_cap_5":
            for repeat in range(6):
                duplicate = dict(row)
                duplicate["old_step"] += 100 + repeat * 30
                duplicate["new_step"] = duplicate["old_step"] + 5
                repeats.append(duplicate)
    augmented = association_estimates(rows + repeats, "variance_total", "revision_translation", eligible)
    assert augmented["episode_weighted_marginal_spearman"] == pytest.approx(base["episode_weighted_marginal_spearman"])
    assert augmented["adjusted_rank_correlation"] == pytest.approx(base["adjusted_rank_correlation"])


def test_fixed_first_call_retained_but_old_dvac_bootstrap_excluded_and_cohorts_separate():
    adaptive = events()
    for row in adaptive:
        row["bootstrap"] = row["old_step"] == 0
    fixed = events(states=2, methods=("fixed_5",), cohort="supplemental")
    supplemental_dvac = events(states=2, methods=("dvac",), cohort="supplemental")
    result = analyze_statistics(adaptive + fixed + supplemental_dvac, draws=100)
    assert result["groups"]["capped"]["excluded_bootstrap_events"] == 6
    assert result["groups"]["supplemental_fixed"]["excluded_bootstrap_events"] == 0
    assert result["groups"]["supplemental_fixed"]["nonbootstrap_support"]["events"] == 16
    assert result["groups"]["supplemental_fixed"]["associations"][0]["adjusted_support"]["eligible_strata"] == 2
    assert result["groups"]["supplemental_dvac"]["nonbootstrap_support"]["events"] == 16


def test_single_physical_state_per_stratum_is_not_pseudo_replicated():
    result = analyze_statistics(events(states=1, calls=20), draws=100)
    association = result["groups"]["capped"]["associations"][0]
    assert association["adjusted_rank_correlation"]["value"] is None
    assert association["adjusted_rank_correlation"]["ci_low"] is None
    assert association["adjusted_support"]["events"] == 0
    assert association["adjusted_support"]["excluded_events"] == 40


def test_constant_response_and_empty_groups_return_null_instead_of_zero():
    rows = events()
    for row in rows:
        row["revision_translation"] = 2
    result = analyze_statistics(rows, draws=100)
    association = result["groups"]["capped"]["associations"][0]
    assert association["marginal_spearman"]["value"] is None
    assert association["adjusted_rank_correlation"]["valid_bootstrap_draws"] == 0
    empty = result["groups"]["supplemental_fixed"]
    assert empty["nonbootstrap_support"]["events"] == 0
    assert empty["associations"][0]["adjusted_rank_correlation"]["value"] is None


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "Duplicate"),
    (lambda rows: rows[0].update(k=50, new_step=50), "1–49"),
    (lambda rows: rows[0].update(k=4), "exact elapsed"),
    (lambda rows: rows[0].update(variance_total=float("nan")), "measurement"),
    (lambda rows: rows[0].update(revision_translation=-1), "measurement"),
    (lambda rows: rows[0].update(bootstrap=1), "boolean"),
    (lambda rows: rows[0].update(method="autohorizon_cap_5"), "method"),
])
def test_invalid_extracted_events_rejected(mutation, message):
    rows = events()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        validate_events(rows)


def test_same_seed_reproduces_cluster_intervals():
    first = analyze_statistics(events(), draws=100, seed=7)
    second = analyze_statistics(events(), draws=100, seed=7)
    assert first == second
