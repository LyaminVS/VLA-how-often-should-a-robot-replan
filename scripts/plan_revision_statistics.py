"""Cluster-aware descriptive association of FM variance with later plan revision.

No rollout or model inference is performed here. Events are complete, verified
adjacent-plan pairs produced by the extraction script. Association concerns
command revision, not an action-error label or the benefit of replanning.
"""

import math
from collections import Counter, defaultdict

ASSOCIATIONS = (
    ("total_variance", "variance_total", "revision_translation"),
    ("translation_variance", "variance_translation", "revision_translation"),
    ("clipped_revision", "variance_total", "revision_translation_clipped"),
)
GROUP_NAMES = ("capped", "supplemental_fixed", "supplemental_dvac")
STATISTICS = ("adjusted_rank_correlation", "marginal_spearman", "episode_weighted_marginal_spearman")


def physical_key(row):
    return row["suite"], row["task"], row["state"]


def stratum_key(row):
    return row["method"], row["suite"], row["task"], row["k"]


def _group(row):
    if row["cohort"] == "capped":
        if row["method"] not in {f"dvac_cap_{cap}" for cap in (5, 10, 25, 50)}:
            raise ValueError("Unexpected method in capped cohort")
        return "capped"
    if row["cohort"] == "supplemental":
        if row["method"] in ("fixed_5", "fixed_10", "fixed_25"):
            return "supplemental_fixed"
        if row["method"] == "dvac":
            return "supplemental_dvac"
    raise ValueError("Unexpected cohort or method")


def validate_events(events):
    """Require real overlap targets and reject duplicate transitions or NaNs."""
    grouped = {name: [] for name in GROUP_NAMES}
    seen = set()
    for row in events:
        group = _group(row)
        for field in ("episode_id", "suite", "task", "method"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Missing event identity field: {field}")
        for field in ("state", "policy_seed", "old_step", "new_step", "k"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"Invalid integer event field: {field}")
        if not 1 <= row["k"] < 50 or row["new_step"] - row["old_step"] != row["k"]:
            raise ValueError("Plan overlap requires exact elapsed K in 1–49")
        if type(row.get("bootstrap")) is not bool or type(row.get("gripper_flip")) is not bool:
            raise ValueError("Bootstrap and gripper-flip flags must be boolean")
        if row["method"].startswith("fixed_") and row["bootstrap"]:
            raise ValueError("Fixed scheduling does not use DVAC bootstrap")
        for field in ("variance_total", "variance_translation", "revision_translation", "revision_translation_clipped"):
            if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid nonnegative event measurement: {field}")
        key = row["cohort"], row["episode_id"], row["old_step"], row["new_step"]
        if key in seen:
            raise ValueError("Duplicate adjacent-plan event")
        seen.add(key)
        grouped[group].append(row)
    return grouped


def weighted_midranks(values, weights):
    """Weighted empirical-CDF midranks in [0,1], retaining exact tie groups."""
    import numpy as np

    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ValueError("Ranks require equally sized nonempty vectors")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Ranks require finite values and positive finite weights")
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_values[1:] != sorted_values[:-1])]
    totals = np.add.reduceat(sorted_weights, starts)
    midpoint = (np.cumsum(totals) - .5 * totals) / weights.sum()
    ranked = np.repeat(midpoint, np.diff(np.r_[starts, len(values)]))
    output = np.empty_like(values)
    output[order] = ranked
    return output


def weighted_correlation(x, y, weights):
    import numpy as np

    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
    if len(x) < 2 or weights.sum() <= 0:
        return None
    xc = x - np.average(x, weights=weights)
    yc = y - np.average(y, weights=weights)
    xx, yy = np.sum(weights * xc * xc), np.sum(weights * yc * yc)
    if xx <= 1e-28 or yy <= 1e-28:
        return None
    return float(np.clip(np.sum(weights * xc * yc) / math.sqrt(xx * yy), -1, 1))


def _support(rows):
    return {
        "events": len(rows), "episodes": len({r["episode_id"] for r in rows}),
        "physical_settings": len({physical_key(r) for r in rows}),
        "tasks": len({(r["suite"], r["task"]) for r in rows}),
        "policy_seeds": sorted({r["policy_seed"] for r in rows}),
    }


def eligible_strata(rows, xfield, yfield):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[stratum_key(row)].append(index)
    keys, table = set(), []
    for key, indices in sorted(groups.items()):
        group = [rows[index] for index in indices]
        constant_x = len({row[xfield] for row in group}) < 2
        constant_y = len({row[yfield] for row in group}) < 2
        count = len({physical_key(row) for row in group})
        reasons = []
        if len(group) < 3:
            reasons.append("fewer than three events")
        if count < 2:
            reasons.append("fewer than two physical settings")
        if constant_x:
            reasons.append("constant predictor")
        if constant_y:
            reasons.append("constant response")
        include = not reasons
        if include:
            keys.add(key)
        table.append({"method": key[0], "suite": key[1], "task": key[2], "k": key[3],
                      **_support(group), "eligible": include, "exclusion_reason": "; ".join(reasons)})
    return keys, table


def _episode_weights(rows, multiplicities):
    import numpy as np

    episode_sizes = Counter(row["episode_id"] for row in rows)
    return np.array([multiplicities[physical_key(row)] / episode_sizes[row["episode_id"]] for row in rows], dtype=float)


def _adjusted_arrays(rows, xfield, yfield, eligible, multiplicities):
    """Recompute weighted ranks after resampling, preserving fixed point support."""
    import numpy as np

    active = [row for row in rows if stratum_key(row) in eligible and multiplicities.get(physical_key(row), 0) > 0]
    groups = defaultdict(list)
    for row in active:
        groups[stratum_key(row)].append(row)
    # A bootstrap draw can retain only one original state in a stratum. This is
    # a cluster resample, not new independent states; only constant strata lack
    # a rank covariance and are dropped. Eligibility keys stay fixed at point.
    usable = {key for key, group in groups.items()
              if len({row[xfield] for row in group}) > 1 and len({row[yfield] for row in group}) > 1}
    active = [row for row in active if stratum_key(row) in usable]
    if not active:
        return active, np.array([]), np.array([]), np.array([])
    weights = _episode_weights(active, multiplicities)
    rx, ry = np.zeros(len(active)), np.zeros(len(active))
    indices = defaultdict(list)
    for index, row in enumerate(active):
        indices[stratum_key(row)].append(index)
    for selected in indices.values():
        selected = np.array(selected)
        w = weights[selected]
        x = weighted_midranks([active[index][xfield] for index in selected], w)
        y = weighted_midranks([active[index][yfield] for index in selected], w)
        rx[selected] = x - np.average(x, weights=w)
        ry[selected] = y - np.average(y, weights=w)
    return active, rx, ry, weights


def association_estimates(rows, xfield, yfield, eligible, multiplicities=None):
    import numpy as np

    multiplicities = multiplicities or {physical_key(row): 1 for row in rows}
    active = [row for row in rows if multiplicities.get(physical_key(row), 0) > 0]
    result = {name: None for name in STATISTICS}
    if not active:
        return result
    x, y = [row[xfield] for row in active], [row[yfield] for row in active]
    call_weights = np.array([multiplicities[physical_key(row)] for row in active], dtype=float)
    episode_weights = _episode_weights(active, multiplicities)
    result["marginal_spearman"] = weighted_correlation(
        weighted_midranks(x, call_weights), weighted_midranks(y, call_weights), call_weights)
    result["episode_weighted_marginal_spearman"] = weighted_correlation(
        weighted_midranks(x, episode_weights), weighted_midranks(y, episode_weights), episode_weights)
    _, rx, ry, weights = _adjusted_arrays(rows, xfield, yfield, eligible, multiplicities)
    result["adjusted_rank_correlation"] = weighted_correlation(rx, ry, weights)
    return result


def binned_summary(rows, xfield, yfield, *, bins=5):
    """Marginal, episode-weighted bins for plotting; no independent-call errors."""
    import numpy as np

    if not rows:
        return []
    weights = _episode_weights(rows, {physical_key(row): 1 for row in rows})
    ranks = weighted_midranks([row[xfield] for row in rows], weights)
    assignments = np.minimum(bins - 1, (ranks * bins).astype(int))
    result = []
    for index in range(bins):
        positions = np.flatnonzero(assignments == index)
        if not len(positions):
            continue
        group, w = [rows[i] for i in positions], weights[positions]
        result.append({
            "bin": index + 1, "weight": float(w.sum()), **_support(group),
            "predictor_min": min(row[xfield] for row in group),
            "predictor_max": max(row[xfield] for row in group),
            "predictor_mean": float(np.average([row[xfield] for row in group], weights=w)),
            "revision_mean": float(np.average([row[yfield] for row in group], weights=w)),
            "gripper_flip_fraction": float(np.average([row["gripper_flip"] for row in group], weights=w)),
        })
    return result


def _draw_multiplicities(group, draws, rng):
    tasks = defaultdict(set)
    for row in group:
        tasks[(row["suite"], row["task"])].add(physical_key(row))
    by_task = [sorted(values) for _, values in sorted(tasks.items())]
    result = []
    for _ in range(draws):
        counts = Counter()
        for keys in by_task:
            for index in rng.integers(0, len(keys), len(keys)):
                counts[keys[index]] += 1
        result.append(counts)
    return result


def _interval(value, samples, draws):
    import numpy as np

    valid = [sample for sample in samples if sample is not None and math.isfinite(sample)]
    enough = value is not None and len(valid) >= max(100, math.ceil(.8 * draws))
    low, high = np.quantile(valid, [.025, .975]) if enough else (None, None)
    return {"value": value, "ci_low": float(low) if low is not None else None,
            "ci_high": float(high) if high is not None else None,
            "valid_bootstrap_draws": len(valid), "requested_bootstrap_draws": draws,
            "interval_available": enough,
            "interval_note": "Descriptive clustered percentile interval" if enough else "Insufficient nondegenerate bootstrap draws"}


def group_statistics(group, *, draws, rng):
    # A fixed method's first call has bootstrap=False and remains in the data.
    rows = [row for row in group if not row["bootstrap"]]
    samples = _draw_multiplicities(group, draws, rng)
    associations = []
    for name, xfield, yfield in ASSOCIATIONS:
        eligible, table = eligible_strata(rows, xfield, yfield)
        point = association_estimates(rows, xfield, yfield, eligible)
        estimates = {statistic: [] for statistic in STATISTICS}
        if rows and any(value is not None for value in point.values()):
            for multiplicities in samples:
                estimate = association_estimates(rows, xfield, yfield, eligible, multiplicities)
                for statistic in STATISTICS:
                    estimates[statistic].append(estimate[statistic])
        adjusted_rows = [row for row in rows if stratum_key(row) in eligible]
        associations.append({
            "name": name, "predictor": xfield, "response": yfield,
            "role": "primary descriptive association" if name == "total_variance" else "sensitivity analysis",
            **{statistic: _interval(point[statistic], estimates[statistic], draws) for statistic in STATISTICS},
            "adjusted_support": {**_support(adjusted_rows), "eligible_strata": len(eligible),
                                 "candidate_strata": len(table), "excluded_events": len(rows) - len(adjusted_rows)},
            "strata": table, "marginal_bins": binned_summary(rows, xfield, yfield),
        })
    by_method = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        by_method.append({"method": method, **_support(selected),
                          "k_min": min(row["k"] for row in selected), "k_max": max(row["k"] for row in selected),
                          "exact_k_values": sorted({row["k"] for row in selected})})
    return {
        "available_support": _support(group), "nonbootstrap_support": _support(rows),
        "excluded_bootstrap_events": len(group) - len(rows), "by_method": by_method,
        "associations": associations,
        "small_cluster_warning": len({physical_key(row) for row in group}) < 10,
        "interpretation": ("Few physical settings: cluster-bootstrap intervals are unstable descriptive summaries."
                           if len({physical_key(row) for row in group}) < 10 else
                           "Descriptive associations on previously examined trajectories; no causal or calibrated-confidence claim."),
    }


def analyze_statistics(events, draws=2000, seed=20261024):
    """Return separate cohort associations and state-cluster bootstrap intervals.

    Bootstrap multiplicities represent duplicated sampled states without
    inventing additional independent state identities. Point-eligible strata
    are held fixed; ranks, nonconstant support, and episode weights are recomputed
    in every draw. All methods, seeds and calls belonging to a state move together.
    """
    import numpy as np

    if type(draws) is not int or draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    grouped = validate_events(events)
    rng = np.random.default_rng(seed)
    return {
        "groups": {name: group_statistics(grouped[name], draws=draws, rng=rng) for name in GROUP_NAMES},
        "bootstrap": {
            "draws": draws, "seed": seed, "confidence_level": .95,
            "unit": "physical (suite, task, initial-state) setting; all methods, seeds and calls stay together",
            "strata": "task; resample its observed number of physical settings with replacement",
            "support_rule": "Point-eligible (method, task, exact K) strata: at least 3 events, at least 2 original physical settings, nonconstant predictor and response",
            "recomputation": "Hold point-eligible stratum keys fixed; recompute weighted ranks and episode weights in each draw; omit strata constant in that draw",
            "weighting": "Each retained episode has equal total weight; calls within an episode share that weight",
            "marginal_weighting": "marginal_spearman weights transitions equally and is descriptive only; episode_weighted_marginal_spearman gives equal episode weight",
            "interval_availability": "At least 100 and at least 80% of requested draws must be nondegenerate",
        },
        "limitations": [
            "D measures a change in commanded translation, not a measured action error or a causal benefit of replanning.",
            "Changing observations and the policy sampling seed can both cause revision.",
            "DVAC chooses K using variance, so selected replanning boundaries have selection and range-restriction bias even after exact-K adjustment.",
            "Only transitions with another model call and K<50 have an overlap target; terminal calls and K=50 are absent rather than zero.",
            "Total FM variance uses normalized translation, rotation and gripper coordinates; translation-only variance is a scale-alignment sensitivity, not physical position-error variance.",
            "The clipped-revision sensitivity compares clipped command vectors; clipping does not turn D into displacement or task-error ground truth.",
            "Supplemental fixed scheduling chooses K independently of variance but has only six physical task/state settings and is analyzed separately.",
            "These are exploratory descriptive intervals without multiple-comparison correction; no calibrated confidence, equivalence or causality claim is made.",
        ],
    }
