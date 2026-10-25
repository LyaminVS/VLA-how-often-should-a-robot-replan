"""Held-out prediction of later plan revision from information at the old call.

No inference or robot rollout is performed. The comparison asks whether FM
variance improves a specified prediction model on unseen physical starts. It
does not estimate the causal benefit of replanning. Cross-validation excludes
every cap, seed and transition of the held-out (suite, task, state). Bootstrap
intervals resample those held-out state clusters within task without refitting
the cross-validation models; they are conditional descriptive intervals.
"""

import math
from collections import Counter, defaultdict

from scripts.plan_revision_statistics import GROUP_NAMES, physical_key, validate_events

MODEL_NAMES = ("constant", "baseline", "baseline_plus_fm_total", "baseline_plus_fm_translation")
NUMERIC_FEATURES = (
    "log1p_k", "log1p_k_squared", "step_fraction", "old_translation_norm",
    "previous_revision", "previous_revision_missing", "old_command_gripper",
)
ALPHA = 1.0
VARIANCE_FLOOR = 1e-12


def _state_record(key):
    return {"suite": key[0], "task": key[1], "state": key[2]}


def _support(rows):
    return {
        "events": len(rows), "episodes": len({row["episode_id"] for row in rows}),
        "physical_settings": len({physical_key(row) for row in rows}),
        "tasks": len({(row["suite"], row["task"]) for row in rows}),
        "policy_seeds": sorted({row["policy_seed"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
    }


def _validate_predictors(grouped):
    for rows in grouped.values():
        identities = {}
        for row in rows:
            identity = (*physical_key(row), row["policy_seed"], row["method"])
            previous_identity = identities.setdefault(row["episode_id"], identity)
            if identity != previous_identity:
                raise ValueError("Episode identity changes across prediction events")
            for field in ("step_fraction", "old_translation_norm", "old_command_gripper"):
                if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]):
                    raise ValueError(f"Invalid prospective predictor: {field}")
            if not 0 <= row["step_fraction"] <= 1 or row["old_translation_norm"] < 0:
                raise ValueError("Invalid step fraction or old command norm")
            if "episode_max_steps" in row:
                maximum = row["episode_max_steps"]
                if (type(maximum) is not int or maximum < 1 or row["old_step"] >= maximum
                        or not math.isclose(row["step_fraction"], row["old_step"] / maximum,
                                            rel_tol=1e-12, abs_tol=1e-14)):
                    raise ValueError("Step fraction disagrees with old step and episode limit")
            missing = row.get("previous_revision_missing")
            value = row.get("previous_revision")
            if type(missing) is not bool:
                raise ValueError("Previous-revision missing indicator must be boolean")
            if missing:
                if value is not None and value != 0:
                    raise ValueError("A missing previous revision must be null or zero")
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Previous revision must be finite and nonnegative when present")


def episode_weights(rows, *, normalize_mean=False):
    """Each episode has total weight one before optional mean-one scaling."""
    import numpy as np

    counts = Counter(row["episode_id"] for row in rows)
    weights = np.asarray([1.0 / counts[row["episode_id"]] for row in rows], dtype=np.float64)
    if normalize_mean and len(weights):
        weights = weights / weights.mean()
    return weights


def _feature_matrix(rows, tasks, methods, *, extra=None):
    import numpy as np

    names = [f"task:{suite}/{task}" for suite, task in tasks]
    names += [f"method:{method}" for method in methods]
    names += list(NUMERIC_FEATURES)
    if extra is not None:
        names.append(f"log10_{extra}")
    matrix = []
    for row in rows:
        values = [float((row["suite"], row["task"]) == key) for key in tasks]
        values += [float(row["method"] == method) for method in methods]
        log_k = math.log1p(row["k"])
        values += [log_k, log_k * log_k, row["step_fraction"], row["old_translation_norm"],
                   0.0 if row["previous_revision_missing"] else row["previous_revision"],
                   float(row["previous_revision_missing"]), row["old_command_gripper"]]
        if extra is not None:
            values.append(math.log10(max(row[extra], VARIANCE_FLOOR)))
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64).reshape(len(rows), len(names)), names


def fit_ridge_predict(train_x, train_y, train_weights, test_x, *, alpha=ALPHA):
    """Weighted training-only standardization and an unpenalized intercept."""
    import numpy as np

    x, y, weights, test = (np.asarray(value, dtype=np.float64)
                           for value in (train_x, train_y, train_weights, test_x))
    if (x.ndim != 2 or test.ndim != 2 or test.shape[1] != x.shape[1]
            or y.shape != (len(x),) or weights.shape != y.shape or not len(x)
            or not all(np.isfinite(value).all() for value in (x, y, weights, test))
            or np.any(weights <= 0) or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Invalid weighted ridge inputs")
    means = np.average(x, axis=0, weights=weights)
    variance = np.average((x - means) ** 2, axis=0, weights=weights)
    scales = np.sqrt(variance)
    constant = scales <= 1e-12
    scales[constant] = 1.0
    centered_x = (x - means) / scales
    target_mean = float(np.average(y, weights=weights))
    gram = centered_x.T @ (weights[:, None] * centered_x)
    coefficients = np.linalg.solve(gram + alpha * np.eye(x.shape[1]),
                                   centered_x.T @ (weights * (y - target_mean)))
    predicted = target_mean + ((test - means) / scales) @ coefficients
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite held-out log predictions")
    return predicted, {
        "train_feature_mean": means.tolist(), "train_feature_scale": scales.tolist(),
        "constant_train_features": constant.tolist(), "coefficients": coefficients.tolist(),
        "intercept_after_centering": target_mean, "alpha": float(alpha),
        "training_weight_sum": float(weights.sum()), "training_weight_mean": float(weights.mean()),
    }


def weighted_metrics(target, prediction, weights):
    import numpy as np

    target, prediction, weights = (np.asarray(value, dtype=np.float64) for value in (target, prediction, weights))
    if not len(target) or weights.sum() <= 0:
        return {"mae": None, "rmse": None, "r2": None}
    residual = prediction - target
    mse = float(np.average(residual ** 2, weights=weights))
    variance = float(np.average((target - np.average(target, weights=weights)) ** 2, weights=weights))
    return {"mae": float(np.average(np.abs(residual), weights=weights)), "rmse": math.sqrt(mse),
            "r2": 1.0 - mse / variance if variance > 1e-28 else None}


def _interval(point, samples, draws, *, resampleable=True):
    import numpy as np

    values = [float(value) for value in samples if value is not None and math.isfinite(value)]
    enough = point is not None and resampleable and len(values) >= max(100, math.ceil(0.8 * draws))
    lower, upper = np.quantile(values, [0.025, 0.975]) if enough else (None, None)
    return {"value": point, "ci_low": float(lower) if lower is not None else None,
            "ci_high": float(upper) if upper is not None else None, "interval_available": enough,
            "valid_bootstrap_draws": len(values), "requested_bootstrap_draws": draws,
            "interval_note": "Conditional on fixed out-of-fold models; no model refitting"
            if enough else "Insufficient resampleable physical states or valid bootstrap draws"}


def _cluster_draws(rows, draws, rng):
    """State multiplicities within task; never sample individual transitions."""
    import numpy as np

    states = sorted({physical_key(row) for row in rows})
    position = {state: index for index, state in enumerate(states)}
    by_task = defaultdict(list)
    for state in states:
        by_task[state[:2]].append(position[state])
    counts = np.zeros((draws, len(states)), dtype=np.float64)
    for indices in by_task.values():
        indices = np.asarray(indices, dtype=int)
        selected = rng.choice(indices, size=(draws, len(indices)), replace=True)
        for index in indices:
            counts[:, index] = (selected == index).sum(axis=1)
    return states, counts, any(len(indices) > 1 for indices in by_task.values())


def _comparison(rows, target, baseline, added, *, name, draws, state_order, multiplicities, resampleable):
    import numpy as np

    weights = episode_weights(rows)
    baseline_metrics, added_metrics = (weighted_metrics(target, values, weights) for values in (baseline, added))
    difference = added_metrics["mae"] - baseline_metrics["mae"]
    improvement = -100.0 * difference / baseline_metrics["mae"] if baseline_metrics["mae"] > 0 else None
    states = {state: index for index, state in enumerate(state_order)}
    totals = np.zeros((len(state_order), 3), dtype=np.float64)
    baseline_error, added_error = np.abs(baseline - target), np.abs(added - target)
    for index, row in enumerate(rows):
        totals[states[physical_key(row)]] += weights[index] * np.array([baseline_error[index], added_error[index], 1.0])
    sampled = multiplicities @ totals
    valid = sampled[:, 2] > 0
    base_mae = sampled[valid, 0] / sampled[valid, 2]
    add_mae = sampled[valid, 1] / sampled[valid, 2]
    sampled_difference = add_mae - base_mae
    sampled_improvement = [float(-100 * delta / base) if base > 0 else None
                           for delta, base in zip(sampled_difference, base_mae, strict=True)]
    state_rows = []
    wins = losses = ties = 0
    for state, (base_sum, add_sum, mass) in zip(state_order, totals, strict=True):
        if mass <= 0:
            continue
        delta = float((add_sum - base_sum) / mass)
        win, loss = delta < -1e-12, delta > 1e-12
        wins += int(win)
        losses += int(loss)
        ties += int(not win and not loss)
        state_rows.append({**_state_record(state), "episodes": int(round(mass)),
                           "baseline_mae": float(base_sum / mass), "added_mae": float(add_sum / mass),
                           "mae_difference": delta, "outcome": "win" if win else "loss" if loss else "tie"})
    return {
        "model": name, "reference_model": "baseline", **_support(rows),
        "baseline_metrics": baseline_metrics, "added_metrics": added_metrics,
        "mae_difference": _interval(float(difference), sampled_difference, draws, resampleable=resampleable),
        "percent_mae_improvement": _interval(improvement, sampled_improvement, draws, resampleable=resampleable),
        "state_wins": wins, "state_losses": losses, "state_ties": ties, "by_state": state_rows,
    }


def _analyze_group(group, rows, *, draws, rng):
    import numpy as np

    included = sorted((row for row in rows if not row["bootstrap"]), key=lambda row: (
        row["suite"], row["task"], row["state"], row["episode_id"], row["old_step"], row["new_step"]))
    states = sorted({physical_key(row) for row in included})
    base_result = {
        "available_support": _support(rows), "nonbootstrap_support": _support(included),
        "excluded_bootstrap_events": len(rows) - len(included),
        "small_cluster_warning": len(states) < 10,
    }
    if len(states) < 2:
        return {**base_result, "available": False, "reason": "At least two physical settings are needed for held-out prediction",
                "folds": [], "models": {}, "comparisons": []}, []
    target = np.asarray([row["revision_translation"] for row in included], dtype=np.float64)
    target_log = np.log1p(target)
    predictions = {name: np.full(len(included), np.nan) for name in MODEL_NAMES}
    log_predictions = {name: np.full(len(included), np.nan) for name in MODEL_NAMES}
    fold_ids = [None] * len(included)
    folds = []
    for held_out in states:
        test_indices = np.asarray([i for i, row in enumerate(included) if physical_key(row) == held_out], dtype=int)
        train_indices = np.asarray([i for i, row in enumerate(included) if physical_key(row) != held_out], dtype=int)
        train, test = [included[i] for i in train_indices], [included[i] for i in test_indices]
        train_states = sorted({physical_key(row) for row in train})
        tasks = sorted({(row["suite"], row["task"]) for row in train})
        methods = sorted({row["method"] for row in train})
        weights = episode_weights(train, normalize_mean=True)
        fold_id = f"{group}|{held_out[0]}/{held_out[1]}/state{held_out[2]}"
        for index in test_indices:
            fold_ids[index] = fold_id
        training_log_mean = float(np.average(target_log[train_indices], weights=weights))
        model_audit = {"constant": {"training_log_target_mean": training_log_mean}}
        for name in MODEL_NAMES:
            if name == "constant":
                fitted_log = np.full(len(test), training_log_mean)
            else:
                extra = {"baseline": None, "baseline_plus_fm_total": "variance_total",
                         "baseline_plus_fm_translation": "variance_translation"}[name]
                train_x, feature_names = _feature_matrix(train, tasks, methods, extra=extra)
                test_x, _ = _feature_matrix(test, tasks, methods, extra=extra)
                fitted_log, audit = fit_ridge_predict(train_x, target_log[train_indices], weights, test_x)
                model_audit[name] = {"feature_names": feature_names, **audit}
            with np.errstate(over="raise", invalid="raise"):
                try:
                    fitted = np.maximum(0.0, np.expm1(fitted_log))
                except FloatingPointError as exc:
                    raise ValueError("Held-out prediction overflow; no silent clipping is allowed") from exc
            predictions[name][test_indices] = fitted
            log_predictions[name][test_indices] = fitted_log
        folds.append({
            "fold_id": fold_id, "test_state": _state_record(held_out),
            "train_states": [_state_record(key) for key in train_states],
            "train_episode_ids": sorted({row["episode_id"] for row in train}),
            "test_episode_ids": sorted({row["episode_id"] for row in test}),
            "train_support": _support(train), "test_support": _support(test),
            "task_categories_from_training": [list(key) for key in tasks],
            "method_categories_from_training": methods,
            "unseen_test_tasks": [list(key) for key in sorted({(row["suite"], row["task"]) for row in test} - set(tasks))],
            "unseen_test_methods": sorted({row["method"] for row in test} - set(methods)),
            "models": model_audit,
        })
    weights = episode_weights(included)
    if any(not np.isfinite(values).all() for values in predictions.values()) or any(value is None for value in fold_ids):
        raise ValueError("Cross-validation did not predict every included event exactly once")
    models = {name: weighted_metrics(target, values, weights) for name, values in predictions.items()}
    state_order, multiplicities, resampleable = _cluster_draws(included, draws, rng)
    comparisons = []
    for name in ("baseline_plus_fm_total", "baseline_plus_fm_translation"):
        comparison = _comparison(included, target, predictions["baseline"], predictions[name], name=name,
                                 draws=draws, state_order=state_order, multiplicities=multiplicities,
                                 resampleable=resampleable)
        by_method, by_task = [], []
        for kind in ("method", "task"):
            keys = sorted({row["method"] if kind == "method" else (row["suite"], row["task"]) for row in included})
            for key in keys:
                indices = np.asarray([i for i, row in enumerate(included)
                                      if (row["method"] if kind == "method" else (row["suite"], row["task"])) == key])
                selected = [included[i] for i in indices]
                subgroup = _comparison(selected, target[indices], predictions["baseline"][indices],
                                       predictions[name][indices], name=name, draws=draws,
                                       state_order=state_order, multiplicities=multiplicities,
                                       resampleable=len({physical_key(row) for row in selected}) > 1)
                if kind == "method":
                    subgroup["method"] = key
                    by_method.append(subgroup)
                else:
                    subgroup.update(suite=key[0], task=key[1])
                    by_task.append(subgroup)
        comparison.update(by_method=by_method, by_task=by_task)
        comparisons.append(comparison)
    prediction_rows = []
    for index, row in enumerate(included):
        for name in MODEL_NAMES:
            error = float(predictions[name][index] - target[index])
            prediction_rows.append({
                "group": group, "model": name, "fold_id": fold_ids[index],
                **{key: row[key] for key in ("cohort", "episode_id", "suite", "task", "state", "policy_seed",
                                            "method", "old_step", "new_step", "k")},
                "target": float(target[index]), "prediction": float(predictions[name][index]),
                "prediction_log": float(log_predictions[name][index]), "error": error,
                "absolute_error": abs(error), "squared_error": error * error,
                "episode_weight": float(weights[index]),
            })
    return {**base_result, "available": True, "folds": folds, "models": models,
            "comparisons": comparisons}, prediction_rows


def compare_predictions(events, draws=2000, seed=20261024):
    """Return grouped OOF prediction comparisons and long-form prediction rows."""
    import numpy as np

    if type(draws) is not int or draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    grouped = validate_events(events)
    _validate_predictors(grouped)
    groups, prediction_rows = {}, []
    for index, name in enumerate(GROUP_NAMES):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        groups[name], rows = _analyze_group(name, grouped[name], draws=draws, rng=rng)
        prediction_rows.extend(rows)
    return {
        "analysis_kind": "prospective_held_out_plan_revision_prediction",
        "groups": groups,
        "design": {
            "cross_validation": "Leave one physical (suite, task, state) out; all caps, policy seeds and transitions move together",
            "cohorts": "Fit and evaluate capped, supplemental_fixed, supplemental_dvac separately",
            "training": "Weighted Ridge, alpha=1; weighted training-only feature mean/std; unpenalized intercept; no hyperparameter selection",
            "training_weighting": "Each training episode has total weight 1, then weights are normalized to mean 1",
            "evaluation_weighting": "Each included episode has total weight 1; its transitions share that weight",
            "target": "log1p(revision_translation); predictions transformed by max(0, expm1(predicted_log))",
            "constant": "Training episode-weighted mean log target, then inverse transformed",
            "baseline_features": ["task one-hot", "method/cap one-hot", *NUMERIC_FEATURES],
            "fm_total_feature": "log10(max(variance_total, 1e-12))",
            "fm_translation_sensitivity": "log10(max(variance_translation, 1e-12))",
            "categorical_encoding": "Vocabulary fitted on training rows only; unseen test categories use all-zero indicators and are audited",
            "bootstrap_exclusion": "Exclude transitions whose old call used forced DVAC bootstrap; fixed first calls remain eligible",
            "mae_difference_sign": "Added-model MAE minus baseline MAE; negative means improvement",
            "percent_improvement": "100 * (baseline MAE - added MAE) / baseline MAE",
        },
        "bootstrap": {
            "draws": draws, "seed": seed, "confidence_level": 0.95,
            "unit": "Held-out physical state within task, preserving all caps, seeds and transitions",
            "conditioning": "Fixed out-of-fold predictions; models are not refitted in bootstrap draws",
            "interpretation": "Conditional descriptive paired percentile intervals, not unconditional model-training uncertainty",
        },
        "limitations": [
            "Prediction of a later command change does not establish action error or a causal benefit of replanning.",
            "Both the observation and sampling noise change at the next call.",
            "Only boundaries with a next call and an overlapping old action (K<50) have a target; terminal and exhausted plans are absent.",
            "DVAC uses variance to choose K, so its observed boundaries are selected by the same signal being tested.",
            "Task/method indicators, log K and squared log K control selected measured scheduling differences, not every confounder.",
            "The supplemental cohorts contain only six physical settings; their intervals are especially unstable.",
            "This prespecified model comparison is exploratory on previously collected data, not a tuned scheduler or a causal experiment.",
        ],
    }, prediction_rows
