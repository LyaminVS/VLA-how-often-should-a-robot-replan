"""Paired analysis of retained fixed controls and capped DVAC on the same starts."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from vla_replan.storage import atomic_json

METHODS = ("fixed_1", "fixed_5", "fixed_10", "fixed_25", "fixed_50",
           "dvac_cap_5", "dvac_cap_10", "dvac_cap_25", "dvac_cap_50")
CAP_METHODS = METHODS[5:]
DESIGNATED = (("dvac_cap_10", "fixed_10"), ("dvac_cap_10", "dvac_cap_50"))
PLOT_CONTRASTS = tuple((f"dvac_cap_{cap}", f"fixed_{cap}") for cap in (5, 10, 25, 50)) + (DESIGNATED[1],)
METRICS = ("success", "model_calls", "steps", "eef_path_m", "gripper_close_commands")
TASK_NAMES = {
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": "Soup into basket",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": "Bowl onto plate",
    "open_the_middle_drawer_of_the_cabinet": "Open middle drawer",
}


def label(method):
    return f"DVAC cap {method.rsplit('_', 1)[1]}" if method.startswith("dvac_cap_") else f"Fixed K={method.rsplit('_', 1)[1]}"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def paired_layout(records, methods=METHODS):
    """Require complete method pairs and the same seed set in every physical setting."""
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Methods must be a nonempty unique sequence")
    cells, task_states, seed_sets = {}, defaultdict(set), defaultdict(set)
    for record in records:
        row = record["summary"]
        key = (row["suite"], row["task"], row["state"], row["policy_seed"], row["method"])
        if key in cells or row["method"] not in methods:
            raise ValueError("Duplicate or unexpected method episode")
        if type(row.get("success")) is not bool:
            raise ValueError("Outcomes must be boolean")
        for metric in METRICS[1:]:
            value = row.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid metric: {metric}")
        if row["steps"] <= 0 or row["model_calls"] <= 0:
            raise ValueError("Each episode must contain actions and model calls")
        cells[key] = row
        task_states[(row["suite"], row["task"])].add(row["state"])
        seed_sets[(row["suite"], row["task"], row["state"])].add(row["policy_seed"])
    if not cells:
        raise ValueError("No verified episodes")
    seeds = sorted(next(iter(seed_sets.values())))
    if any(sorted(values) != seeds for values in seed_sets.values()):
        raise ValueError("Physical settings have different policy seed repetitions")
    for (suite, task), states in task_states.items():
        for state in states:
            for seed in seeds:
                for method in methods:
                    if (suite, task, state, seed, method) not in cells:
                        raise ValueError("Incomplete paired method block")
    return cells, {task: sorted(states) for task, states in sorted(task_states.items())}, seeds


def cluster_statistics(records, *, methods=METHODS, draws=10000, seed=20261024):
    """Task-stratified paired state bootstrap, retaining all seed repetitions together."""
    import numpy as np

    if type(draws) is not int or draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    methods = tuple(methods)
    cells, task_states, seeds = paired_layout(records, methods)
    if any(len(states) < 2 for states in task_states.values()):
        raise ValueError("Bootstrap intervals require at least two states per task")
    rng = np.random.default_rng(seed)
    point_by_task, bootstrap_by_task, task_estimates = [], [], []
    metric_names = (*METRICS, "calls_per_100_actions")

    def extend_ratios(matrix):
        ratio = 100 * matrix[..., METRICS.index("model_calls")] / matrix[..., METRICS.index("steps")]
        return np.concatenate([matrix, ratio[..., None]], axis=-1)

    def estimate_rows(point, boot, summaries, *, task="all", suite="all"):
        output = []
        for index, method in enumerate(methods):
            group = [row for row in summaries if row["method"] == method]
            result = {"method": method, "suite": suite, "task": task,
                      "episodes": len(group), "successes": sum(row["success"] for row in group),
                      "physical_settings": len({(row["suite"], row["task"], row["state"]) for row in group}),
                      "policy_seed_repetitions": len(seeds)}
            for metric_index, metric in enumerate(metric_names):
                low, high = np.quantile(boot[:, index, metric_index], [.025, .975])
                result[metric + "_mean"] = float(point[index, metric_index])
                result[metric + "_ci_low"] = float(low)
                result[metric + "_ci_high"] = float(high)
            output.append(result)
        return output

    for (suite, task), states in task_states.items():
        matrix = np.array([[
            [mean(float(cells[(suite, task, state, policy_seed, method)][metric]) for policy_seed in seeds)
             for metric in METRICS]
            for method in methods] for state in states], dtype=float)
        selected = rng.integers(0, len(states), size=(draws, len(states)))
        point, boot = matrix.mean(axis=0), matrix[selected].mean(axis=1)
        point_by_task.append(point)
        bootstrap_by_task.append(boot)
        summaries = [row for key, row in cells.items() if key[:2] == (suite, task)]
        task_estimates += estimate_rows(extend_ratios(point), extend_ratios(boot), summaries, task=task, suite=suite)
    point = extend_ratios(np.mean(point_by_task, axis=0))
    boot = extend_ratios(np.mean(bootstrap_by_task, axis=0))
    estimates = estimate_rows(point, boot, list(cells.values()))
    contrasts = []
    for first, second in combinations(methods, 2):
        method, reference = second, first
        if (first, second) in DESIGNATED:
            method, reference = first, second
        left, right = methods.index(method), methods.index(reference)
        comparison = {"method": method, "reference": reference, "designated_contrast": (method, reference) in DESIGNATED,
                      "paired_seed_runs": len(cells) // len(methods), "physical_settings": sum(map(len, task_states.values()))}
        discordance = Counter()
        for (suite, task), states in task_states.items():
            for state in states:
                for policy_seed in seeds:
                    a = cells[(suite, task, state, policy_seed, method)]["success"]
                    b = cells[(suite, task, state, policy_seed, reference)]["success"]
                    discordance[(a, b)] += 1
        comparison.update(method_only_success=discordance[(True, False)], reference_only_success=discordance[(False, True)],
                          both_success=discordance[(True, True)], both_failure=discordance[(False, False)])
        for index, metric in enumerate(metric_names):
            low, high = np.quantile(boot[:, left, index] - boot[:, right, index], [.025, .975])
            comparison[metric + "_difference"] = float(point[left, index] - point[right, index])
            comparison[metric + "_ci_low"] = float(low)
            comparison[metric + "_ci_high"] = float(high)
        contrasts.append(comparison)
    outcomes = []
    for (suite, task), states in task_states.items():
        for state in states:
            for policy_seed in seeds:
                row = {"suite": suite, "task": task, "state": state, "policy_seed": policy_seed}
                for method in methods:
                    source = cells[(suite, task, state, policy_seed, method)]
                    row[method + "_success"] = source["success"]
                    row[method + "_calls"] = source["model_calls"]
                outcomes.append(row)
    return {"methods": estimates, "by_task": task_estimates, "contrasts": contrasts, "outcome_matrix": outcomes,
            "bootstrap": {"draws": draws, "seed": seed, "confidence_level": .95,
                          "unit": "physical task/initial-state setting; all methods and seeds kept together",
                          "strata": "tasks; fixed equal task weights",
                          "physical_settings": sum(map(len, task_states.values())),
                          "states_per_task": {suite + "/" + task: len(states) for (suite, task), states in task_states.items()},
                          "policy_seeds": seeds,
                          "interpretation": "Descriptive paired bootstrap intervals on previously examined states; no confirmatory significance claims."}}


def cap_behavior(records):
    result, histogram, decisions, timing = [], Counter(), [], []
    for method in CAP_METHODS:
        group = [record for record in records if record["summary"]["method"] == method]
        if not group:
            raise ValueError(f"Missing capped method: {method}")
        chunks = [chunk for record in group for chunk in record["chunks"]]
        selected = [chunk["decision"]["chosen_k"] for chunk in chunks]
        raw = [chunk["decision"]["raw_chosen_k"] for chunk in chunks]
        executed = [chunk["executed_steps"] for chunk in chunks]
        nonbootstrap = [chunk for chunk in chunks if not chunk["decision"]["bootstrap"]]
        cap = int(method.rsplit("_", 1)[1])
        result.append({
            "method": method, "execution_cap": cap, "episodes": len(group), "model_calls": len(chunks),
            "raw_chosen_k_mean_per_call": mean(raw), "chosen_k_mean_per_call": mean(selected),
            "executed_prefix_mean_per_call": mean(executed), "chosen_k_min": min(selected), "chosen_k_max": max(selected),
            "bootstrap_calls": len(chunks) - len(nonbootstrap), "nonbootstrap_calls": len(nonbootstrap),
            "cap_applied_calls": sum(chunk["decision"]["cap_applied"] for chunk in chunks),
            "cap_applied_fraction": mean(chunk["decision"]["cap_applied"] for chunk in chunks),
            "nonbootstrap_k1_fraction": mean(chunk["decision"]["chosen_k"] == 1 for chunk in nonbootstrap),
            "nonbootstrap_at_cap_fraction": mean(chunk["decision"]["chosen_k"] == cap for chunk in nonbootstrap),
            "nonbootstrap_no_crossing_calls": sum(chunk["decision"]["first_crossing"] is None for chunk in nonbootstrap),
            "nonbootstrap_no_crossing_fraction": mean(chunk["decision"]["first_crossing"] is None for chunk in nonbootstrap),
        })
        for record in group:
            summary = record["summary"]
            for chunk in record["chunks"]:
                decision = chunk["decision"]
                histogram[(method, decision["bootstrap"], decision["raw_chosen_k"], decision["chosen_k"])] += 1
                decisions.append({"episode_id": summary["episode_id"], "method": method, "step": chunk["step"],
                                  **decision, "executed_steps": chunk["executed_steps"]})
        row = {"method": method, "episodes": len(group), "timing_population": "instrumented capped methods only"}
        for field in ("inference_s", "selector_s", "simulation_s", "compute_s", "episode_wall_s"):
            row[field + "_total"] = math.fsum(record["summary"][field] for record in group)
            row[field + "_mean_per_episode"] = mean(record["summary"][field] for record in group)
        row["inference_s_per_call"] = row["inference_s_total"] / len(chunks)
        row["selector_s_per_call"] = row["selector_s_total"] / len(chunks)
        timing.append(row)
    return result, [
        {"method": method, "bootstrap": bootstrap, "raw_chosen_k": raw, "chosen_k": chosen, "calls": count}
        for (method, bootstrap, raw, chosen), count in sorted(histogram.items())], decisions, timing


def frontier(estimates):
    """Observed Pareto set only; no inference of a true dominating method."""
    result = []
    for row in estimates:
        dominating = [other["method"] for other in estimates
                      if other["success_mean"] >= row["success_mean"]
                      and other["model_calls_mean"] <= row["model_calls_mean"]
                      and (other["success_mean"] > row["success_mean"] or other["model_calls_mean"] < row["model_calls_mean"])]
        result.append({"method": row["method"], "observed_nondominated": not dominating,
                       "dominated_by_observed_means": ",".join(dominating)})
    return result


def make_plots(output, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {method: color for method, color in zip(METHODS,
              ("#3d536f", "#557ba4", "#66a2bc", "#80b9b0", "#a3cdb0", "#c8793c", "#b84b55", "#925d96", "#645289"), strict=True)}
    rows = results["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for index, row in enumerate(rows):
        method, y = row["method"], row["success_mean"]
        yerror = [[max(0, y - row["success_ci_low"])], [max(0, row["success_ci_high"] - y)]]
        axes[0].bar(index, y, color=colors[method], width=.7)
        axes[0].errorbar(index, y, yerr=yerror, color="#222222", capsize=3, fmt="none")
        axes[0].text(index, max(y, row["success_ci_high"]) + .035, f"{row['successes']}/{row['episodes']}", ha="center", fontsize=8)
        x = row["model_calls_mean"]
        xerror = [[max(0, x - row["model_calls_ci_low"])], [max(0, row["model_calls_ci_high"] - x)]]
        axes[1].errorbar(x, y, xerr=xerror, yerr=yerror, fmt="o" if method.startswith("fixed_") else "D",
                         markersize=7, capsize=2, color=colors[method], label=label(method), alpha=.9)
    axes[0].set(xticks=range(len(rows)), xticklabels=[label(row["method"]) for row in rows],
                ylabel="Completion fraction", ylim=(-.02, 1.17), title="Completion on matched starts")
    axes[0].tick_params(axis="x", labelrotation=55, labelsize=9)
    axes[1].set(xlabel="Mean model calls per episode", ylabel="Completion fraction", ylim=(-.02, 1.08),
                title="Completion / model-call trade-off")
    axes[1].legend(fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("Retained fixed controls and capped DVAC; descriptive paired state-bootstrap 95% intervals")
    fig.savefig(output / "success_calls_frontier.png", dpi=180)
    plt.close(fig)

    tasks = list(dict.fromkeys(row["task"] for row in results["by_task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(15, 4.8), squeeze=False, layout="constrained")
    for index, task in enumerate(tasks):
        group = [row for row in results["by_task"] if row["task"] == task]
        axis = axes[0, index]
        axis.bar(range(len(group)), [row["success_mean"] for row in group], color=[colors[row["method"]] for row in group])
        for position, row in enumerate(group):
            axis.text(position, row["success_mean"] + .03, f"{row['successes']}/{row['episodes']}", ha="center", fontsize=8)
        axis.set(xticks=range(len(group)), xticklabels=[label(row["method"]) for row in group],
                 ylim=(0, 1.15), ylabel="Completion fraction", title=TASK_NAMES.get(task, task))
        axis.tick_params(axis="x", labelrotation=65, labelsize=8)
    fig.savefig(output / "task_success.png", dpi=180)
    plt.close(fig)

    behavior = results["cap_behavior"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(len(behavior))
    for shift, metric, title, color in ((-.25, "raw_chosen_k_mean_per_call", "Before cap", "#b9c5d2"),
                                       (0, "chosen_k_mean_per_call", "Selected after cap", "#a85e74"),
                                       (.25, "executed_prefix_mean_per_call", "Actually executed", "#547f8a")):
        axes[0].bar(positions + shift, [row[metric] for row in behavior], width=.25, label=title, color=color)
    axes[0].set(xticks=positions, xticklabels=[label(row["method"]) for row in behavior],
                ylabel="Mean actions per policy call", title="Selected versus executed prefix")
    axes[0].legend(fontsize=8)
    for shift, metric, title, color in ((-.2, "nonbootstrap_at_cap_fraction", "Chosen K equals cap", "#a85e74"),
                                       (.2, "nonbootstrap_k1_fraction", "Chosen K=1", "#547f8a")):
        axes[1].bar(positions + shift, [row[metric] for row in behavior], width=.4, label=title, color=color)
    axes[1].set(xticks=positions, xticklabels=[label(row["method"]) for row in behavior], ylabel="Fraction of calls",
                ylim=(0, 1.1), title="Choices after mandatory first-call K=1")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "cap_horizon_behavior.png", dpi=180)
    plt.close(fig)

    lookup = {(row["method"], row["reference"]): row for row in results["contrasts"]}
    comparisons = [lookup[key] for key in PLOT_CONTRASTS]
    fig, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for index, row in enumerate(comparisons):
        value = 100 * row["success_difference"]
        lower, upper = 100 * row["success_ci_low"], 100 * row["success_ci_high"]
        axis.errorbar(value, index, xerr=[[max(0, value - lower)], [max(0, upper - value)]],
                      fmt="o", color="#aa546e" if row["designated_contrast"] else "#66869b", capsize=4)
    axis.axvline(0, color="#999999", linestyle="--", linewidth=1)
    axis.set(yticks=range(len(comparisons)), yticklabels=[f"{label(row['method'])} minus {label(row['reference'])}" for row in comparisons],
             xlabel="Completion difference (percentage points)", title="Paired differences; descriptive 95% intervals")
    axis.invert_yaxis()
    fig.savefig(output / "paired_success_differences.png", dpi=180)
    plt.close(fig)

    matrix_rows = results["outcome_matrix"]
    matrix = np.array([[int(row[method + "_success"]) for method in METHODS] for row in matrix_rows])
    fig, axis = plt.subplots(figsize=(12, max(5, .29 * len(matrix_rows) + 2)), layout="constrained")
    axis.imshow(matrix, aspect="auto", cmap=ListedColormap(["#e7bbc0", "#69b59c"]), vmin=0, vmax=1)
    for row_index, row in enumerate(matrix_rows):
        for column, method in enumerate(METHODS):
            axis.text(column, row_index, str(row[method + "_calls"]), ha="center", va="center", fontsize=8)
    axis.set(xticks=range(len(METHODS)), xticklabels=[label(method) for method in METHODS],
             yticks=range(len(matrix_rows)), yticklabels=[f"{TASK_NAMES.get(row['task'], row['task'])} | s{row['state']} r{row['policy_seed']}" for row in matrix_rows],
             title="Every matched outcome: green = success, pink = failure; numbers = model calls")
    axis.tick_params(axis="x", labelrotation=45)
    axis.tick_params(axis="y", labelsize=8)
    fig.savefig(output / "outcome_matrix.png", dpi=180)
    plt.close(fig)


def report_text(result):
    count = result["episodes"]
    fixed_count, adaptive_count = result["reused_fixed_episodes"], result["capped_episodes"]
    settings = result["bootstrap"]["physical_settings"]
    lines = ["# SmolVLA: fixed and capped adaptive replanning", "",
             f"This matched comparison contains **{count} episodes**: {fixed_count} retained fixed-interval controls "
             f"and {adaptive_count} capped-DVAC episodes, covering the same {settings} task/initial-state settings "
             "with policy seed 11. There are five fixed intervals (1, 5, 10, 25, 50) and four adaptive caps (5, 10, 25, 50). "
             "The checkpoint generates 50 actions with ten flow-matching steps for every method.", "",
             "## Matched completion and model calls", "",
             "| Method | Completed | Completion, descriptive 95% interval | Calls/episode, descriptive 95% interval | Calls/100 actions |",
             "|---|---:|---:|---:|---:|"]
    for row in result["methods"]:
        lines.append(f"| {label(row['method'])} | {row['successes']}/{row['episodes']} | "
                     f"{100*row['success_mean']:.1f}% [{100*row['success_ci_low']:.1f}, {100*row['success_ci_high']:.1f}] | "
                     f"{row['model_calls_mean']:.1f} [{row['model_calls_ci_low']:.1f}, {row['model_calls_ci_high']:.1f}] | "
                     f"{row['calls_per_100_actions_mean']:.2f} |")
    lines += ["", "![Completion and model calls](success_calls_frontier.png)", "",
              "Calls per episode include early stopping on success. Calls per 100 actions use the ratio of total "
              "calls to total executed actions, so the two measures answer different questions. The observed Pareto "
              "set in `observed_frontier.csv` describes sample means; it does not establish a statistically superior method.", "",
              "## Designated comparisons", ""]
    comparisons = {(row["method"], row["reference"]): row for row in result["contrasts"]}
    for key in DESIGNATED:
        row = comparisons[key]
        lines.append(f"**{label(row['method'])} versus {label(row['reference'])}:** completion differed by "
                     f"{100*row['success_difference']:+.1f} percentage points "
                     f"(descriptive paired 95% interval {100*row['success_ci_low']:+.1f} to {100*row['success_ci_high']:+.1f}); "
                     f"model calls differed by {row['model_calls_difference']:+.1f} per episode. "
                     f"There were {row['method_only_success']} settings where only {label(row['method'])} succeeded, "
                     f"and {row['reference_only_success']} with the reverse outcome.")
        lines.append("")
    lines += ["DVAC cap 10 versus fixed K=10 tests the value of sometimes updating earlier under the same maximum "
              "execution interval. A cap-10 policy cannot reduce calls per executed action below the fixed-10 "
              "schedule: its prefixes are at most ten actions. Cap 10 versus cap 50 tests whether restricting long "
              "open-loop prefixes changes completion and cost. Fixed K=1 and K=5 help locate the overall trade-off.", "",
              "![Paired completion differences](paired_success_differences.png)", "",
              "## Task results and every matched outcome", "", "![Results by task](task_success.png)", "",
              "![Every outcome and call count](outcome_matrix.png)", "",
              "The complete numerical task table is in `by_task.csv`. `outcome_matrix.csv` contains every task, "
              "initial state, seed, method outcome and call count; no episodes are selected according to success.", "",
              "## What the execution cap changes", "",
              "DVAC calculates one variance for each of the 50 future actions from its last five internal clean-action "
              "estimates. The threshold uses all 50 variance values from each of up to five preceding calls. "
              "The chosen uncapped prefix is then limited by `K = min(execution_cap, raw_DVAC_K)`. The horizon, "
              "variance calculation, alpha=2, history length, and mandatory first-call K=1 are unchanged. "
              "The cap does not shorten the model prediction or remove future actions from the threshold statistics.", "",
              "| Cap | Mean raw DVAC K | Mean selected K | Mean executed prefix | Calls limited by cap | K=1 after initialization |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in result["cap_behavior"]:
        k1 = row["nonbootstrap_k1_fraction"]
        k1_text = f"{100*k1:.1f}%" if k1 is not None else "N/A"
        lines.append(f"| {row['execution_cap']} | {row['raw_chosen_k_mean_per_call']:.2f} | {row['chosen_k_mean_per_call']:.2f} | "
                     f"{row['executed_prefix_mean_per_call']:.2f} | {100*row['cap_applied_fraction']:.1f}% | "
                     f"{k1_text} |")
    lines += ["", "Mean K values above weight policy calls equally. Executed prefixes may be shorter than selected "
              "prefixes because an episode terminates. Raw K statistics across caps come from different trajectories "
              "and observations; they are not a replay of an identical variance sequence.", "",
              "![Selected and executed prefix lengths](cap_horizon_behavior.png)", "",
              "## Reuse checks and timing", "",
              "The retained fixed controls match the adaptive experiment's checkpoint and backbone revisions, base "
              "policy source hash, dependency lock, package versions, simulator asset revision, tasks and limits, "
              "precision, observation processing, 20 Hz control, settling and sampling-seed rule. The independent "
              "verification checks identical initial-state and observation hashes and identical complete first action "
              "predictions across the matched controls and adaptive runs. Raw logs and their saved checksums remain intact.", "",
              "Historical controls used six Torch CPU threads; adaptive collection uses one and records internal "
              "flow-matching diagnostics. Their wall times and inference times therefore do not provide a controlled "
              "fixed-versus-adaptive speed comparison. The common cost measure is model calls. "
              "`adaptive_timing.csv` reports timing only among the four adaptive caps, which share instrumentation "
              "and run conditions. The synchronous simulator pauses during inference.", "",
              "## Statistical interpretation", "",
              f"Intervals use {result['bootstrap']['draws']:,} paired bootstrap draws. Each draw resamples initial states "
              "with replacement separately within each task, keeping every method and all seed repetitions together. "
              "Task means receive equal weight. Ratios are recomputed in each draw. These are descriptive intervals "
              "for the observed design, not multiplicity-adjusted confirmatory tests. The state outcomes were already "
              "examined when developing the comparison, so this is not a held-out evaluation. No p-value or claim "
              "of equivalence is made. A degenerate bootstrap interval can occur when all observed states have the "
              "same outcome; it does not guarantee that unseen states will behave identically.", "",
              "Only one policy seed is included in this matched cohort. The results quantify variation across "
              "initial situations but do not establish robustness to generation noise. The three selected tasks "
              "are fixed, so the intervals do not estimate uncertainty over arbitrary new task families. "
              "Internal denoising variance is not a calibrated failure probability; stable internal predictions "
              "can still lead to inaccurate grasps. Gripper close-command counts are action diagnostics, not "
              "verified grasp-error labels. Changing K combines new observations with stochastic action regeneration.", "",
              "All generated tables and figures are bound to `analysis.json` by `artifacts.json`. "
              "`verification.json` records full coverage, action/decision replay, source hashes and matching checks.", ""]
    return "\n".join(lines)


def analyze(root, output, *, draws=10000, seed=20261024, decode_videos=False, plots=True, reference_roots=None):
    from scripts.verify_capped import load_and_verify

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Analysis and raw evidence must use separate directories")
    references = reference_roots if reference_roots is not None else (PROJECT / "outputs/compact", PROJECT / "outputs/validation")
    if any(output.is_relative_to(Path(path).resolve()) or Path(path).resolve().is_relative_to(output) for path in references):
        raise ValueError("Analysis must not overwrite retained fixed evidence")
    cfg, records, verification = load_and_verify(root, decode_videos=decode_videos, reference_roots=reference_roots)
    result = cluster_statistics(records, draws=draws, seed=seed)
    behavior, histogram, decisions, timing = cap_behavior(records)
    fixed = sum(record["summary"]["method"].startswith("fixed_") for record in records)
    adaptive = len(records) - fixed
    if len(records) != 270 or fixed != 150 or adaptive != 120:
        raise ValueError("This report requires exactly 150 retained fixed and 120 capped adaptive episodes")
    result.update(complete=True, episodes=len(records), reused_fixed_episodes=fixed, capped_episodes=adaptive,
                  config_fingerprint=cfg.fingerprint, cap_behavior=behavior, adaptive_timing=timing,
                  observed_frontier=frontier(result["methods"]), analysis_source_sha256=sha256(__file__),
                  verification={key: value for key, value in verification.items() if key not in ("evidence", "episodes")},
                  timing_comparability="Fixed versus adaptive timing is confounded by diagnostic instrumentation and Torch CPU thread count; compare model calls instead.")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "verification.json", verification)
    atomic_json(output / "analysis.json", result)
    tables = {"methods.csv": result["methods"], "by_task.csv": result["by_task"], "paired_contrasts.csv": result["contrasts"],
              "outcome_matrix.csv": result["outcome_matrix"], "cap_behavior.csv": behavior,
              "interval_histogram.csv": histogram, "decisions.csv": decisions,
              "adaptive_timing.csv": timing, "observed_frontier.csv": result["observed_frontier"]}
    tables["episodes.csv"] = [{
        **{key: record["summary"][key] for key in ("episode_id", "suite", "task", "state", "policy_seed", "method",
                                                   "success", "model_calls", "steps", "eef_path_m", "gripper_close_commands")},
        "origin": record["origin"], "raw_source_root": record["raw_source_root"],
    } for record in records]
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    if plots:
        make_plots(output, result)
    text = report_text(result)
    if not plots:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("![")) + "\n"
    (output / "report.md").write_text(text)
    files = ["analysis.json", "verification.json", "report.md", *tables]
    if plots:
        files += ["success_calls_frontier.png", "task_success.png", "cap_horizon_behavior.png",
                  "paired_success_differences.png", "outcome_matrix.png"]
    atomic_json(output / "artifacts.json", {
        "statistics_filename": "analysis.json", "statistics_sha256": sha256(output / "analysis.json"),
        "analysis_source_sha256": sha256(__file__), "config_fingerprint": cfg.fingerprint,
        "files": {name: sha256(output / name) for name in files},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--decode-videos", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--reference-root", action="append", type=Path)
    args = parser.parse_args()
    result = analyze(args.input, args.output, draws=args.draws, seed=args.seed, decode_videos=args.decode_videos,
                     plots=not args.no_plots, reference_roots=args.reference_root)
    print(json.dumps({"episodes": result["episodes"], "complete": result["complete"],
                      "methods": [{key: row[key] for key in ("method", "successes", "episodes", "model_calls_mean")}
                                  for row in result["methods"]]}, indent=2))


if __name__ == "__main__":
    main()
