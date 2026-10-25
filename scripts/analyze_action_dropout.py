#!/usr/bin/env python3
"""Analyze balanced translation-command-dropout panels and retained p=0 runs.

Incomplete method panels are inventoried separately and never enter comparison
estimates. No rollout, policy call, or model fitting is performed by this script.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.analyze_capped import METHODS, TASK_NAMES, mean, sha256, write_csv
from vla_replan.storage import atomic_json

PROBABILITIES = (0.0, 0.1, 0.3)
TASKS = (
    ("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    ("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"),
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"),
)
BASE_METRICS = ("success", "model_calls", "steps", "translation_drop_count", "drop_fraction_per_episode")
METRICS = (*BASE_METRICS, "calls_per_100_actions", "realized_drop_fraction")


def label(method):
    return (f"DVAC C={method.rsplit('_', 1)[1]}" if method.startswith("dvac_")
            else f"Fixed K={method.rsplit('_', 1)[1]}")


def identity(row):
    return row["suite"], row["task"], row["state"], row["policy_seed"], row["method"], row["dropout_p"]


def load_existing_export(path):
    """Verify the published baseline's hashes and reconstruct its episode grid."""
    path = Path(path)
    artifacts = json.loads((path.parent / "artifacts.json").read_text())
    csv_path = path.parent / "episodes.csv"
    for filename, source in (("analysis.json", path), ("episodes.csv", csv_path)):
        if artifacts.get("files", {}).get(filename) != sha256(source):
            raise ValueError(f"Retained baseline export hash mismatch: {filename}")
    if artifacts.get("statistics_sha256") != sha256(path):
        raise ValueError("Retained statistics hash mismatch")
    analysis = json.loads(path.read_text())
    if (analysis.get("complete") is not True or analysis.get("episodes") != 270
            or analysis.get("verification", {}).get("complete") is not True):
        raise ValueError("The retained baseline must contain all 270 verified episodes")
    if (len(analysis.get("methods", [])) != len(METHODS)
            or {row["method"] for row in analysis["methods"]} != set(METHODS)):
        raise ValueError("Retained analysis must contain the nine expected methods")
    rows = []
    with csv_path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("success") not in ("True", "False"):
                raise ValueError("Retained success must be boolean")
            rows.append({"episode_id": raw["episode_id"], "suite": raw["suite"], "task": raw["task"],
                         "method": raw["method"], "state": int(raw["state"]), "policy_seed": int(raw["policy_seed"]),
                         "success": raw["success"] == "True", "model_calls": int(raw["model_calls"]),
                         "steps": int(raw["steps"]), "dropout_p": 0.0, "translation_drop_count": 0})
    cells, _ = panel_inventory([{"summary": row} for row in rows])
    if len(cells) != 270:
        raise ValueError("Retained baseline grid is incomplete")
    for old in analysis["methods"]:
        selected = [row for row in rows if row["method"] == old["method"]]
        if len(selected) != 30 or sum(row["success"] for row in selected) != old["successes"]:
            raise ValueError("Retained baseline totals disagree with its episode export")
        for metric in ("success", "model_calls", "steps"):
            if not math.isclose(mean(float(row[metric]) for row in selected), old[metric + "_mean"], abs_tol=1e-10):
                raise ValueError("Retained baseline means disagree with its episode export")
    return rows, {"analysis_sha256": sha256(path), "episodes_csv_sha256": sha256(csv_path),
                  "config_fingerprint": analysis["config_fingerprint"]}


def validated_probabilities(probabilities):
    """Require the retained control plus distinct, increasing integer percentages."""
    from action_dropout.config import validate_probability

    probabilities = tuple(validate_probability(value) for value in probabilities)
    if (len(probabilities) < 2 or probabilities[0] != 0
            or probabilities != tuple(sorted(set(probabilities)))):
        raise ValueError("Analysis probabilities must start at p=0 and contain unique increasing positive probabilities")
    return probabilities


def panel_inventory(records, *, probabilities=PROBABILITIES):
    """Validate every completed row and inventory all nine complete/partial panels."""
    probabilities = validated_probabilities(probabilities)
    cells = {}
    for record in records:
        row = dict(record["summary"])
        probability = row.get("dropout_p")
        if type(probability) not in (int, float) or probability not in probabilities:
            raise ValueError("Unexpected dropout probability")
        if row.get("method") not in METHODS or (row.get("suite"), row.get("task")) not in TASKS:
            raise ValueError("Unexpected method or task")
        if (type(row.get("state")) is not int or row["state"] not in range(10)
                or type(row.get("policy_seed")) is not int or row["policy_seed"] != 11):
            raise ValueError("Expected physical states 0–9 and policy seed 11")
        if type(row.get("success")) is not bool:
            raise ValueError("Success must be boolean")
        for field in ("steps", "model_calls"):
            if type(row.get(field)) is not int or row[field] < 1:
                raise ValueError(f"Invalid positive integer metric: {field}")
        drops = row.get("translation_drop_count", 0 if probability == 0 else None)
        if type(drops) is not int or not 0 <= drops <= row["steps"]:
            raise ValueError("Invalid translation-drop count")
        if probability == 0 and drops != 0:
            raise ValueError("The p=0 baseline cannot contain translation drops")
        row["translation_drop_count"] = drops
        row["drop_fraction_per_episode"] = drops / row["steps"]
        key = identity(row)
        if key in cells:
            raise ValueError("Duplicate method/probability/physical-start episode")
        cells[key] = row
    baseline_expected = {(suite, task, state, 11, method, 0.0)
                         for suite, task in TASKS for state in range(10) for method in METHODS}
    if not baseline_expected.issubset(cells):
        raise ValueError("All 270 retained p=0 episodes are required")
    inventory = []
    for method in METHODS:
        missing = []
        counts = {}
        for probability in probabilities[1:]:
            expected = [(suite, task, state, 11, method, probability) for suite, task in TASKS for state in range(10)]
            absent = [key for key in expected if key not in cells]
            counts[f"p{round(probability * 100):02d}"] = 30 - len(absent)
            missing.extend({"suite": key[0], "task": key[1], "state": key[2], "policy_seed": key[3],
                            "dropout_p": key[5]} for key in absent)
        inventory.append({"method": method, "balanced_complete": not missing,
                          "collection_status": "complete" if not missing else "partial" if sum(counts.values()) else "not started",
                          "new_completed_episodes": sum(counts.values()),
                          "new_expected_episodes": 30 * (len(probabilities) - 1),
                          "completed_by_probability": counts, "missing": missing})
    return cells, inventory


def statistics(records, *, probabilities=PROBABILITIES, draws=2000, seed=20261025):
    """Paired task-stratified physical-state bootstrap across every included cell."""
    import numpy as np

    if type(draws) is not int or draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    probabilities = validated_probabilities(probabilities)
    cells, inventory = panel_inventory(records, probabilities=probabilities)
    panel_size = 30 * (len(probabilities) - 1)
    methods = tuple(row["method"] for row in inventory if row["balanced_complete"])
    output = {
        "balanced_methods": list(methods), "method_panels": inventory, "probabilities": list(probabilities),
        "balanced_new_episodes": panel_size * len(methods), "balanced_baseline_episodes": 30 * len(methods),
        "balanced_total_episodes": 30 * len(probabilities) * len(methods), "retained_baseline_episodes": 270,
        "completed_new_episodes": sum(row["new_completed_episodes"] for row in inventory),
        "excluded_partial_new_episodes": sum(row["new_completed_episodes"] for row in inventory if not row["balanced_complete"]),
        "all_nine_methods_complete": len(methods) == len(METHODS),
        "methods": [], "by_task": [], "degradation": [], "matched_method_contrasts": [],
        "estimands": {
            "model_calls_mean": "Mean model calls across all completed attempts, including failures",
            "model_calls_success_mean": "Mean model calls conditional on episode success; null when model_calls_success_n is zero. Successful subsets can differ across methods and probabilities; no paired contrast or bootstrap interval is reported for this conditional mean",
            "model_calls_success_n": "Number of successful episodes contributing to the conditional mean",
        },
        "bootstrap": {"draws": draws, "seed": seed, "confidence_level": .95,
                      "unit": "physical task/initial-state setting; preserve every included probability and method together",
                      "strata": "tasks with equal weights; ten states resampled within each task",
                      "interpretation": "Descriptive paired percentile intervals; no adjusted hypothesis tests or causal generalization beyond the specified disturbance model"},
    }
    if not methods:
        output["analysis_status"] = f"No complete {panel_size}-episode method panel; partial rows are inventory only"
        return output
    rng = np.random.default_rng(seed)

    def extend(matrix):
        calls = 100 * matrix[..., 1] / matrix[..., 2]
        fraction = matrix[..., 3] / matrix[..., 2]
        return np.concatenate((matrix, calls[..., None], fraction[..., None]), axis=-1)

    def summary_rows(point, boot, *, task="all", suite="all"):
        rows = []
        for mi, method in enumerate(methods):
            for pi, probability in enumerate(probabilities):
                selected = [row for key, row in cells.items() if key[4:] == (method, probability)
                            and (task == "all" or key[:2] == (suite, task))]
                successful_calls = [row["model_calls"] for row in selected if row["success"]]
                row = {"method": method, "dropout_p": probability, "suite": suite, "task": task,
                       "episodes": len(selected), "successes": sum(r["success"] for r in selected),
                       "physical_settings": len(selected),
                       "model_calls_success_n": len(successful_calls),
                       "model_calls_success_mean": mean(successful_calls)}
                for metric_index, metric in enumerate(METRICS):
                    low, high = np.quantile(boot[:, mi, pi, metric_index], [.025, .975])
                    row[metric + "_mean"] = float(point[mi, pi, metric_index])
                    row[metric + "_ci_low"] = float(low)
                    row[metric + "_ci_high"] = float(high)
                rows.append(row)
        return rows

    task_points, task_bootstraps = [], []
    for suite, task in TASKS:
        matrix = np.array([[[[float(cells[(suite, task, state, 11, method, probability)][metric])
                             for metric in BASE_METRICS] for probability in probabilities]
                           for method in methods] for state in range(10)])
        indices = rng.integers(0, 10, size=(draws, 10))
        point, boot = matrix.mean(axis=0), matrix[indices].mean(axis=1)
        task_points.append(point)
        task_bootstraps.append(boot)
        output["by_task"].extend(summary_rows(extend(point), extend(boot), task=task, suite=suite))
    point = extend(np.mean(task_points, axis=0))
    boot = extend(np.mean(task_bootstraps, axis=0))
    output["methods"] = summary_rows(point, boot)

    def difference(mi, pi, reference_mi, reference_pi):
        result = {"method": methods[mi], "dropout_p": probabilities[pi],
                  "reference_method": methods[reference_mi], "reference_p": probabilities[reference_pi],
                  "paired_physical_settings": 30}
        for metric_index, metric in enumerate(METRICS):
            delta = boot[:, mi, pi, metric_index] - boot[:, reference_mi, reference_pi, metric_index]
            low, high = np.quantile(delta, [.025, .975])
            result[metric + "_difference"] = float(point[mi, pi, metric_index] - point[reference_mi, reference_pi, metric_index])
            result[metric + "_ci_low"] = float(low)
            result[metric + "_ci_high"] = float(high)
        discordance = defaultdict(int)
        for suite, task in TASKS:
            for state in range(10):
                left = cells[(suite, task, state, 11, methods[mi], probabilities[pi])]["success"]
                right = cells[(suite, task, state, 11, methods[reference_mi], probabilities[reference_pi])]["success"]
                discordance[left, right] += 1
        result.update(method_only_success=discordance[True, False], reference_only_success=discordance[False, True],
                      both_success=discordance[True, True], both_failure=discordance[False, False])
        return result

    for mi, method in enumerate(methods):
        for pi in range(1, len(probabilities)):
            output["degradation"].append(difference(mi, pi, mi, 0))
        if method.startswith("dvac_cap_"):
            fixed = "fixed_" + method.rsplit("_", 1)[1]
            if fixed in methods:
                for pi in range(len(probabilities)):
                    comparison = difference(mi, pi, methods.index(fixed), pi)
                    comparison["comparison_role"] = "Same nominal cap/interval; realized computational budgets can differ"
                    output["matched_method_contrasts"].append(comparison)
    output["analysis_status"] = "Balanced completed method panels only"
    return output


def make_plots(output, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, ScalarFormatter

    if not result["balanced_methods"]:
        return []
    probabilities = validated_probabilities(result["probabilities"])
    colors = {method: color for method, color in zip(METHODS,
              ("#243c59", "#3974aa", "#65a2c3", "#408b83", "#90bdb7", "#d18b43", "#be673b", "#a74755", "#7c467c"), strict=True)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    files = []

    def save(fig, stem):
        for extension in ("png", "svg"):
            name = stem + "." + extension
            fig.savefig(output / name, dpi=180, facecolor="white")
            files.append(name)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.97, bottom=.23, top=.78, wspace=.12)
    for index, (prefix, title) in enumerate((("fixed_", "Fixed execution intervals"), ("dvac_", "DVAC execution caps"))):
        ax = axes[index]
        for method in result["balanced_methods"]:
            if not method.startswith(prefix):
                continue
            rows = sorted((row for row in result["methods"] if row["method"] == method), key=lambda row: row["dropout_p"])
            ax.plot([row["dropout_p"] for row in rows], [row["success_mean"] for row in rows],
                    marker="o" if index == 0 else "D", lw=2.3, markersize=7, color=colors[method], label=label(method))
        ax.set(title=title, xlabel="Probability of dropping a translation command", ylim=(0, 1.04),
               xticks=probabilities, xlim=(-.012, probabilities[-1] + .012))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.2)
        if ax.lines:
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
        else:
            ax.text(.5, .5, "No complete method panel", transform=ax.transAxes, ha="center")
    axes[0].set_ylabel("Task success rate")
    fig.text(.075, .93, "Task completion under translation-command dropout", fontsize=24, weight="bold")
    fig.text(.075, .865, "30 matched starts per point. Translation is zeroed on dropped steps; rotation and gripper commands remain.", fontsize=13)
    fig.text(.075, .055, "Point estimates; paired 95% intervals are in the tables. Only methods complete at every configured probability are included.", fontsize=11, color="#566570")
    save(fig, "success_vs_dropout")

    columns = min(3, len(probabilities))
    rows = math.ceil(len(probabilities) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 6 if rows == 1 else 9), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.075, right=.97, bottom=.14, top=.83, hspace=.32, wspace=.15)
    for ax, probability in zip(axes.flat, probabilities):
        rows = [row for row in result["methods"] if row["dropout_p"] == probability]
        for index, row in enumerate(sorted(rows, key=lambda row: (row["success_mean"], row["model_calls_mean"]))):
            method = row["method"]
            ax.scatter(row["model_calls_mean"], row["success_mean"], color=colors[method], s=65,
                       marker="D" if method.startswith("dvac_") else "o", zorder=3)
            # Short offset labels are deterministic; the exported figure is visually reviewed before delivery.
            ax.annotate(label(method), (row["model_calls_mean"], row["success_mean"]),
                        xytext=(6, 8 + (index % 3) * 14), textcoords="offset points", fontsize=9,
                        color=colors[method], arrowprops={"arrowstyle": "-", "lw": .5, "color": colors[method]})
        ax.set_xscale("log")
        ax.set(title=f"Dropout probability {probability:.0%}", ylim=(0, 1.18), xlim=(3.5, 390))
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_xticks([5, 10, 20, 50, 100, 200])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.2)
    for ax in list(axes.flat)[len(probabilities):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Mean model calls per episode (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Task success rate")
    fig.text(.075, .93, "Success versus model calls under command dropout", fontsize=24, weight="bold")
    fig.text(.075, .055, "Same 30 starts at each probability; all successful and failed episodes contribute to call counts.", fontsize=11, color="#566570")
    save(fig, "success_vs_calls")
    return files


def analyze(root, output, *, existing_analysis=PROJECT / "outputs/capped-report/analysis.json",
            baseline_root=PROJECT / "outputs/capped", draws=2000, seed=20261025, plots=True):
    from scripts.verify_action_dropout import load_and_verify

    root, output = Path(root).resolve(), Path(output).resolve()
    existing_analysis = Path(existing_analysis).resolve()
    for source in (root, Path(baseline_root).resolve(), existing_analysis.parent):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Analysis output must be separate from raw and retained evidence")
    baseline, provenance = load_existing_export(existing_analysis)
    config, records, verification = load_and_verify(root, baseline_root=baseline_root)
    expected = {identity(row): row for row in baseline}
    actual = {identity(record["summary"]): record["summary"] for record in records if record["summary"]["dropout_p"] == 0}
    if set(actual) != set(expected):
        raise ValueError("Verified raw baseline differs from the retained episode export")
    for key, row in expected.items():
        if any(actual[key][field] != row[field] for field in ("success", "steps", "model_calls")):
            raise ValueError("Verified raw baseline measurements differ from retained published values")
    probabilities = (0.0, *sorted(config.probabilities))
    result = statistics(records, probabilities=probabilities, draws=draws, seed=seed)
    result.update(config_fingerprint=config.fingerprint, existing_evidence=provenance,
                  verification_complete=verification.get("complete", False),
                  source_sha256=sha256(__file__),
                  limitations=[
                      f"Only balanced methods completed at probabilities {list(probabilities)} enter the estimates; partial method panels are not compared.",
                      "One paired disturbance seed and one policy seed are used; intervals describe variation across the selected starts, not arbitrary disturbance sequences.",
                      "Realized drop fraction is translation-drop count divided by executed steps; early termination and finite samples can make it differ from nominal p.",
                      "A dropped command has zero translation but retains its rotation and gripper components; this is not observation delay or movement of the target object.",
                      "Success degradation is dropout success minus the same method's retained p=0 success; a negative value means lower completion.",
                      "Historical and new timing conditions differ; model-call counts provide the cross-method cost comparison, not wall-clock speed.",
                  ])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "analysis.json", result)
    atomic_json(output / "verification.json", verification)
    tables = {"methods.csv": result["methods"], "by_task.csv": result["by_task"],
              "paired_degradation.csv": result["degradation"], "matched_method_contrasts.csv": result["matched_method_contrasts"],
              "panel_coverage.csv": [{"method": row["method"], "balanced_complete": row["balanced_complete"],
                                      "collection_status": row["collection_status"],
                                      "new_completed_episodes": row["new_completed_episodes"],
                                      **row["completed_by_probability"]} for row in result["method_panels"]]}
    for name, rows in tables.items():
        if rows:
            write_csv(output / name, rows)
        elif (output / name).exists():
            (output / name).unlink()
    figures = make_plots(output, result) if plots else []
    files = ["analysis.json", "verification.json", *[name for name, rows in tables.items() if rows], *figures]
    atomic_json(output / "artifacts.json", {"statistics_sha256": sha256(output / "analysis.json"),
                                          "analysis_source_sha256": sha256(__file__),
                                          "files": {name: sha256(output / name) for name in files}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "outputs/action-dropout")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/action-dropout-analysis")
    parser.add_argument("--existing-analysis", type=Path, default=PROJECT / "outputs/capped-report/analysis.json")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT / "outputs/capped")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze(args.root, args.output, existing_analysis=args.existing_analysis, baseline_root=args.baseline_root,
                     draws=args.draws, seed=args.seed, plots=not args.no_plots)
    print(json.dumps({key: result[key] for key in ("balanced_methods", "balanced_new_episodes",
                                                  "completed_new_episodes", "excluded_partial_new_episodes")}, indent=2))


if __name__ == "__main__":
    main()
