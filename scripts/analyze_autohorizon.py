#!/usr/bin/env python3
"""Paired analysis of AutoHorizon caps and the retained fixed/DVAC comparison.

The existing 270 episodes are read from their hash-bound analysis export. New
AutoHorizon evidence must pass its independent verifier before it is combined.
All methods retain the same 30 physical settings; no outcome-based selection is
performed, and historical wall times are not treated as controlled comparisons.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.analyze_capped import (METHODS as EXISTING_METHODS, METRICS,
                                    cluster_statistics, mean, paired_layout,
                                    sha256, write_csv)
from vla_replan.storage import atomic_json

CAPS = (5, 10, 25, 50)
AUTO_METHODS = tuple(f"autohorizon_cap_{cap}" for cap in CAPS)
METHODS = (*EXISTING_METHODS, *AUTO_METHODS)
PRIMARY_CONTRASTS = tuple(
    (f"autohorizon_cap_{cap}", comparator)
    for cap in CAPS
    for comparator in (f"dvac_cap_{cap}", f"fixed_{cap}")
)
TASKS = {
    ("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    ("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"),
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"),
}
TIMING_FIELDS = ("inference_s", "selector_s", "simulation_s", "compute_s", "episode_wall_s")


def label(method):
    if method.startswith("autohorizon_cap_"):
        return f"AutoHorizon C={method.rsplit('_', 1)[1]}"
    if method.startswith("dvac_cap_"):
        return f"DVAC C={method.rsplit('_', 1)[1]}"
    return f"Fixed K={method.rsplit('_', 1)[1]}"


def _pair(row):
    return row["suite"], row["task"], row["state"], row["policy_seed"]


def require_complete_layout(records, methods):
    """Require the frozen three-task, ten-state, seed-11 grid, not just 30 rows."""
    cells, tasks, seeds = paired_layout(records, methods)
    if set(tasks) != TASKS or any(states != list(range(10)) for states in tasks.values()) or seeds != [11]:
        raise ValueError("Expected the same three tasks, states 0–9 and policy seed 11")
    if len(cells) != 30 * len(methods):
        raise ValueError("Incomplete frozen comparison grid")
    return cells


def load_existing(path):
    """Read immutable per-episode metrics, rejecting missing or stale exports."""
    path = Path(path)
    artifacts_path = path.parent / "artifacts.json"
    csv_path = path.parent / "episodes.csv"
    artifacts = json.loads(artifacts_path.read_text())
    for source, key in ((path, "analysis.json"), (csv_path, "episodes.csv")):
        if artifacts.get("files", {}).get(key) != sha256(source):
            raise ValueError(f"Existing evidence export hash mismatch: {source.name}")
    if artifacts.get("statistics_sha256") != sha256(path):
        raise ValueError("Existing statistics hash mismatch")
    result = json.loads(path.read_text())
    if (result.get("complete") is not True or result.get("episodes") != 270
            or result.get("reused_fixed_episodes") != 150 or result.get("capped_episodes") != 120
            or result.get("verification", {}).get("complete") is not True
            or artifacts.get("config_fingerprint") != result.get("config_fingerprint")):
        raise ValueError("Existing comparison must contain all 270 verified episodes")
    records = []
    with csv_path.open(newline="") as stream:
        for source in csv.DictReader(stream):
            if source.get("success") not in ("True", "False"):
                raise ValueError("Existing success must be a boolean CSV value")
            summary = dict(source)
            summary["success"] = source["success"] == "True"
            for field in ("state", "policy_seed", "model_calls", "steps", "gripper_close_commands"):
                summary[field] = int(source[field])
            summary["eef_path_m"] = float(source["eef_path_m"])
            records.append({"summary": summary, "origin": source.get("origin", "retained")})
    require_complete_layout(records, EXISTING_METHODS)
    existing_rows = {row["method"]: row for row in result["methods"]}
    if set(existing_rows) != set(EXISTING_METHODS):
        raise ValueError("Existing analysis has unexpected methods")
    for method in EXISTING_METHODS:
        group = [r["summary"] for r in records if r["summary"]["method"] == method]
        row = existing_rows[method]
        if row["episodes"] != 30 or row["successes"] != sum(r["success"] for r in group):
            raise ValueError("Existing analysis and episode export disagree on outcomes")
        for metric in METRICS:
            value = mean(float(r[metric]) for r in group)
            if not math.isclose(row[metric + "_mean"], value, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Existing analysis and episode export disagree: {method}/{metric}")
    return records, result, {
        "analysis_sha256": sha256(path), "episodes_csv_sha256": sha256(csv_path),
        "artifacts_sha256": sha256(artifacts_path), "config_fingerprint": result["config_fingerprint"],
    }


def comparison_statistics(existing, new, *, draws=10000, seed=20261024):
    require_complete_layout(existing, EXISTING_METHODS)
    require_complete_layout(new, AUTO_METHODS)
    old_pairs = {_pair(record["summary"]) for record in existing}
    new_pairs = {_pair(record["summary"]) for record in new}
    if old_pairs != new_pairs:
        raise ValueError("AutoHorizon and retained methods have unmatched physical settings")
    for record in new:
        row = record["summary"]
        cap = int(row["method"].rsplit("_", 1)[1])
        if row.get("execution_cap") != cap or row.get("k") is not None:
            raise ValueError("AutoHorizon method identity and execution cap disagree")
    result = cluster_statistics([*existing, *new], methods=METHODS, draws=draws, seed=seed)
    for row in result["contrasts"]:
        is_primary = (row["method"], row["reference"]) in PRIMARY_CONTRASTS
        row["designated_contrast"] = is_primary
        row["comparison_role"] = "primary descriptive comparison" if is_primary else "exploratory"
    if sum(row["designated_contrast"] for row in result["contrasts"]) != 8:
        raise ValueError("The eight designated AutoHorizon contrasts are missing")
    result["bootstrap"]["interpretation"] = (
        "Descriptive task-stratified paired state-bootstrap intervals on previously examined states; "
        "all 13 methods remain paired within each of 30 physical settings. No multiplicity-adjusted "
        "confirmatory test, equivalence claim or calibrated-confidence claim is made."
    )
    return result


def horizon_behavior(records):
    """Distinguish selected K from executed prefixes truncated at termination."""
    summaries, histograms, decisions, timing = [], [], [], []
    for method in AUTO_METHODS:
        group = [record for record in records if record["summary"]["method"] == method]
        chosen, raw, executed, first = [], [], [], []
        histogram = Counter()
        cap = int(method.rsplit("_", 1)[1])
        for record in group:
            row, chunks = record["summary"], record["chunks"]
            if len(chunks) != row["model_calls"]:
                raise ValueError("Chunk count disagrees with model calls")
            for call_index, chunk in enumerate(chunks):
                decision = chunk["decision"]
                k, raw_k = decision["chosen_k"], decision["raw_chosen_k"]
                n = chunk["executed_steps"]
                if (type(k) is not int or type(raw_k) is not int or type(n) is not int
                        or not 1 <= raw_k <= 50 or k != min(cap, raw_k) or not 1 <= n <= k):
                    raise ValueError("Invalid AutoHorizon selected or executed prefix")
                if decision.get("bootstrap") is True:
                    raise ValueError("AutoHorizon must select K on the first call without forced bootstrap")
                chosen.append(k)
                raw.append(raw_k)
                executed.append(n)
                if call_index == 0:
                    first.append(k)
                histogram[(call_index == 0, raw_k, k, n)] += 1
                decisions.append({
                    "episode_id": row["episode_id"], "method": method, "step": chunk["step"],
                    "call_index": call_index, "first_call": call_index == 0,
                    "raw_chosen_k": raw_k, "chosen_k": k, "execution_cap": cap,
                    "executed_steps": n, "cap_applied": raw_k > cap,
                })
            if sum(c["executed_steps"] for c in chunks) != row["steps"]:
                raise ValueError("Executed prefixes disagree with episode length")
        summaries.append({
            "method": method, "execution_cap": cap, "episodes": len(group), "model_calls": len(chosen),
            "raw_chosen_k_mean_per_call": mean(raw), "chosen_k_mean_per_call": mean(chosen),
            "executed_prefix_mean_per_call": mean(executed), "chosen_k_min": min(chosen),
            "chosen_k_max": max(chosen), "first_call_chosen_k_mean": mean(first),
            "k1_fraction": mean(k == 1 for k in chosen), "at_cap_fraction": mean(k == cap for k in chosen),
            "cap_applied_fraction": mean(k > cap for k in raw),
        })
        histograms.extend({"method": method, "first_call": initial, "raw_chosen_k": raw_k,
                           "chosen_k": k, "executed_steps": n, "calls": count}
                          for (initial, raw_k, k, n), count in sorted(histogram.items()))
        timed = {"method": method, "episodes": len(group), "timing_population": "AutoHorizon only"}
        for field in TIMING_FIELDS:
            values = [record["summary"].get(field) for record in group]
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Invalid AutoHorizon runtime metric: {field}")
            timed[field + "_mean_per_episode"] = mean(values)
            timed[field + "_total"] = math.fsum(values)
        timed["inference_s_per_call"] = math.fsum(r["summary"]["inference_s"] for r in group) / len(chosen)
        timing.append(timed)
    return summaries, histograms, decisions, timing


def _place_labels(ax, rows):
    """Greedily place direct labels in display coordinates, then draw leaders."""
    import numpy as np

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    points = [ax.transData.transform((r["model_calls_mean"], 100 * r["success_mean"])) for r in rows]
    placed = []
    # Sparse, high-call points first; tie breaking is stable and independent of success.
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]["model_calls_mean"], rows[i]["method"]))
    for index in order:
        row, point = rows[index], points[index]
        text = f"{label(row['method'])}\n{100 * row['success_mean']:.1f}% · {row['model_calls_mean']:.1f} calls"
        temporary = ax.text(0, 0, text, fontsize=11.3, ha="center", va="center", linespacing=1.3)
        box = temporary.get_window_extent(renderer)
        temporary.remove()
        half_width, half_height = box.width / 2 + 8, box.height / 2 + 6
        candidates = []
        for dy in (45, -45, 85, -85, 125, -125, 170, -170, 215, -215):
            for dx in (0, 65, -65, 120, -120, 185, -185, 250, -250):
                x = float(np.clip(point[0] + dx, bounds.x0 + half_width, bounds.x1 - half_width))
                y = float(np.clip(point[1] + dy, bounds.y0 + half_height, bounds.y1 - half_height))
                candidate = (x - half_width, y - half_height, x + half_width, y + half_height)
                overlap = sum(max(0, min(candidate[2], b[2]) - max(candidate[0], b[0]))
                              * max(0, min(candidate[3], b[3]) - max(candidate[1], b[1])) for b in placed)
                point_hits = sum(candidate[0] - 8 < p[0] < candidate[2] + 8
                                 and candidate[1] - 8 < p[1] < candidate[3] + 8 for p in points)
                distance = (x - point[0]) ** 2 + (y - point[1]) ** 2
                candidates.append((overlap * 1e6 + point_hits * 1e8 + distance, x, y, candidate))
        _, x, y, chosen_box = min(candidates, key=lambda candidate: candidate[0])
        placed.append(chosen_box)
        location = ax.transData.inverted().transform((x, y))
        color, _ = style(row["method"])
        ax.annotate(text, (row["model_calls_mean"], 100 * row["success_mean"]), xytext=location,
                    ha="center", va="center", fontsize=11.3, linespacing=1.3, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2, "alpha": .94},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .8, "alpha": .6,
                                "shrinkA": 4, "shrinkB": 7})


def style(method):
    if method.startswith("autohorizon_"):
        return "#208166", "s"
    if method.startswith("dvac_"):
        return "#c76b26", "D"
    return "#2b628f", "o"


def make_plots(output, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, ScalarFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    generated = []
    for intervals in (False, True):
        fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="white")
        ax = fig.add_axes((.075, .18, .89, .64))
        ax.set_xscale("log")
        xmax = max(220, 1.18 * max(r["model_calls_ci_high"] for r in result["methods"]))
        xmin = min(3.8, .8 * min(r["model_calls_ci_low"] for r in result["methods"]))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 106)
        ax.set_xticks([tick for tick in (1, 2, 5, 10, 20, 50, 100, 200, 500) if xmin <= tick <= xmax])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", color="#e7ebef", lw=.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean model calls per episode (logarithmic scale)", labelpad=12)
        ax.set_ylabel("Task success rate", labelpad=10)
        for row in result["methods"]:
            color, marker = style(row["method"])
            x, y = row["model_calls_mean"], 100 * row["success_mean"]
            if intervals:
                ax.errorbar(x, y,
                            xerr=[[max(0, x - row["model_calls_ci_low"])],
                                  [max(0, row["model_calls_ci_high"] - x)]],
                            yerr=[[max(0, y - 100 * row["success_ci_low"])],
                                  [max(0, 100 * row["success_ci_high"] - y)]],
                            fmt="none", ecolor=color, alpha=.25, capsize=3, lw=1.4, zorder=2)
            ax.scatter(x, y, marker=marker, s=90, color=color, edgecolor="white", linewidth=.8, zorder=4)
        fig.text(.075, .935, "Task success versus model calls", fontsize=26, weight="bold", color="#182630")
        fig.text(.075, .88, "Fixed intervals", color=style("fixed_1")[0], fontsize=15, weight="bold")
        fig.text(.26, .88, "DVAC caps", color=style("dvac_cap_5")[0], fontsize=15, weight="bold")
        fig.text(.415, .88, "AutoHorizon caps", color=style("autohorizon_cap_5")[0], fontsize=15, weight="bold")
        fig.text(.965, .88, "30 matched starts per method · 390 episodes", ha="right", fontsize=12, color="#566570")
        _place_labels(ax, result["methods"])
        footer = ("Whiskers: descriptive 95% state-bootstrap intervals. Use paired contrasts to compare methods."
                  if intervals else "Point estimates. Descriptive 95% paired differences and interval plots are included in the report.")
        fig.text(.075, .065, footer, fontsize=11, color="#566570")
        fig.text(.075, .035, "C is an execution cap; selected K may be smaller. Model calls include successful and failed episodes.",
                 fontsize=11, color="#566570")
        stem = "success_vs_calls_with_intervals" if intervals else "success_vs_calls"
        for extension in ("png", "svg"):
            filename = f"{stem}.{extension}"
            fig.savefig(output / filename, facecolor="white")
            generated.append(filename)
        plt.close(fig)
    return generated


def report_text(result):
    lines = ["# AutoHorizon, DVAC and fixed replanning intervals", "",
             "This comparison adds 120 AutoHorizon episodes to 270 retained fixed-interval and DVAC episodes. "
             "Each of the 13 methods is evaluated on the same 30 physical starts: three LIBERO tasks, "
             "initial-state indices 0–9 and policy seed 11. No episodes are selected by outcome.", "",
             "The model predicts H=50 actions at each call. Fixed methods execute K actions. The adaptive "
             "methods choose an interval and then apply the execution cap C ∈ {5, 10, 25, 50}. "
             "AutoHorizon selects its interval from the first call; DVAC uses its original first-call K=1 rule. "
             "Thus the comparison evaluates each complete scheduling method, including initialization.", "",
             "## Success and computational cost", "",
             "Calls are averaged over all episodes, including failures. Calls per 100 actions are the ratio "
             "of the mean call count to the mean executed-action count, multiplied by 100. This second "
             "measure helps distinguish scheduling frequency from changes in episode length.", "",
             "| Method | Successes | Success rate [95% interval] | Calls per episode [95% interval] | Calls / 100 actions |",
             "|---|---:|---:|---:|---:|"]
    for row in result["methods"]:
        lines.append(f"| {label(row['method'])} | {row['successes']}/30 | "
                     f"{100 * row['success_mean']:.1f}% [{100 * row['success_ci_low']:.1f}, {100 * row['success_ci_high']:.1f}] | "
                     f"{row['model_calls_mean']:.2f} [{row['model_calls_ci_low']:.2f}, {row['model_calls_ci_high']:.2f}] | "
                     f"{row['calls_per_100_actions_mean']:.2f} |")
    lines += ["", "![Success versus model calls](success_vs_calls.png)", "",
              "![Success versus model calls with intervals](success_vs_calls_with_intervals.png)", "",
              "## Designated paired comparisons", "",
              "Each difference below is AutoHorizon minus the comparator at the same cap/interval. "
              "A positive success difference favours AutoHorizon; a positive call difference means "
              "AutoHorizon uses more model calls. Equal caps do not guarantee equal realized intervals "
              "or equal computational budgets. These eight comparisons were designated for reporting; "
              "their intervals are descriptive and are not adjusted for multiple comparisons.", "",
              "| Comparison | Success difference, pp [95% interval] | Call difference [95% interval] | Auto-only / comparator-only successes |",
              "|---|---:|---:|---:|"]
    contrast_map = {(row["method"], row["reference"]): row for row in result["contrasts"]}
    for pair in PRIMARY_CONTRASTS:
        row = contrast_map[pair]
        lines.append(f"| {label(row['method'])} − {label(row['reference'])} | "
                     f"{100 * row['success_difference']:+.1f} [{100 * row['success_ci_low']:+.1f}, {100 * row['success_ci_high']:+.1f}] | "
                     f"{row['model_calls_difference']:+.2f} [{row['model_calls_ci_low']:+.2f}, {row['model_calls_ci_high']:+.2f}] | "
                     f"{row['method_only_success']} / {row['reference_only_success']} |")
    lines += ["", "The remaining pairwise contrasts in `paired_contrasts.csv` are explicitly exploratory. "
              "Neither overlapping marginal whiskers nor a one-episode difference establishes equivalence "
              "or superiority. The paired differences above retain the dependence between methods on each start.", "",
              "## Results by task", "",
              "| Method | Soup into basket | Bowl onto plate | Open middle drawer |", "|---|---:|---:|---:|"]
    task_order = ("pick_up_the_alphabet_soup_and_place_it_in_the_basket",
                  "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                  "open_the_middle_drawer_of_the_cabinet")
    tasks = {(row["method"], row["task"]): row for row in result["by_task"]}
    for method in METHODS:
        lines.append("| " + label(method) + " | " + " | ".join(f"{tasks[method, task]['successes']}/10" for task in task_order) + " |")
    lines += ["", "## Selected and executed intervals", "",
              "| AutoHorizon cap | Mean raw K | Mean selected K | Mean executed prefix | First-call mean K | Calls limited by cap |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in result["horizon_behavior"]:
        lines.append(f"| {row['execution_cap']} | {row['raw_chosen_k_mean_per_call']:.2f} | "
                     f"{row['chosen_k_mean_per_call']:.2f} | {row['executed_prefix_mean_per_call']:.2f} | "
                     f"{row['first_call_chosen_k_mean']:.2f} | {100 * row['cap_applied_fraction']:.1f}% |")
    lines += ["", "These averages weight model calls equally. The executed prefix can be shorter than the "
              "selected interval when success or the episode limit terminates a rollout. Different methods "
              "observe different trajectories after their first scheduling decision.", "",
              "## Runtime and interpretation", "",
              result["timing_comparability"], "",
              "`autohorizon_timing.csv` reports runtime only for the new AutoHorizon collection. "
              "The synchronous simulator pauses during model inference; fewer calls do not establish "
              "performance under real-world observation or actuation delays.", "",
              f"Intervals use {result['bootstrap']['draws']:,} task-stratified paired bootstrap draws "
              "over the 30 physical task/initial-state settings. Each draw resamples ten states within "
              "each task and keeps all 13 methods together. Tasks have equal weight; ratios are recomputed "
              "in every draw. The selected tasks are fixed, and only one policy seed is used. The intervals "
              "therefore do not quantify generalization to arbitrary tasks or generation seeds. Degenerate "
              "bootstrap intervals can occur for uniform observed outcomes and do not imply certainty.", "",
              "The starts were already examined during the project, so this is a descriptive comparison, "
              "not a held-out confirmatory study. Different success totals and call counts should be read "
              "jointly: a method that consumes more calls has not demonstrated a gain at equal budget. "
              "A favourable success/call trade-off also does not prove that an internal score is a calibrated "
              "failure probability or isolate why a scheduling method works. This experiment compares "
              "complete scheduling methods; it is not the randomized-scheduling ablation.", "",
              "## Evidence", "",
              "The retained per-episode export and analysis are checked against their artifact hashes. "
              "New raw AutoHorizon traces must pass the dedicated verifier before analysis. "
              "`verification.json` records that audit, `episodes.csv` contains all 390 per-episode metrics, "
              "and `outcome_matrix.csv` preserves the 30 matched starts. `artifacts.json` binds every "
              "generated report, table and plot to the current analysis and source hashes.", ""]
    return "\n".join(lines)


def analyze(root, existing_analysis, output, *, draws=10000, seed=20261024, plots=True):
    from scripts.verify_autohorizon import load_and_verify

    root, existing_analysis, output = Path(root).resolve(), Path(existing_analysis).resolve(), Path(output).resolve()
    for source in (root, existing_analysis.parent):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Analysis output must be separate from raw and retained evidence")
    existing_records, old_result, provenance = load_existing(existing_analysis)
    config, new_records, verification = load_and_verify(root)
    if verification.get("complete") is not True:
        raise ValueError("AutoHorizon verification is incomplete")
    base = config.base
    if (base.chunk_size != 50 or base.num_denoising_steps != 10 or base.control_hz != 20
            or base.precision != "fp32"):
        raise ValueError("AutoHorizon model/control configuration is not comparable")
    result = comparison_statistics(existing_records, new_records, draws=draws, seed=seed)
    behavior, histogram, decisions, timing = horizon_behavior(new_records)
    result.update(
        complete=True, episodes=390, retained_episodes=270, autohorizon_episodes=120,
        retained_fixed_episodes=150, retained_dvac_episodes=120,
        config_fingerprint=config.fingerprint, existing_evidence=provenance,
        horizon_behavior=behavior, autohorizon_timing=timing,
        verification={key: value for key, value in verification.items() if key not in ("episodes", "evidence")},
        timing_comparability=("Historical fixed runs, retained DVAC runs and new AutoHorizon runs have "
                              "different diagnostic instrumentation and potentially different thread/runtime "
                              "conditions. Their wall and inference times are not controlled cross-method "
                              "speed measurements. Compare model calls per episode and per 100 actions."),
        source_sha256={"scripts/analyze_autohorizon.py": sha256(__file__),
                       "scripts/analyze_capped.py": sha256(PROJECT / "scripts/analyze_capped.py")},
    )
    # Reusing the old records must preserve all previously published point estimates.
    current = {row["method"]: row for row in result["methods"]}
    for old in old_result["methods"]:
        for metric in (*METRICS, "calls_per_100_actions"):
            if not math.isclose(old[metric + "_mean"], current[old["method"]][metric + "_mean"], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Retained results changed during integration")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "verification.json", verification)
    atomic_json(output / "analysis.json", result)
    tables = {
        "methods.csv": result["methods"], "by_task.csv": result["by_task"],
        "paired_contrasts.csv": result["contrasts"], "outcome_matrix.csv": result["outcome_matrix"],
        "horizon_behavior.csv": behavior, "interval_histogram.csv": histogram,
        "decisions.csv": decisions, "autohorizon_timing.csv": timing,
        "episodes.csv": [{**{key: record["summary"][key] for key in
                             ("episode_id", "suite", "task", "state", "policy_seed", "method", *METRICS)},
                          "origin": "new_autohorizon" if record["summary"]["method"] in AUTO_METHODS else "retained"}
                         for record in [*existing_records, *new_records]],
    }
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    plot_files = make_plots(output, result) if plots else []
    report = report_text(result)
    if not plots:
        report = "\n".join(line for line in report.splitlines() if not line.startswith("![")) + "\n"
    (output / "report.md").write_text(report)
    files = ["analysis.json", "verification.json", "report.md", *tables, *plot_files]
    atomic_json(output / "artifacts.json", {
        "statistics_filename": "analysis.json", "statistics_sha256": sha256(output / "analysis.json"),
        "source_sha256": result["source_sha256"], "config_fingerprint": config.fingerprint,
        "files": {filename: sha256(output / filename) for filename in files},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "outputs/autohorizon")
    parser.add_argument("--existing-analysis", type=Path, default=PROJECT / "outputs/capped-report/analysis.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/autohorizon-report")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze(args.root, args.existing_analysis, args.output, draws=args.draws, seed=args.seed, plots=not args.no_plots)
    print(json.dumps({"complete": result["complete"], "episodes": result["episodes"],
                      "methods": [{key: row[key] for key in ("method", "episodes", "successes", "model_calls_mean")}
                                  for row in result["methods"]]}, indent=2))


if __name__ == "__main__":
    main()
