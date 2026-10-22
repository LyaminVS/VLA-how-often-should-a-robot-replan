"""Paired, task-stratified cluster bootstrap; seeds stay inside their initial-state cluster."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .config import Experiment
from .storage import atomic_json


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_results(root, allow_incomplete=False):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("data_kind") != "real_libero":
        raise ValueError("Research reports only accept real LIBERO data, never synthetic test fixtures")
    # Reuse full schema validation without a temporary config file.
    data = manifest["config"]
    from .config import Task

    cfg = Experiment(
        **{
            **data,
            "tasks": tuple(Task(**t) for t in data["tasks"]),
            **{k: tuple(data[k]) for k in ("intervals", "initial_states", "policy_seeds")},
        }
    )
    cfg.validate()
    expected = {e.id: e for e in cfg.episodes()}
    if manifest.get("config_fingerprint") != cfg.fingerprint:
        raise ValueError("Manifest configuration fingerprint is invalid")
    rows = []
    seen = set()
    for file in sorted(root.glob("episodes/*/summary.json")):
        row = json.loads(file.read_text())
        identity = row["episode_id"]
        if identity in seen or identity not in expected:
            raise ValueError(f"Duplicate or unexpected episode: {identity}")
        seen.add(identity)
        if row["data_kind"] != "real_libero" or row["config_fingerprint"] != cfg.fingerprint:
            raise ValueError("Mixed configurations or synthetic data detected")
        spec = expected[identity]
        if any(row[k] != v for k, v in spec.asdict().items()) or row["pair_id"] != spec.pair_id:
            raise ValueError(f"Episode identity does not match its specification: {identity}")
        if type(row["success"]) is not bool or not 1 <= row["steps"] <= spec.max_steps:
            raise ValueError("Invalid outcome or step count")
        if row["model_calls"] != (row["steps"] + spec.k - 1) // spec.k:
            raise ValueError("Model call count does not agree with execution interval")
        for field in (
            "compute_s",
            "simulation_s",
            "inference_s",
            "episode_wall_s",
            "sim_time_s",
            "eef_path_m",
        ):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid metric: {field}")
        rows.append(row)
    if not rows:
        raise ValueError("No completed real episodes. Run the smoke experiment first.")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["task"], row["state"])].append(row)
    complete, omitted = [], []
    required = {(k, s) for k in cfg.intervals for s in cfg.policy_seeds}
    for key, group in groups.items():
        if {(r["k"], r["policy_seed"]) for r in group} != required:
            omitted.append(key)
            continue
        # Compare physical starts across all K AND policy sampling replicates.
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            if len({r[field] for r in group}) != 1:
                raise ValueError(f"Initial states differ within {key}: {field}")
        complete.extend(group)
    missing = len(expected) - len(rows)
    if (missing or omitted) and not allow_incomplete:
        raise ValueError(
            f"Incomplete experiment: {len(rows)}/{len(expected)} episodes. "
            "Finish the run or explicitly use --allow-incomplete for a provisional report."
        )
    if not complete:
        raise ValueError(
            "No complete paired state blocks; run every K and policy seed for at least one state"
        )
    return (
        cfg,
        complete,
        {
            "completed_episodes": len(rows),
            "planned_episodes": len(expected),
            "analyzed_episodes": len(complete),
            "omitted_incomplete_state_blocks": len(omitted),
            "provisional": bool(missing or omitted),
        },
    )


def cluster_estimates(rows, intervals, metric, *, draws=5000, seed=20261017, reference=10):
    import numpy as np

    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        groups[(row["suite"], row["task"])][row["state"]][row["k"]].append(float(row[metric]))
    rng = np.random.default_rng(seed)
    means, boot = [], []
    per_task = []
    min_states = min(len(states) for states in groups.values())
    for task, states in sorted(groups.items()):
        matrix = np.array([[np.mean(by_k[k]) for k in intervals] for _, by_k in sorted(states.items())])
        draw_indices = rng.integers(0, len(matrix), size=(draws, len(matrix)))
        sampled = matrix[draw_indices].mean(axis=1)
        means.append(matrix.mean(axis=0))
        boot.append(sampled)
        for i, k in enumerate(intervals):
            lo, hi = np.quantile(sampled[:, i], [0.025, 0.975])
            per_task.append(
                {
                    "suite": task[0],
                    "task": task[1],
                    "metric": metric,
                    "k": k,
                    "mean": float(means[-1][i]),
                    "ci_low": float(lo) if len(matrix) >= 2 else None,
                    "ci_high": float(hi) if len(matrix) >= 2 else None,
                    "initial_states": len(matrix),
                }
            )
    average = np.mean(means, axis=0)  # equal task weights, equal state weights, mean over seeds
    bootstrap = np.mean(boot, axis=0)
    summary = []
    ref = list(intervals).index(reference)
    for i, k in enumerate(intervals):
        low, high = np.quantile(bootstrap[:, i], [0.025, 0.975])
        delta = bootstrap[:, i] - bootstrap[:, ref]
        dlo, dhi = np.quantile(delta, [0.025, 0.975])
        summary.append(
            {
                "metric": metric,
                "k": k,
                "mean": float(average[i]),
                "ci_low": float(low) if min_states >= 2 else None,
                "ci_high": float(high) if min_states >= 2 else None,
                "reference_k": reference,
                "paired_difference": float(average[i] - average[ref]),
                "difference_ci_low": float(dlo) if min_states >= 2 else None,
                "difference_ci_high": float(dhi) if min_states >= 2 else None,
            }
        )
    return summary, per_task


ANNOTATION_FIELDS = [
    "episode_id",
    "reviewed",
    "missed_grasps",
    "dropped_objects",
    "correction_after_replan",
    "failure_category",
    "notes",
]
CATEGORIES = {
    "none",
    "missed_grasp",
    "dropped_object",
    "wrong_object",
    "placement",
    "collision",
    "timeout",
    "other",
    "unclear",
}


def annotation_template(root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Annotation file already exists; refusing to overwrite manual work")
    files = sorted(Path(root).glob("episodes/*/summary.json"))
    if not files:
        raise ValueError("No episodes to annotate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for file in files:
            writer.writerow({"episode_id": json.loads(file.read_text())["episode_id"], "reviewed": "no"})


def summarize_annotations(path, rows):
    by_id = {r["episode_id"]: r for r in rows}
    reviewed = defaultdict(list)
    seen = set()
    with Path(path).open(newline="") as handle:
        for label in csv.DictReader(handle):
            identity = label["episode_id"]
            if identity in seen:
                raise ValueError("Duplicate annotation")
            seen.add(identity)
            if identity not in by_id:
                raise ValueError(f"Annotation is not part of the analyzed cohort: {identity}")
            if label["reviewed"] not in ("yes", "no"):
                raise ValueError("reviewed must be yes or no")
            if label["reviewed"] == "no":
                continue
            for field in ("missed_grasps", "dropped_objects"):
                label[field] = int(label[field])
                if label[field] < 0:
                    raise ValueError("Negative event counts")
            if label["failure_category"] not in CATEGORIES:
                raise ValueError("Unknown failure category")
            if label["correction_after_replan"] not in ("yes", "no", "unclear"):
                raise ValueError("Correction label must be yes/no/unclear")
            reviewed[by_id[identity]["k"]].append(label)
    result = []
    for k in sorted({r["k"] for r in rows}):
        labels = reviewed[k]
        result.append(
            {
                "k": k,
                "reviewed": len(labels),
                "total_episodes": sum(r["k"] == k for r in rows),
                "missed_grasps_per_reviewed_episode": sum(r["missed_grasps"] for r in labels) / len(labels)
                if labels
                else None,
                "drops_per_reviewed_episode": sum(r["dropped_objects"] for r in labels) / len(labels)
                if labels
                else None,
            }
        )
    return result


def make_plots(summary, per_task, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, metric, title in zip(
        axes.flat[:3],
        ["success", "compute_s", "model_calls"],
        ["Task success", "Inference + simulation (seconds)", "Model calls per episode"],
        strict=True,
    ):
        selected = [r for r in summary if r["metric"] == metric]
        x, y = [r["k"] for r in selected], np.array([r["mean"] for r in selected])
        ax.plot(x, y, "o-", color="#2165b8", linewidth=2)
        if all(r["ci_low"] is not None for r in selected):
            ax.fill_between(
                x,
                [r["ci_low"] for r in selected],
                [r["ci_high"] for r in selected],
                color="#2165b8",
                alpha=0.15,
            )
        ax.set(xlabel="Actions executed before replanning (K)", title=title)
        ax.set_xticks(x)
        ax.grid(alpha=0.2)
        if metric == "success":
            ax.set_ylim(0, 1.05)
    ax = axes.flat[3]
    tasks = sorted({r["task"] for r in per_task})
    for task in tasks:
        selected = [r for r in per_task if r["metric"] == "success" and r["task"] == task]
        ax.plot(
            [r["k"] for r in selected], [r["mean"] for r in selected], "o-", label=task.replace("_", " ")[:46]
        )
    ax.set(xlabel="K", ylabel="Success fraction", title="Results by task", ylim=(0, 1.05))
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)
    figure.suptitle(
        "SmolVLA / LIBERO — paired initial states\nShading: 95% state-cluster bootstrap intervals",
        fontsize=13,
    )
    figure.savefig(output / "comparison.png", dpi=180)
    figure.savefig(output / "comparison.pdf")
    plt.close(figure)


def analyze(root, output, *, allow_incomplete=False, draws=5000, reference=10, annotations=None):
    cfg, rows, coverage = load_results(root, allow_incomplete)
    if reference not in cfg.intervals:
        raise ValueError("Reference K is absent from the experiment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, per_task = [], []
    for metric in (
        "success",
        "compute_s",
        "model_calls",
        "sim_time_s",
        "eef_path_m",
        "gripper_close_commands",
        "episode_wall_s",
    ):
        summary, tasks = cluster_estimates(
            rows, sorted(cfg.intervals), metric, draws=draws, reference=reference
        )
        summaries.extend(summary)
        per_task.extend(tasks)
    warnings = [
        "Intervals describe variation across sampled initial states, conditional on chosen tasks and seeds.",
        "Comparisons are exploratory; intervals are not adjusted for multiple comparisons.",
        "Gripper-close commands and action changes are proxies, not grasp errors or proof of correction.",
    ]
    if min(r["initial_states"] for r in per_task) < 10:
        warnings.append(
            "Few initial states: treat results as a pilot; small or degenerate intervals are not proof of equivalence."
        )
    if len({(r["suite"], r["task"]) for r in rows}) != len(cfg.tasks):
        warnings.append(
            "Some planned tasks have no complete blocks; macro averages cover observed tasks only."
        )
    if coverage["provisional"]:
        warnings.append(
            "INCOMPLETE RUN: only fully paired state blocks are analyzed; finish the run before final conclusions."
        )
    labels = summarize_annotations(annotations, rows) if annotations else None
    atomic_json(
        output / "statistics.json",
        {
            "coverage": coverage,
            "warnings": warnings,
            "estimates": summaries,
            "per_task": per_task,
            "annotations": labels,
        },
    )
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "estimates.csv", summaries)
    write_csv(output / "by_task.csv", per_task)
    if labels is not None:
        write_csv(output / "grasp_annotations.csv", labels)
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]][row["k"]] = row
    candidates = []
    low, high = min(cfg.intervals), max(cfg.intervals)
    for pair, by_k in sorted(pairs.items()):
        a, b = by_k[low], by_k[high]
        candidates.append(
            {
                "pair_id": pair,
                "low_k_episode": a["episode_id"],
                "high_k_episode": b["episode_id"],
                "low_k_success": a["success"],
                "high_k_success": b["success"],
                "outcomes_differ": a["success"] != b["success"],
            }
        )
    write_csv(output / "demo_candidates.csv", candidates)
    make_plots(summaries, per_task, output)
    return coverage


def trace_episode(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    times = [r["sim_time_s"] for r in rows]
    for index, label in enumerate(("x", "y", "z")):
        axes[0].plot(times, [r["eef"][index] for r in rows], label=label)
    axes[0].set(ylabel="End-effector position (m)")
    axes[0].legend()
    axes[1].plot(times, [r["action"][6] for r in rows], label="gripper command")
    axes[1].set(ylabel="Open (-1) / close (+1)")
    axes[2].plot(times, [r["action_translation_jump_l2"] for r in rows], label="successive action change")
    axes[2].set(ylabel="Translation command L2", xlabel="Simulation time (s)")
    for ax in axes:
        for row in rows:
            if row["replanned"]:
                ax.axvline(row["sim_time_s"], color="#db9340", alpha=0.18, linewidth=0.7)
        ax.grid(alpha=0.15)
    fig.suptitle(
        f"K={summary['k']} | success={summary['success']} | orange lines = replans\n"
        + summary["instruction"],
        fontsize=10,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def estimate_budget(config, root):
    """Empirical full-horizon budget; not a confidence interval or a runtime promise."""
    source_config, rows, coverage = load_results(root, allow_incomplete=True)
    for field in (
        "model_id",
        "model_revision",
        "backbone_revision",
        "precision",
        "num_denoising_steps",
        "chunk_size",
        "control_hz",
        "record_video",
    ):
        if getattr(source_config, field) != getattr(config, field):
            raise ValueError(f"Cannot extrapolate across differing {field}")
    calls = sum(r["model_calls"] for r in rows)
    steps = sum(r["steps"] for r in rows)
    seconds_per_call = sum(r["inference_s"] for r in rows) / calls
    seconds_per_step = sum(r["simulation_s"] for r in rows) / steps
    overhead_per_episode = sum(max(0, r["episode_wall_s"] - r["compute_s"]) for r in rows) / len(rows)
    budget = config.budget()
    hours = (
        seconds_per_call * budget["model_calls_upper_bound"]
        + seconds_per_step * budget["control_steps_upper_bound"]
        + overhead_per_episode * budget["episodes"]
    ) / 3600
    return {
        **budget,
        "measured_seconds_per_call": seconds_per_call,
        "measured_seconds_per_control_step": seconds_per_step,
        "projected_full_horizon_hours": hours,
        "calibration": coverage,
        "note": "Assumes the same hardware and timings, all episodes reach their step limit; "
        "video/reset overhead is extrapolated per episode. Not a statistical upper bound.",
    }
