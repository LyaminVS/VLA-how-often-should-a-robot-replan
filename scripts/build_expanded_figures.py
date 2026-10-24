"""Build figures for the complete 150-episode study.

No simulator is run. Stored estimates are recomputed to check consistency without
changing the analysis. Both source cohorts and the stored provenance must validate
before any output is created.

Usage: python scripts/build_expanded_figures.py --run outputs/expanded
"""

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_extension import (  # noqa: E402
    METRICS,
    VARIABLE_CONFIG_FIELDS,
    annotation_statistics,
    load_annotations,
    load_phase,
    phase_statistics,
    sha256,
    validate_compatibility,
)
from scripts.build_report import validate_discussion  # noqa: E402

KS = (1, 5, 10, 25, 50)
TASKS = {
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"): "Open the middle drawer",
    ("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"): "Soup can into basket",
    ("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"):
        "Black bowl onto plate",
}
PHASES = ("pilot", "validation", "combined")
STUDY_COLOR = "#087F8C"
FIGURES = {
    "study-by-task": (11.6, 3.7),
    "study-cost": (11.6, 3.9),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_snapshot(path):
    """Bind the parsed document to its input bytes before rendering begins."""
    path = Path(path).resolve()
    digest = sha256(path)
    data = read_json(path)
    verify_unchanged({path: digest})
    return data, digest


def verify_unchanged(snapshots):
    for path, digest in snapshots.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during authoring; rerun with stable inputs: {path}")


def mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.mean(values) if values else None


def _equal_number(actual, expected, label):
    if not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise ValueError(f"Missing or non-finite estimate: {label}")
    if not math.isclose(actual, expected, abs_tol=1e-9):
        raise ValueError(f"Stale analysis: {label}")


def _validate_estimates(phase, rows, expected_states, reference):
    estimates = phase["estimates"]
    if len(estimates) != len(METRICS) * len(KS):
        raise ValueError("Unexpected number of phase estimates")
    indexed = {(r["metric"], r["k"]): r for r in estimates}
    if set(indexed) != {(metric, k) for metric in METRICS for k in KS}:
        raise ValueError("Duplicate or missing estimates")
    for (metric, k), estimate in indexed.items():
        value = mean([r for r in rows if r["k"] == k], metric)
        ref_value = mean([r for r in rows if r["k"] == reference], metric)
        _equal_number(estimate["mean"], value, f"{metric}, K={k}")
        _equal_number(estimate["paired_difference"], value - ref_value, f"paired {metric}, K={k}")
        if estimate["reference_k"] != reference:
            raise ValueError("Reference K differs between estimates and provenance")
        for low, high in (("ci_low", "ci_high"), ("difference_ci_low", "difference_ci_high")):
            bounds = [estimate[low], estimate[high]]
            if any(v is None or not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid uncertainty interval for {metric}, K={k}")
    tasks = phase["per_task"]
    expected = {(suite, task, metric, k) for suite, task in TASKS for metric in METRICS for k in KS}
    keys = [(r["suite"], r["task"], r["metric"], r["k"]) for r in tasks]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing or duplicated task estimates")
    for item in tasks:
        selected = [r for r in rows if (r["suite"], r["task"], r["k"]) ==
                    (item["suite"], item["task"], item["k"])]
        _equal_number(item["mean"], mean(selected, item["metric"]), "per-task mean")
        if item["initial_states"] != expected_states:
            raise ValueError("Wrong initial-state denominator in task estimates")
        if any(item[key] is None or not math.isfinite(item[key]) for key in ("ci_low", "ci_high")):
            raise ValueError("Invalid per-task uncertainty interval")
        if item["ci_low"] > item["ci_high"]:
            raise ValueError("Reversed per-task uncertainty interval")


def load_bundle(run, pilot, validation):
    """Return verified statistics, source runs, rows and labels; never write files."""
    run = Path(run).resolve()
    stats_path = run / "report/statistics.json"
    stats, statistics_digest = read_json_snapshot(stats_path)
    if (stats.get("schema_version") != 1 or stats.get("primary_phase") != "combined"
            or stats.get("combined_analysis_role") != "main_descriptive"
            or stats.get("analysis_kind") != "disjoint_paired_extension"
            or set(stats.get("phases", {})) != set(PHASES)):
        raise ValueError("Expected the complete combined descriptive analysis with original source provenance")
    sources = {"pilot": load_phase(pilot, "pilot"), "validation": load_phase(validation, "validation")}
    validate_compatibility(sources["pilot"], sources["validation"])
    states = {"pilot": (0, 1), "validation": tuple(range(2, 10)), "combined": tuple(range(10))}
    for phase, source in sources.items():
        cfg = source["config"]
        if (tuple(sorted(cfg.initial_states)) != states[phase] or tuple(sorted(cfg.intervals)) != KS
                or tuple(cfg.policy_seeds) != (11,) or cfg.chunk_size != 50 or cfg.control_hz != 20
                or {(t.suite, t.name) for t in cfg.tasks} != set(TASKS)):
            raise ValueError(f"{phase}: expected the fixed 3-task, one-seed extension protocol")
    selection = stats["annotation_selection"]
    if sorted(selection["pilot_states"]) != [0, 1] or sorted(selection["validation_states"]) != [2, 3]:
        raise ValueError("Visual-review subset must be pilot 0/1 and validation 2/3")
    provenance = stats["provenance"]
    if (provenance["unique_episode_count"] != 150 or provenance["reference_k"] != 10
            or provenance["bootstrap_draws"] != 5000 or provenance["bootstrap_seed"] != 20261017):
        raise ValueError("Wrong cohort size, reference or bootstrap configuration")
    if provenance["analysis_script_sha256"] != sha256(Path(__file__).with_name("analyze_extension.py")):
        raise ValueError("Analysis script changed; regenerate the expanded statistics")
    if provenance["initial_states"] != {key: list(states[key]) for key in sources}:
        raise ValueError("State provenance disagrees with the sources")
    first = sources["pilot"]
    shared = {k: v for k, v in first["config"].asdict().items() if k not in VARIABLE_CONFIG_FIELDS}
    shared = json.loads(json.dumps(shared))
    if (provenance["shared_config"] != shared or provenance["runtime"] != first["manifest"]["runtime"]
            or provenance["hardware"] != first["manifest"]["hardware"]):
        raise ValueError("Shared configuration/runtime provenance is stale")
    stored_sources = {item["phase"]: item for item in provenance["sources"]}
    if len(provenance["sources"]) != 2 or set(stored_sources) != set(sources):
        raise ValueError("Expected exactly two source provenance records")
    labels = {}
    for phase, source in sources.items():
        stored = stored_sources[phase]
        current = source["provenance"]
        for key in ("manifest_sha256", "config_fingerprint", "created_utc"):
            if stored[key] != current[key]:
                raise ValueError(f"{phase}: source manifest changed")
        def digests(items):
            return {item["episode_id"]: {k: v["sha256"] for k, v in item["files"].items()} for item in items}
        if len(stored["episodes"]) != len(current["episodes"]) or digests(stored["episodes"]) != digests(current["episodes"]):
            raise ValueError(f"{phase}: raw episode files changed since analysis")
        annotation = source["root"] / "annotations.csv"
        recorded = stored.get("annotations")
        if bool(recorded) != annotation.is_file() or (recorded and recorded["sha256"] != sha256(annotation)):
            raise ValueError(f"{phase}: annotations changed; rerun analysis")
        labels[phase] = load_annotations(annotation if annotation.is_file() else None, source,
                                         selection[f"{phase}_states"])
        old_inspection, new_inspection = stored.get("inspection"), source["provenance"].get("inspection")
        if bool(old_inspection) != bool(new_inspection) or (old_inspection and any(
                old_inspection[key] != new_inspection[key] for key in
                ("sha256", "method", "selected_episode_ids", "inspected_episode_ids"))):
            raise ValueError(f"{phase}: inspection evidence changed; rerun analysis")
    rows = {phase: source["rows"] for phase, source in sources.items()}
    rows["combined"] = rows["pilot"] + rows["validation"]
    labels["combined"] = labels["pilot"] + labels["validation"]
    for phase in PHASES:
        data = stats["phases"][phase]
        count = {"pilot": 30, "validation": 120, "combined": 150}[phase]
        expected_coverage = {"completed_episodes": count, "planned_episodes": count,
                             "analyzed_episodes": count, "omitted_incomplete_state_blocks": 0,
                             "provisional": False}
        if data["coverage"] != expected_coverage or len(rows[phase]) != count:
            raise ValueError(f"{phase}: report requires the complete cohort")
        _validate_estimates(data, rows[phase], len(states[phase]), provenance["reference_k"])
        recalculated = phase_statistics(rows[phase], KS, labels[phase], phase=phase,
                                        draws=provenance["bootstrap_draws"], reference=provenance["reference_k"])
        for key in ("estimates", "per_task", "annotations", "annotation_coverage"):
            if data[key] != recalculated[key]:
                raise ValueError(f"{phase}: stored {key} differs from recomputed paired analysis")
        standard, coverage = annotation_statistics(labels[phase], KS)
        if data["annotations"] != standard or data["annotation_coverage"] != coverage:
            raise ValueError(f"{phase}: stale annotation denominators")
    for key in ("coverage", "estimates", "per_task", "annotations", "annotation_coverage"):
        if stats[key] != stats["phases"]["combined"][key]:
            raise ValueError("Top-level combined statistics disagree with the combined phase")
    verify_unchanged({stats_path: statistics_digest})
    return {"run": run, "statistics_path": stats_path, "statistics_sha256": statistics_digest, "stats": stats,
            "sources": sources, "rows": rows, "labels": labels}


def validate_expanded_discussion(data, rows):
    """Existing discussion schema; cases must use matched inspected states 0-3.

    Optional case metadata: id/kind (strings), caveats/protocol_refs (string lists).
    Unknown extra metadata is preserved by callers but never rendered as instructions.
    """
    validate_discussion(data, rows)
    by_id = {row["episode_id"]: row for row in rows}
    for case in data["cases"]:
        if not 2 <= len(case["episode_ids"]) <= 5:
            raise ValueError("Each case must compare at least two K values")
        if any(by_id[identity]["state"] not in (0, 1, 2, 3) for identity in case["episode_ids"]):
            raise ValueError("Cases must come from the inspected review states 0-3")
        for field in ("id", "kind"):
            if field in case and (not isinstance(case[field], str) or len(case[field]) > 120):
                raise ValueError(f"Optional case {field} must be a short string")
        for field in ("caveats", "protocol_refs"):
            if field in case and (not isinstance(case[field], list) or len(case[field]) > 4 or any(
                    not isinstance(value, str) or not value.strip() or len(value) > 300 for value in case[field])):
                raise ValueError(f"Optional case {field} must contain up to four short strings")


def publish(staged, target, overwrite=False):
    """Atomic replace or atomic no-clobber publication on the same filesystem."""
    if overwrite:
        os.replace(staged, target)
    else:
        os.link(staged, target)
        Path(staged).unlink()


def build_figures(args):
    generator = Path(__file__).resolve()
    generator_digest = sha256(generator)
    bundle = load_bundle(args.run, args.pilot, args.validation)
    snapshots = {generator: generator_digest, bundle["statistics_path"]: bundle["statistics_sha256"]}
    output = bundle["run"] / "report"
    targets = [output / f"{name}.{extension}" for name in FIGURES for extension in ("png", "pdf")]
    targets.append(output / "expanded-figures.json")
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Expanded figures already exist; pass --overwrite to replace them")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    stats = bundle["stats"]

    def interval(ax, estimates, metric, color, label=None, offset=0):
        entries = {(r["metric"], r["k"]): r for r in estimates}
        y = [entries[(metric, k)]["mean"] for k in KS]
        x = [index + offset for index in range(len(KS))]
        # Intervals are drawn explicitly, so asymmetric percentile intervals need
        # not be artificially expanded to contain the point estimate.
        ax.plot(x, y, "o-", color=color, label=label, linewidth=1.6, markersize=4)
        for position, k in zip(x, KS, strict=True):
            low, high = (entries[(metric, k)][key] for key in ("ci_low", "ci_high"))
            ax.vlines(position, low, high, color=color, linewidth=1)
            ax.hlines([low, high], position - 0.04, position + 0.04, color=color, linewidth=1)

    def axis(ax, ylabel, success=False):
        ax.set_xticks(range(len(KS)), KS)
        ax.set_xlim(-0.3, 4.3)
        ax.set_xlabel("K - actions before the next observation")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DFE7EC", linewidth=0.7)
        ax.set_axisbelow(True)
        if success:
            ax.set_ylim(-0.04, 1.08)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            ax.set_ylim(bottom=0)

    def setup(name, panels):
        fig, axes = plt.subplots(1, panels, figsize=FIGURES[name])
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.25, top=0.79, wspace=0.36)
        return fig, axes

    figures = {}
    fig, axes = setup("study-by-task", 3)
    for ax, ((suite, task), label), color in zip(axes, TASKS.items(), ("#27658B", "#D17C32", "#428657"), strict=True):
        subset = [r for r in stats["phases"]["combined"]["per_task"] if (r["suite"], r["task"]) == (suite, task)]
        interval(ax, subset, "success", color)
        axis(ax, "Success rate", success=True)
        ax.set_title(label, loc="left", pad=9)
    fig.suptitle("Task completion across all 150 episodes", fontsize=16, y=0.98)
    fig.text(0.5, 0.875, "3 tasks | states 0-9 | 10 episodes per task and K", ha="center")
    fig.text(0.07, 0.04, "Points: means. Bars: 95% state-bootstrap intervals within each task. Lines do not imply interpolation.", fontsize=9.5)
    figures["study-by-task"] = fig

    fig, axes = setup("study-cost", 3)
    study_stats = stats["phases"]["combined"]
    interval(axes[0], study_stats["estimates"], "model_calls", STUDY_COLOR)
    axis(axes[0], "Mean calls per episode")
    axes[0].set_title("Model calls", loc="left")
    interval(axes[1], study_stats["estimates"], "episode_wall_s", STUDY_COLOR)
    axis(axes[1], "Mean episode wall time, s")
    axes[1].set_title("Wall time", loc="left")
    weighted_latency = []
    for k in KS:
        selected = [r for r in bundle["rows"]["combined"] if r["k"] == k]
        weighted_latency.append(sum(r["inference_s"] for r in selected) / sum(r["model_calls"] for r in selected))
    axes[2].plot(range(len(KS)), weighted_latency, "o-", color="#27658B", linewidth=1.6)
    axis(axes[2], "Seconds per model call")
    axes[2].set_title("Call-weighted model latency", loc="left")
    fig.suptitle("Calls, wall time and inference latency across all 150 episodes", fontsize=16, y=0.98)
    fig.text(0.5, 0.875, "30 episodes per K | one GPU | synchronous simulation", ha="center")
    fig.text(0.07, 0.08, "First two panels: means with 95% paired-state bootstrap intervals. Latency = total inference seconds / calls.", fontsize=9.5)
    fig.text(0.07, 0.035, "Latency includes preprocessing and postprocessing. Simulation pauses for inference; this is not a real-time test.", fontsize=9.5)
    figures["study-cost"] = fig

    with tempfile.TemporaryDirectory(prefix=".expanded-figures-", dir=output) as directory:
        staged = Path(directory)
        entries = {}
        try:
            for name, fig in figures.items():
                for extension in ("png", "pdf"):
                    target = staged / f"{name}.{extension}"
                    fig.savefig(target, dpi=300)
                    entries[target.name] = sha256(target)
            record = {"statistics_sha256": bundle["statistics_sha256"],
                      "generator_sha256": generator_digest, "files": entries,
                      "primary_phase": "combined", "combined_analysis_role": "main_descriptive",
                      "episodes": {"pilot": 30, "validation": 120, "combined": 150}}
            (staged / "expanded-figures.json").write_text(json.dumps(record, indent=2) + "\n")
            verify_unchanged(snapshots)
            for target in targets:
                publish(staged / target.name, target, args.overwrite)
        finally:
            for fig in figures.values():
                plt.close(fig)
    print(json.dumps({"output": str(output), "figures": list(entries), "visual_review_required": True}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--pilot", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_figures(args)


if __name__ == "__main__":
    main()
