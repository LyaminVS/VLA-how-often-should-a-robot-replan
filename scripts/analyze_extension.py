"""Analyze all episodes together while preserving both original source runs.

Usage: python scripts/analyze_extension.py --pilot outputs/compact \
    --validation outputs/validation --output outputs/expanded/report
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_replan.analysis import (
    ANNOTATION_FIELDS,
    CATEGORIES,
    cluster_estimates,
    load_results,
    make_plots,
    write_csv,
)
from vla_replan.storage import atomic_json

METRICS = (
    "success", "compute_s", "model_calls", "sim_time_s", "eef_path_m",
    "gripper_close_commands", "episode_wall_s",
)
VARIABLE_CONFIG_FIELDS = {"name", "initial_states", "order_seed"}
RUNTIME_FIELDS = {
    "python", "system", "machine", "packages", "code_sha256", "dependency_lock_sha256",
    "simulator_assets", "render_backend", "cpu_count",
}
HARDWARE_FIELDS = {
    "device", "vram_bytes", "cuda", "precision", "deterministic_algorithms", "torch_cpu_threads",
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def local_source_hashes():
    """Use the runner's exact hash convention, without comparing local CPU packages."""
    project = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((project / "vla_replan").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "code_sha256": digest.hexdigest(),
        "dependency_lock_sha256": sha256(project / "requirements/linux.lock.txt"),
    }


def load_phase(root, phase):
    root = Path(root).resolve()
    cfg, rows, coverage = load_results(root)  # Reject every incomplete or unpaired source run.
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    planned = manifest.get("planned_episode_ids")
    if not isinstance(planned, list) or len(planned) != len(set(planned)) or set(planned) != {
        row["episode_id"] for row in rows
    }:
        raise ValueError(f"{phase}: manifest planned episode IDs disagree with the completed run")
    for section, required in (("runtime", RUNTIME_FIELDS), ("hardware", HARDWARE_FIELDS)):
        metadata = manifest.get(section)
        if not isinstance(metadata, dict) or not required <= metadata.keys():
            raise ValueError(f"{phase}: missing required {section} metadata")
        if any(metadata[key] is None for key in required):
            raise ValueError(f"{phase}: unknown {section} metadata cannot establish comparability")
    runtime = manifest["runtime"]
    if not runtime["packages"] or any(value is None for value in runtime["packages"].values()):
        raise ValueError(f"{phase}: incomplete runtime package versions")
    for field, current in local_source_hashes().items():
        if runtime[field] != current:
            raise ValueError(f"{phase}: {field} differs from the current frozen analysis source/lock")
    sources = []
    labeled_rows = []
    for row in rows:
        directory = root / "episodes" / row["episode_id"]
        required_files = ["summary.json", "steps.jsonl", "chunks.jsonl"]
        if cfg.record_video:
            required_files.append("video.mp4")
        files = {}
        for name in required_files:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{phase}: missing or empty episode artifact: {path}")
            files[name] = {"path": str(path), "sha256": sha256(path)}
        sources.append({"episode_id": row["episode_id"], "phase": phase, "files": files})
        labeled_rows.append({
            **row, "phase": phase, "source_run": str(root),
            "source_summary": str(directory / "summary.json"),
        })
    return {
        "phase": phase, "root": root, "config": cfg, "rows": labeled_rows, "coverage": coverage,
        "manifest": manifest,
        "provenance": {
            "phase": phase, "root": str(root), "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "config_fingerprint": manifest["config_fingerprint"],
            "created_utc": manifest.get("created_utc"), "episodes": sources,
        },
    }


def validate_compatibility(pilot, validation):
    a = pilot["config"].asdict()
    b = validation["config"].asdict()
    for field in sorted((a.keys() | b.keys()) - VARIABLE_CONFIG_FIELDS):
        if a.get(field) != b.get(field):
            raise ValueError(f"Run settings differ: {field}")
    overlap = set(a["initial_states"]) & set(b["initial_states"])
    if overlap:
        raise ValueError(f"Pilot and validation initial states overlap: {sorted(overlap)}")
    identities = [row["episode_id"] for source in (pilot, validation) for row in source["rows"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate episode IDs across phases")
    for section in ("runtime", "hardware"):
        if pilot["manifest"][section] != validation["manifest"][section]:
            changed = sorted(
                key for key in pilot["manifest"][section].keys() | validation["manifest"][section].keys()
                if pilot["manifest"][section].get(key) != validation["manifest"][section].get(key)
            )
            raise ValueError(f"Run {section} differs: {', '.join(changed)}")


def load_inspection(phase, selected_states):
    """A review-selection plan does not establish that a recording was inspected."""
    path = phase["root"] / "inspection.json"
    if not path.exists():
        phase["provenance"]["inspection"] = None
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("method"), str) or not data["method"].strip():
        raise ValueError("Inspection evidence requires a nonempty method")
    for field in ("selected_episode_ids", "inspected_episode_ids"):
        identities = data.get(field)
        if not isinstance(identities, list) or any(not isinstance(identity, str) for identity in identities):
            raise ValueError(f"Inspection {field} must be a list of episode IDs")
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate inspection IDs: {field}")
    selected = {row["episode_id"] for row in phase["rows"] if row["state"] in selected_states}
    if set(data["selected_episode_ids"]) != selected:
        raise ValueError("Inspection selection does not equal the configured visual-review subset")
    inspected = set(data["inspected_episode_ids"])
    if not inspected <= selected:
        raise ValueError("Inspected episode IDs are outside the selected review subset")
    phase["provenance"]["inspection"] = {
        "path": str(path), "sha256": sha256(path), "method": data["method"],
        "selected_episode_ids": data["selected_episode_ids"],
        "inspected_episode_ids": data["inspected_episode_ids"],
    }
    return inspected


def load_annotations(path, phase, selected_states):
    """Missing or ambiguous labels remain unknown, including individual event counts."""
    by_id = {row["episode_id"]: row for row in phase["rows"]}
    inspected = load_inspection(phase, selected_states)
    supplied = {}
    if path is not None:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(ANNOTATION_FIELDS) <= set(reader.fieldnames or []):
                raise ValueError("Annotation CSV is missing required columns")
            for label in reader:
                label = {field: label[field] for field in ANNOTATION_FIELDS}
                identity = label["episode_id"]
                if identity in supplied or identity not in by_id:
                    raise ValueError(f"Duplicate or out-of-cohort annotation: {identity}")
                if label["reviewed"] not in ("yes", "no"):
                    raise ValueError("reviewed must be yes or no")
                for field in ("missed_grasps", "dropped_objects"):
                    value = label[field].strip()
                    if value and not re.fullmatch(r"[0-9]+", value):
                        raise ValueError(f"Invalid nonnegative integer annotation: {field}")
                    label[field] = int(value) if value else None
                if label["reviewed"] == "yes":
                    if any(label[field] is None for field in ("missed_grasps", "dropped_objects")):
                        raise ValueError("Fully reviewed annotations require both event counts")
                    if label["failure_category"] not in CATEGORIES:
                        raise ValueError("Unknown failure category")
                    if label["correction_after_replan"] not in ("yes", "no", "unclear"):
                        raise ValueError("Correction label must be yes/no/unclear")
                if by_id[identity]["state"] not in selected_states and (
                    label["reviewed"] == "yes"
                    or any(label[field] is not None for field in ("missed_grasps", "dropped_objects"))
                ):
                    raise ValueError("Event annotations lie outside the prespecified visual-review subset")
                if inspected is not None and identity not in inspected and (
                    label["reviewed"] == "yes"
                    or any(label[field] is not None for field in ("missed_grasps", "dropped_objects"))
                ):
                    raise ValueError("Resolved event labels require an inspected episode in inspection.json")
                supplied[identity] = label
    labels = []
    for identity, row in by_id.items():
        label = supplied.get(identity, {
            **dict.fromkeys(ANNOTATION_FIELDS, ""), "episode_id": identity, "reviewed": "no",
            "missed_grasps": None, "dropped_objects": None,
            "notes": "No event annotation available; unknown counts are not zeros.",
        })
        labels.append({
            **label, "phase": phase["phase"], "k": row["k"],
            "selected_for_visual_review": row["state"] in selected_states,
            "inspected": identity in inspected if inspected is not None else None,
            "annotation_provided": identity in supplied,
        })
    return labels


def annotation_statistics(labels, intervals):
    standard, coverage = [], []
    for k in sorted(intervals):
        group = [label for label in labels if label["k"] == k]
        reviewed = [label for label in group if label["reviewed"] == "yes"]
        standard.append({
            "k": k, "reviewed": len(reviewed), "total_episodes": len(group),
            "missed_grasps_per_reviewed_episode": (
                sum(row["missed_grasps"] for row in reviewed) / len(reviewed) if reviewed else None
            ),
            "drops_per_reviewed_episode": (
                sum(row["dropped_objects"] for row in reviewed) / len(reviewed) if reviewed else None
            ),
        })
        item = {
            "k": k, "total_episodes": len(group),
            "selected_for_visual_review": sum(row["selected_for_visual_review"] for row in group),
            "inspected_episodes": (
                sum(row["inspected"] for row in group)
                if all(row["inspected"] is not None for row in group) else None
            ),
            "annotation_rows_provided": sum(row["annotation_provided"] for row in group),
            "fully_reviewed_episodes": len(reviewed),
        }
        for field in ("missed_grasps", "dropped_objects"):
            known = [row[field] for row in group if row[field] is not None]
            item[f"{field}_known_episodes"] = len(known)
            item[f"{field}_total_known"] = sum(known) if known else None
        coverage.append(item)
    return standard, coverage


def phase_statistics(rows, intervals, labels, *, phase, draws, reference):
    summaries, per_task = [], []
    for metric in METRICS:
        summary, tasks = cluster_estimates(rows, sorted(intervals), metric, draws=draws, reference=reference)
        summaries.extend(summary)
        per_task.extend(tasks)
    annotations, annotation_coverage = annotation_statistics(labels, intervals)
    warnings = [
        "Intervals describe variation across sampled initial states, conditional on chosen tasks and seeds.",
        "Comparisons are exploratory; intervals are not adjusted for multiple comparisons.",
        "Gripper-close commands and action changes are proxies, not grasp errors or proof of correction.",
        "Visual event counts describe only the annotated subset; missing or ambiguous counts are not zeros.",
    ]
    if min(row["initial_states"] for row in per_task) < 10:
        warnings.append("Few initial states: small or degenerate intervals do not establish equivalence.")
    return {
        "coverage": {
            "completed_episodes": len(rows), "planned_episodes": len(rows),
            "analyzed_episodes": len(rows), "omitted_incomplete_state_blocks": 0, "provisional": False,
        },
        "warnings": warnings, "estimates": summaries, "per_task": per_task,
        "annotations": annotations, "annotation_coverage": annotation_coverage,
    }


def analyze_extension(pilot_root, validation_root, output, *, pilot_annotations=None,
                      validation_annotations=None, validation_review_states=(2, 3), draws=5000,
                      reference=10, plots=True):
    pilot = load_phase(pilot_root, "pilot")
    validation = load_phase(validation_root, "validation")
    validate_compatibility(pilot, validation)
    intervals = pilot["config"].intervals
    if reference not in intervals:
        raise ValueError("Reference K is absent from the experiment")
    if not set(validation_review_states) <= set(validation["config"].initial_states):
        raise ValueError("Visual-review subset contains states absent from validation")
    output = Path(output).resolve()
    if any(output.is_relative_to(source["root"]) for source in (pilot, validation)):
        raise ValueError("The aggregate report must be outside both source run directories")
    labels = {}
    for source, supplied_path, selected in (
        (pilot, pilot_annotations, pilot["config"].initial_states),
        (validation, validation_annotations, validation_review_states),
    ):
        default_path = source["root"] / "annotations.csv"
        path = supplied_path if supplied_path is not None else default_path if default_path.exists() else None
        labels[source["phase"]] = load_annotations(path, source, selected)
        source["provenance"]["annotations"] = (
            {"path": str(Path(path).resolve()), "sha256": sha256(path)} if path is not None else None
        )
    labels["combined"] = labels["pilot"] + labels["validation"]
    rows = {"pilot": pilot["rows"], "validation": validation["rows"]}
    rows["combined"] = rows["pilot"] + rows["validation"]
    phases = {
        phase: phase_statistics(group, intervals, labels[phase], phase=phase, draws=draws, reference=reference)
        for phase, group in rows.items()
    }
    result = {
        **phases["combined"], "schema_version": 1, "analysis_kind": "disjoint_paired_extension",
        "primary_phase": "combined", "combined_analysis_role": "main_descriptive", "phases": phases,
        "annotation_selection": {
            "pilot_states": list(pilot["config"].initial_states),
            "validation_states": list(validation_review_states),
            "rule": "All tasks, all K values and all policy seeds for the listed states.",
        },
        "provenance": {
            "sources": [pilot["provenance"], validation["provenance"]],
            "shared_config": {
                key: value for key, value in pilot["config"].asdict().items()
                if key not in VARIABLE_CONFIG_FIELDS
            },
            "runtime": pilot["manifest"]["runtime"], "hardware": pilot["manifest"]["hardware"],
            "analysis_script_sha256": sha256(__file__), "bootstrap_draws": draws,
            "bootstrap_seed": 20261017, "reference_k": reference,
            "unique_episode_count": len(rows["combined"]),
            "initial_states": {
                "pilot": list(pilot["config"].initial_states),
                "validation": list(validation["config"].initial_states),
            },
            "note": "Raw manifests, configuration fingerprints and episode files are preserved. "
                    "This is a derived report, not a new simulator run or a replacement manifest.",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for phase, statistics in phases.items():
        directory = output / "phases" / phase
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "statistics.json", statistics)
        write_csv(directory / "episodes.csv", rows[phase])
        write_csv(directory / "estimates.csv", statistics["estimates"])
        write_csv(directory / "by_task.csv", statistics["per_task"])
        write_csv(directory / "grasp_annotations.csv", statistics["annotations"])
        write_csv(directory / "annotation_coverage.csv", statistics["annotation_coverage"])
    write_csv(output / "episodes.csv", rows["combined"])
    write_csv(output / "estimates.csv", result["estimates"])
    write_csv(output / "by_task.csv", result["per_task"])
    write_csv(output / "grasp_annotations.csv", result["annotations"])
    write_csv(output / "annotation_coverage.csv", result["annotation_coverage"])
    write_csv(output / "annotations.csv", labels["combined"])
    write_csv(output / "phase-estimates.csv", [
        {"phase": phase, **row} for phase, statistics in phases.items() for row in statistics["estimates"]
    ])
    write_csv(output / "by_phase.csv", [
        {"phase": phase, **row} for phase, statistics in phases.items() for row in statistics["per_task"]
    ])
    pairs = {}
    for row in rows["combined"]:
        pairs.setdefault(row["pair_id"], {})[row["k"]] = row
    candidates = []
    for pair, by_k in sorted(pairs.items()):
        low, high = by_k[min(intervals)], by_k[max(intervals)]
        candidates.append({
            "pair_id": pair, "phase": low["phase"], "source_run": low["source_run"],
            "low_k_episode": low["episode_id"], "high_k_episode": high["episode_id"],
            "low_k_success": low["success"], "high_k_success": high["success"],
            "outcomes_differ": low["success"] != high["success"],
        })
    write_csv(output / "demo_candidates.csv", candidates)
    if plots:
        make_plots(result["estimates"], result["per_task"], output)
    atomic_json(output / "statistics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expanded/report"))
    parser.add_argument("--pilot-annotations", type=Path)
    parser.add_argument("--validation-annotations", type=Path)
    parser.add_argument("--validation-review-states", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--reference", type=int, default=10)
    args = parser.parse_args()
    result = analyze_extension(
        args.pilot, args.validation, args.output, pilot_annotations=args.pilot_annotations,
        validation_annotations=args.validation_annotations,
        validation_review_states=tuple(args.validation_review_states), draws=args.draws,
        reference=args.reference,
    )
    print(json.dumps({"coverage": result["coverage"], "primary_phase": result["primary_phase"]}))


if __name__ == "__main__":
    main()
