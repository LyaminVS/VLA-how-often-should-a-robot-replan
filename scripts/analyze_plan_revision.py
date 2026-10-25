"""Prospective FM variance versus later command revision, using immutable logs only."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.verify_dvac import equal_number, read_jsonl, recompute_variance, sha256, vector
from vla_replan.storage import atomic_json


def variance_translation(clean_estimates, index, *, tail_length=5, horizon=50):
    """Population variance of the same future action, summed over XYZ only."""
    if type(index) is not int or not 0 <= index < horizon:
        raise ValueError("Variance index outside predicted horizon")
    if len(clean_estimates) != tail_length:
        raise ValueError("Incorrect FM tail length")
    for estimate in clean_estimates:
        if len(estimate) != horizon:
            raise ValueError("Incorrect FM horizon")
        vector(estimate[index], 7, "clean action")
    result = []
    for coordinate in range(3):
        values = [estimate[index][coordinate] for estimate in clean_estimates]
        mean = math.fsum(values) / tail_length
        result.append(math.fsum((value - mean) ** 2 for value in values) / tail_length)
    return math.fsum(result)


def extract_episode(summary, steps, chunks, cohort, *, horizon=50, tail_length=5):
    """Align old action K with new action zero; never substitute old K-1."""
    if cohort not in {"capped", "supplemental"}:
        raise ValueError("Unknown evidence cohort")
    if not chunks or summary["model_calls"] != len(chunks):
        raise ValueError("Empty or inconsistent call inventory")
    if len(steps) != summary["steps"] or [row["step"] for row in steps] != list(range(len(steps))):
        raise ValueError("Invalid step inventory")
    if chunks[0]["step"] != 0 or steps[0].get("plan_revision_translation_l2") is not None:
        raise ValueError("The first prediction has no preceding plan")
    events = []
    exhausted = bootstrap = 0
    for index, (old, new) in enumerate(zip(chunks[:-1], chunks[1:], strict=True)):
        k = new["step"] - old["step"]
        if (type(k) is not int or not 1 <= k <= horizon
                or old["executed_steps"] != k or old["decision"]["chosen_k"] != k):
            raise ValueError("Adjacent call boundary disagrees with executed/selected K")
        if not 0 <= new["step"] < len(steps):
            raise ValueError("Next call is outside executed episode")
        boundary = steps[new["step"]]
        if boundary.get("replanned") is not True or boundary.get("chunk_index") != 0:
            raise ValueError("Missing replanning marker at next call")
        for chunk in (old, new):
            if len(chunk["actions"]) != horizon:
                raise ValueError("Incorrect prediction horizon")
            for action in chunk["actions"]:
                vector(action, 7, "predicted action")
        if k == horizon:
            if boundary.get("plan_revision_translation_l2") is not None:
                raise ValueError("An exhausted old plan must have no revision value")
            exhausted += 1
            continue
        previous, replacement = old["actions"][k], new["actions"][0]
        delta = [replacement[j] - previous[j] for j in range(3)]
        revision = math.sqrt(math.fsum(value * value for value in delta))
        equal_number(boundary.get("plan_revision_translation_l2"), revision,
                     "logged plan revision", rel_tol=1e-12, abs_tol=1e-14)
        clean = old["diagnostics"]["clean_estimates"]
        recalculated = recompute_variance(clean, tail_length=tail_length, horizon=horizon)
        logged = vector(old["diagnostics"]["variance"], horizon, "FM variance", 0)
        for actual, expected in zip(logged, recalculated, strict=True):
            equal_number(actual, expected, "FM variance replay")
        bootstrap_flag = old["decision"]["bootstrap"]
        if type(bootstrap_flag) is not bool:
            raise ValueError("Bootstrap flag must be boolean")
        bootstrap += int(bootstrap_flag)
        clipped_delta = [max(-1., min(1., replacement[j])) - max(-1., min(1., previous[j]))
                         for j in range(3)]
        events.append({
            "cohort": cohort, "episode_id": summary["episode_id"],
            **{key: summary[key] for key in ("suite", "task", "state", "policy_seed", "method")},
            "old_call_index": index, "old_step": old["step"], "new_step": new["step"], "k": k,
            "bootstrap": bootstrap_flag, "variance_total": recalculated[k],
            "variance_translation": variance_translation(clean, k, tail_length=tail_length, horizon=horizon),
            "revision_translation": revision,
            "revision_translation_clipped": math.sqrt(math.fsum(value * value for value in clipped_delta)),
            "gripper_flip": (previous[6] > 0) != (replacement[6] > 0),
            "gripper_old": previous[6], "gripper_new": replacement[6],
            "gripper_zero_involved": previous[6] == 0 or replacement[6] == 0,
            "old_translation_norm": math.sqrt(math.fsum(value * value for value in previous[:3])),
            "previous_revision": steps[old["step"]].get("plan_revision_translation_l2"),
            "previous_revision_missing": steps[old["step"]].get("plan_revision_translation_l2") is None,
            "step_fraction": old["step"] / summary["max_steps"],
            "old_command_gripper": previous[6],
            "old_sampling_seed": old["sampling_seed"], "new_sampling_seed": new["sampling_seed"],
            "old_threshold": old["decision"].get("threshold"),
            "old_cap_applied": old["decision"].get("cap_applied", False),
            "episode_success": summary["success"],
        })
    coverage = {
        "cohort": cohort, "episode_id": summary["episode_id"],
        **{key: summary[key] for key in ("suite", "task", "state", "policy_seed", "method")},
        "model_calls": len(chunks), "no_next_call": 1,
        "horizon_exhausted": exhausted, "measurable_transitions": len(events),
        "bootstrap_excluded": bootstrap, "included_nonbootstrap": len(events) - bootstrap,
        "mean_revision_all_measurable": statistics.mean(e["revision_translation"] for e in events) if events else None,
        "mean_revision_nonbootstrap": statistics.mean(e["revision_translation"] for e in events if not e["bootstrap"])
        if len(events) > bootstrap else None,
    }
    if len(chunks) != 1 + exhausted + len(events):
        raise ValueError("Transition coverage does not reconcile")
    return events, coverage


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty evidence export")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_verified(capped_root, supplemental_root):
    from scripts.verify_capped import load_and_verify as verify_capped
    from scripts.verify_dvac import load_and_verify as verify_supplemental

    print("Verifying retained capped and fixed-reference evidence...", flush=True)
    capped_cfg, capped_records, capped_audit = verify_capped(capped_root)
    print("Verifying the separate instrumented fixed/DVAC cohort...", flush=True)
    supplemental_cfg, supplemental_records, supplemental_audit = verify_supplemental(supplemental_root)
    jobs = [("capped", Path(capped_root), capped_cfg,
             [record for record in capped_records if record["origin"] == "capped"], capped_audit),
            ("supplemental", Path(supplemental_root), supplemental_cfg, supplemental_records, supplemental_audit)]
    all_events, coverage = [], []
    for cohort, root, cfg, records, audit in jobs:
        expected_hashes = {record["episode_id"]: record for record in audit["evidence"]}
        for record in records:
            summary = record["summary"]
            directory = root / "episodes" / summary["episode_id"]
            expected = expected_hashes[summary["episode_id"]]
            if sha256(directory / "summary.json") != expected["summary_sha256"]:
                raise ValueError("Summary changed after verification")
            for name in ("chunks.jsonl", "steps.jsonl"):
                if sha256(directory / name) != expected["files"][name]:
                    raise ValueError("Raw trace changed after verification")
            events, row = extract_episode(summary, read_jsonl(directory / "steps.jsonl"),
                                         read_jsonl(directory / "chunks.jsonl"), cohort,
                                         horizon=cfg.base.chunk_size, tail_length=cfg.dvac.tail_length)
            all_events.extend(events)
            coverage.append(row)
    if Counter(row["cohort"] for row in coverage) != {"capped": 120, "supplemental": 48}:
        raise ValueError("Expected the complete 120 + 48 recorded diagnostic episodes")
    return all_events, coverage, {"capped": capped_audit, "supplemental": supplemental_audit}


def analyze(capped_root, supplemental_root, output, *, draws=2000, seed=20261024, stage="all"):
    output = Path(output).resolve()
    for source in (Path(capped_root).resolve(), Path(supplemental_root).resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate from immutable evidence")
    output.mkdir(parents=True, exist_ok=True)
    if stage in {"extract", "all"}:
        events, coverage, verification = extract_verified(capped_root, supplemental_root)
        write_csv(output / "transitions.csv", events)
        write_csv(output / "episode_coverage.csv", coverage)
        atomic_json(output / "extracted.json", {"events": events, "coverage": coverage})
        atomic_json(output / "verification.json", verification)
        atomic_json(output / "extraction.json", {
            "extractor_sha256": sha256(__file__),
            "files": {name: sha256(output / name) for name in (
                "transitions.csv", "episode_coverage.csv", "extracted.json", "verification.json")}})
        if stage == "extract":
            return {"episodes": len(coverage), "measurable_transitions": len(events), "extraction_complete": True}
    else:
        extraction = json.loads((output / "extraction.json").read_text())
        if extraction["extractor_sha256"] != sha256(__file__):
            raise ValueError("Extractor changed since cached extraction")
        for name, checksum in extraction["files"].items():
            if Path(name).name != name or sha256(output / name) != checksum:
                raise ValueError("Cached extraction checksum mismatch")
        data = json.loads((output / "extracted.json").read_text())
        events, coverage = data["events"], data["coverage"]
        verification = json.loads((output / "verification.json").read_text())
    from scripts.plan_revision_statistics import analyze_statistics
    from scripts.plan_revision_prediction import compare_predictions

    print(f"Extracted {len(events)} measurable revisions from {len(coverage)} episodes; computing clustered statistics...", flush=True)
    prediction, predictions = compare_predictions(events, draws=draws, seed=seed)
    atomic_json(output / "prediction.json", prediction)
    write_csv(output / "predictions.csv", predictions)
    print("Held-out prediction comparison complete; calculating association diagnostics...", flush=True)
    result = analyze_statistics(events, draws=draws, seed=seed)
    result["prediction_comparison"] = prediction
    result["coverage"] = coverage
    result["units"] = {
        "revision_translation": "L2 difference of three post-unnormalization, pre-clipping relative controller command coordinates; not metres",
        "variance_total": "Sum of population variances of seven normalized action coordinates across the last five FM clean estimates",
        "variance_translation": "Same population variance summed over normalized XYZ coordinates only",
    }
    result["source_sha256"] = {name: sha256(PROJECT / name) for name in (
        "scripts/analyze_plan_revision.py", "scripts/plan_revision_statistics.py", "scripts/plan_revision_prediction.py")}
    result["raw_manifest_sha256"] = {key: value["manifest_sha256"] for key, value in verification.items()}
    atomic_json(output / "analysis.json", result)
    files = ["analysis.json", "verification.json", "transitions.csv", "episode_coverage.csv", "prediction.json", "predictions.csv"]
    atomic_json(output / "artifacts.json", {"source_sha256": result["source_sha256"],
                                           "files": {name: sha256(output / name) for name in files}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capped-root", type=Path, default=PROJECT / "outputs/capped")
    parser.add_argument("--supplemental-root", type=Path, default=PROJECT / "outputs/dvac")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/plan-revision-analysis")
    parser.add_argument("--stage", choices=("extract", "analyze", "all"), default="all")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    result = analyze(args.capped_root, args.supplemental_root, args.output,
                     draws=args.draws, seed=args.seed, stage=args.stage)
    print(json.dumps({key: value for key, value in result.items() if key != "coverage"}, indent=2))


if __name__ == "__main__":
    main()
