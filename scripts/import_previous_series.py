"""Import compact, path-independent evidence from the previous 300-episode series."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla_replan.storage import atomic_json  # noqa: E402


def import_series(source, output):
    source, output = Path(source), Path(output)
    campaign = source / "campaign_300"
    instrumentation = source / "run_eval.py"
    code_sha = hashlib.sha256(instrumentation.read_bytes()).hexdigest()
    rows, provenance = [], []
    for path in sorted((campaign / "runs").glob("*/run_manifest.json")):
        m = json.loads(path.read_text())
        telemetry_path = path.parent / "telemetry.json"
        t = json.loads(telemetry_path.read_text())
        assert m["status"] == t["status"] == "completed"
        assert m["source_sha256"] == code_sha
        assert len(t["episodes"]) == 10
        provenance.append({"cell": path.parent.name, "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "telemetry_sha256": hashlib.sha256(telemetry_path.read_bytes()).hexdigest()})
        for ep in t["episodes"]:
            assert ep["finished"] and ep["model_calls"] == math.ceil(ep["steps"]/m["k"])
            rows.append({"suite": m["suite"], "task_id": m["task_id"], "task": ep["task_name"],
                         "state": ep["init_state_index"], "policy_seed": ep["seed"], "k": m["k"],
                         "success": ep["success"], "steps": ep["steps"], "model_calls": ep["model_calls"],
                         "inference_s": ep["inference_seconds"], "rollout_s": ep["rollout_seconds"],
                         "initial_state_sha256": ep["initial_state_sha256"],
                         "settled_state_sha256": ep["settled_state_sha256"],
                         "image_sha256": ep["initial_image_sha256"]["image"],
                         "image2_sha256": ep["initial_image_sha256"]["image2"]})
    assert len(provenance) == 30 and len(rows) == 300
    pairs = defaultdict(list)
    for r in rows:
        pairs[(r["suite"],r["task_id"],r["state"])].append(r)
    assert len(pairs) == 60
    for pair in pairs.values():
        assert {r["k"] for r in pair} == {1,5,10,25,50}
        for field in ("initial_state_sha256", "settled_state_sha256", "image_sha256", "image2_sha256", "policy_seed"):
            assert len({r[field] for r in pair}) == 1
    summary = json.loads((campaign / "analysis/summary.json").read_text())
    for s in summary["by_k"]:
        values = [r for r in rows if r["k"] == s["k"]]
        assert len(values) == s["n"] and sum(r["success"] for r in values) == s["successes"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "paired_comparisons.json", json.loads((campaign / "analysis/paired_comparisons.json").read_text()))
    atomic_json(output / "provenance.json", {
        "origin": "Earlier local campaign_300, imported as a separate experiment; not pooled with compact or research_v1",
        "instrumentation_sha256": code_sha,
        "environment": json.loads((campaign / "environment.json").read_text()),
        "sampling_protocol": "Global CPU/MPS RNG reset per episode; noise then follows model-call order. New study uses private noise indexed by simulation step.",
        "checks": {"episodes": len(rows), "paired_cases": len(pairs), "complete_cells": len(provenance),
                   "all_initial_states_and_both_camera_hashes_match": True, "all_call_counts_valid": True},
        "files": provenance,
    })
    (output / "legacy_instrumentation.py.txt").write_bytes(instrumentation.read_bytes())
    probe_path = source / "failure_analysis/probe_summary.json"
    if probe_path.exists():
        probes = json.loads(probe_path.read_text())
        driver = source / "run_diagnostic.py"
        driver_sha = hashlib.sha256(driver.read_bytes()).hexdigest()
        compact = []
        for row in probes["results"]:
            directory = Path(row["directory"])
            manifest = json.loads((directory / "run_manifest.json").read_text())
            assert manifest["status"] == "completed" and manifest["source_sha256"] == driver_sha
            assert row["prefix_sim_max_difference"] == row["first_chunk_max_difference"] == 0
            compact.append({k:v for k,v in row.items() if k != "directory"})
        assert len(compact) == 12
        atomic_json(output / "selected_failure_probes.json", {"scope": probes["scope"],
            "selection": "Four selected K1 failures from earlier checkpoint; exploratory, not population success estimates",
            "diagnostic_driver_sha256": driver_sha, "results": compact})
        (output / "legacy_diagnostic.py.txt").write_bytes(driver.read_bytes())
    print(f"Imported {len(rows)} episodes in {len(pairs)} paired situations to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/previous_300"))
    args = parser.parse_args()
    import_series(args.source, args.output)
