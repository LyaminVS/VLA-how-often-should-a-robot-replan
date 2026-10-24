"""Verify capped DVAC math/execution and reuse only matching historical fixed controls.

Capping is independently replayed after the original full-H DVAC decision.
Historical traces are rechecked against their existing evidence hashes. These
artifacts are read-only; normalized analysis records are created in memory.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from capped_replan.config import CappedExperiment
from scripts.verify_dvac import (equal_number, finite, read_jsonl, recompute_variance,
                                 replay_decision, sha256, vector, verify_video)
from scripts.verify_evidence import verify as verify_fixed_evidence
from vla_replan.analysis import load_results
from vla_replan.config import digest
from vla_replan.storage import atomic_json


def verify_episode(summary, rows, chunks, cfg):
    identity = summary["episode_id"]
    if summary["method"].startswith("dvac_cap_"):
        declared_cap = int(summary["method"].rsplit("_", 1)[1])
        if (type(summary.get("execution_cap")) is not int
                or summary["execution_cap"] != declared_cap
                or not cfg.dvac.min_steps <= declared_cap <= cfg.base.chunk_size
                or summary.get("k") is not None):
            raise ValueError("Adaptive method identity disagrees with its execution cap")
    elif summary["method"].startswith("fixed_"):
        if (summary.get("execution_cap") is not None or type(summary.get("k")) is not int
                or summary["k"] != int(summary["method"].rsplit("_", 1)[1])):
            raise ValueError("Fixed method identity disagrees with its interval")
    else:
        raise ValueError("Unrecognized capped-comparison method")
    count = summary.get("steps")
    if type(count) is not int or not 1 <= count <= summary["max_steps"]:
        raise ValueError(f"Invalid episode length: {identity}")
    if [row.get("step") for row in rows] != list(range(count)) or not chunks:
        raise ValueError(f"Invalid step sequence or empty predictions: {identity}")
    if type(summary.get("model_calls")) is not int or summary["model_calls"] != len(chunks):
        raise ValueError(f"Invalid model call count: {identity}")
    horizon, tail = cfg.base.chunk_size, cfg.dvac.tail_length
    history, horizons, compact, next_step = [], [], [], 0
    raw_horizons, capped_calls = [], 0
    max_variance_error = 0.0
    clipped = closes = 0
    previous_action = None
    for call_index, chunk in enumerate(chunks):
        if chunk.get("step") != next_step:
            raise ValueError(f"Invalid adaptive replan boundary: {identity}")
        expected_seed = (summary["policy_seed"] * 1_000_003 + next_step * 97_409) % (2**63 - 1)
        if chunk.get("sampling_seed") != expected_seed:
            raise ValueError(f"Incorrect sampling seed: {identity}")
        actions = chunk.get("actions")
        if not isinstance(actions, list) or len(actions) != horizon:
            raise ValueError(f"Incorrect prediction horizon: {identity}")
        for action in actions:
            vector(action, 7, "predicted action")
        diagnostic = chunk.get("diagnostics", {})
        calculated = recompute_variance(diagnostic.get("clean_estimates"), tail_length=tail, horizon=horizon)
        logged = vector(diagnostic.get("variance"), horizon, "logged variance", 0)
        for actual, expected in zip(logged, calculated, strict=True):
            equal_number(actual, expected, "denoising variance")
            max_variance_error = max(max_variance_error, abs(actual - expected))
        times = vector(diagnostic.get("fm_times"), tail, "flow times")
        expected_times = [1 - index / cfg.base.num_denoising_steps
                          for index in range(cfg.base.num_denoising_steps - tail, cfg.base.num_denoising_steps)]
        for actual, expected in zip(times, expected_times, strict=True):
            equal_number(actual, expected, "flow-time schedule", rel_tol=0, abs_tol=1e-6)
        if (diagnostic.get("velocity_evaluations") != cfg.base.num_denoising_steps
                or diagnostic.get("action_dimensions") != 7
                or diagnostic.get("statistics_dtype") != "float64"):
            raise ValueError("Invalid sampler diagnostic metadata")
        finite(diagnostic.get("diagnostics_finalize_s"), "diagnostics_finalize_s")
        if summary["method"].startswith("dvac_cap_"):
            expected_decision = replay_decision(calculated, history, alpha=cfg.dvac.alpha, minimum=cfg.dvac.min_steps)
            history.append(calculated)
            history = history[-cfg.dvac.window_size:]
        else:
            expected_decision = {"chosen_k": summary["k"], "threshold": None, "first_crossing": None,
                                 "window_calls": 0, "window_mean": None, "window_std": None, "bootstrap": False}
        raw = expected_decision["chosen_k"]
        cap = summary["execution_cap"]
        expected_decision.update(raw_chosen_k=raw, execution_cap=cap,
                                 cap_applied=cap is not None and raw > cap,
                                 chosen_k=min(raw, cap) if cap is not None else raw)
        raw_horizons.append(raw)
        capped_calls += int(expected_decision["cap_applied"])
        decision = chunk.get("decision", {})
        for field in ("chosen_k", "raw_chosen_k", "execution_cap", "cap_applied",
                      "first_crossing", "window_calls", "bootstrap"):
            if decision.get(field) != expected_decision[field] or type(decision.get(field)) is not type(expected_decision[field]):
                raise ValueError(f"Decision replay mismatch for {field}: {identity}, call {call_index}")
        for field in ("threshold", "window_mean", "window_std"):
            equal_number(decision.get(field), expected_decision[field], field)
        k = expected_decision["chosen_k"]
        horizons.append(k)
        executed = min(k, count - next_step)
        if executed <= 0 or chunk.get("executed_steps") != executed:
            raise ValueError(f"Incorrect executed prefix length: {identity}")
        for field in ("inference_s", "selector_s"):
            finite(chunk.get(field), field)
        if summary["method"].startswith("fixed_") and chunk["selector_s"] != 0:
            raise ValueError("Fixed methods should not run the DVAC selector")
        for offset in range(executed):
            step = next_step + offset
            row = rows[step]
            predicted = actions[offset]
            expected_action = [max(-1.0, min(1.0, value)) for value in predicted]
            actual = vector(row.get("action"), 7, "executed action")
            if actual != expected_action:
                raise ValueError(f"Executed action differs from final predicted prefix: {identity}, step {step}")
            for field, expected in (("chunk_index", offset), ("model_calls", call_index + 1),
                                    ("chosen_k", k), ("next_replan_step", next_step + k)):
                if row.get(field) != expected or type(row.get(field)) is not int:
                    raise ValueError(f"Invalid step {field}: {identity}")
            for field in ("raw_chosen_k", "execution_cap", "cap_applied"):
                if row.get(field) != decision[field] or type(row.get(field)) is not type(decision[field]):
                    raise ValueError(f"Invalid capped step metadata {field}: {identity}")
            if row.get("replanned") is not (offset == 0):
                raise ValueError(f"Invalid replanning marker: {identity}")
            equal_number(row.get("sim_time_s"), step / cfg.base.control_hz, "simulation timestamp")
            equal_number(row.get("threshold"), decision["threshold"], "step threshold")
            for field in ("inference_s", "selector_s"):
                equal_number(row.get(field), chunk[field] if offset == 0 else 0.0, field)
            if type(row.get("success_after_action")) is not bool:
                raise ValueError("Missing boolean post-action success")
            if row["success_after_action"] and step != count - 1:
                raise ValueError("Actions continued after task completion")
            clipped += sum(left != right for left, right in zip(predicted, expected_action, strict=True))
            closes += int(actual[6] > 0 and (previous_action is None or previous_action[6] <= 0))
            previous_action = actual
        compact.append({key: value for key, value in chunk.items() if key not in ("actions", "diagnostics")}
                       | {"variance": calculated, "fm_times": times,
                          "diagnostics_finalize_s": diagnostic["diagnostics_finalize_s"]})
        next_step += executed
    if next_step != count:
        raise ValueError("Predictions do not cover all executed actions")
    for field in ("inference_s", "selector_s"):
        equal_number(summary.get(field), math.fsum(chunk[field] for chunk in chunks), "total " + field, abs_tol=1e-8)
    for field in ("simulation_s", "compute_s", "episode_wall_s", "sim_time_s", "eef_path_m"):
        finite(summary.get(field), field)
    equal_number(summary["compute_s"], summary["inference_s"] + summary["selector_s"] + summary["simulation_s"], "compute total", abs_tol=1e-8)
    equal_number(summary["sim_time_s"], count / cfg.base.control_hz, "episode simulation duration")
    for field, expected in (("mean_chosen_k", math.fsum(horizons) / len(horizons)),
                            ("min_chosen_k", min(horizons)), ("max_chosen_k", max(horizons)),
                            ("mean_executed_prefix", count / len(chunks))):
        equal_number(summary.get(field), expected, field)
    if (summary.get("clipped_action_values") != clipped or summary.get("gripper_close_commands") != closes
            or summary.get("bootstrap_calls") != int(summary["method"].startswith("dvac_cap_"))):
        raise ValueError("Summary command or bootstrap counts disagree with evidence")
    equal_number(summary.get("mean_raw_chosen_k"), math.fsum(raw_horizons) / len(raw_horizons),
                 "mean raw chosen horizon")
    if type(summary.get("cap_applied_calls")) is not int or summary["cap_applied_calls"] != capped_calls:
        raise ValueError("Summary cap count disagrees with replay")
    success = any(row["success_after_action"] for row in rows)
    if type(summary.get("success")) is not bool or summary["success"] != success:
        raise ValueError("Summary success disagrees with executed action outcomes")
    if success and summary.get("termination") != "success":
        raise ValueError("Incorrect successful termination")
    if not success and summary.get("termination") not in ("terminated", "time_limit"):
        raise ValueError("Incorrect failed termination")
    if summary["termination"] == "time_limit" and count != summary["max_steps"]:
        raise ValueError("Time limit was reported before the configured horizon")
    return compact, {"steps": count, "model_calls": len(chunks), "max_variance_error": max_variance_error,
                     "clean_estimate_values": len(chunks) * tail * horizon * 7,
                     "executed_action_values": count * 7}


MATCH_FIELDS = (
    "model_id", "model_revision", "backbone_id", "backbone_revision", "chunk_size",
    "num_denoising_steps", "control_hz", "settle_steps", "precision", "record_video", "warmup_calls",
)


def first_prediction(chunk):
    return digest({"actions": chunk["actions"], "sampling_seed": chunk["sampling_seed"]})


def normalized_fixed(summary, chunks):
    """Preserve measured values while exposing equivalent analysis-only fields."""
    k = summary["k"]
    normalized = {**summary, "method": f"fixed_{k}", "execution_cap": None,
                  "selector_s": 0.0, "mean_chosen_k": float(k), "min_chosen_k": k,
                  "max_chosen_k": k, "mean_raw_chosen_k": float(k), "cap_applied_calls": 0,
                  "mean_executed_prefix": summary["steps"] / summary["model_calls"],
                  "bootstrap_calls": 0}
    compact = []
    for chunk in chunks:
        compact.append({key: value for key, value in chunk.items() if key != "actions"} | {
            "executed_steps": min(k, summary["steps"] - chunk["step"]),
            "selector_s": 0.0,
            "decision": {"chosen_k": k, "raw_chosen_k": k, "execution_cap": None,
                         "cap_applied": False, "threshold": None, "first_crossing": None,
                         "window_calls": 0, "window_mean": None, "window_std": None,
                         "bootstrap": False},
        })
    return normalized, compact


def verify_reference_paths(audit, reference_roots):
    """Relocate only declared reference roots, matched by their manifest checksums."""
    recorded_roots = audit.get("roots", [])
    files = audit.get("files", {})
    actual_roots = [Path(path).resolve() for path in reference_roots]
    if (not recorded_roots or len(set(recorded_roots)) != len(recorded_roots)
            or len(set(actual_roots)) != len(actual_roots)
            or len(recorded_roots) != len(actual_roots)):
        raise ValueError("Historical reference root inventory mismatch")
    mapping = {}
    for original in recorded_roots:
        recorded = Path(original)
        if not recorded.is_absolute():
            raise ValueError("Recorded reference roots must be absolute")
        expected_sha = files.get(str(recorded / "manifest.json"))
        candidates = [root for root in actual_roots
                      if (root / "manifest.json").is_file() and sha256(root / "manifest.json") == expected_sha]
        if len(candidates) != 1:
            raise ValueError("Cannot uniquely relocate a recorded reference manifest")
        mapping[recorded] = candidates[0]
    if len(set(mapping.values())) != len(actual_roots):
        raise ValueError("Multiple recorded references map to one actual root")
    relocated = {}
    for original, checksum in files.items():
        path = Path(original)
        matches = [(recorded, actual) for recorded, actual in mapping.items() if path.is_relative_to(recorded)]
        if len(matches) != 1:
            raise ValueError("Reference evidence lies outside a declared root")
        recorded, actual = matches[0]
        local = (actual / path.relative_to(recorded)).resolve()
        if not local.is_relative_to(actual) or not local.is_file() or sha256(local) != checksum:
            raise ValueError(f"Historical reference checksum mismatch: {original}")
        if local in relocated:
            raise ValueError("Duplicate relocated reference evidence")
        relocated[local] = checksum
    return mapping, relocated


def load_fixed_controls(cfg, manifest, reference_roots, decode_videos=False):
    audit = manifest.get("reference_sources")
    if not isinstance(audit, dict) or not audit:
        raise ValueError("Historical reuse requires recorded reference provenance")
    mapping, declared_files = verify_reference_paths(audit, reference_roots)
    if audit.get("config_match_fields") != [*MATCH_FIELDS, "tasks.suite/name/max_steps"]:
        raise ValueError("Historical configuration comparison fields changed")
    required_pairs = {episode.pair_id for episode in cfg.episodes()}
    intervals = set(cfg.base.intervals)
    expected_settings = {(pair_id, k) for pair_id in required_pairs for k in intervals}
    observed_settings = set()
    expected_files = set()
    records, references, reports = [], {}, []
    base = cfg.base.asdict()
    runtime_rows = audit.get("runtime_comparison", [])
    if len(runtime_rows) != len(mapping):
        raise ValueError("Missing historical runtime comparison")
    runtime_gate = {
        "runtime_equal": True, "hardware_equal_except": ["torch_cpu_threads"],
        "differences": [{"root": row["root"],
                         "historical_torch_cpu_threads": row["hardware"].get("torch_cpu_threads"),
                         "current_torch_cpu_threads": manifest["hardware"].get("torch_cpu_threads")}
                        for row in runtime_rows],
    }
    if manifest.get("reference_runtime_validation") != runtime_gate:
        raise ValueError("Missing or inconsistent historical runtime gate evidence")
    for recorded, root in mapping.items():
        old_manifest = json.loads((root / "manifest.json").read_text())
        if any(old_manifest["config"].get(field) != base[field] for field in MATCH_FIELDS):
            raise ValueError("Historical controls have different inference conditions")
        if old_manifest.get("runtime") != manifest["runtime"]:
            raise ValueError("Historical model/simulator runtime differs")
        hardware = {key: value for key, value in old_manifest["hardware"].items() if key != "torch_cpu_threads"}
        current_hardware = {key: value for key, value in manifest["hardware"].items() if key != "torch_cpu_threads"}
        if hardware != current_hardware:
            raise ValueError("Historical hardware differs beyond CPU thread count")
        runtime_row = [row for row in runtime_rows if row.get("root") == str(recorded)]
        if (len(runtime_row) != 1 or runtime_row[0].get("runtime") != old_manifest["runtime"]
                or runtime_row[0].get("hardware") != old_manifest["hardware"]):
            raise ValueError("Recorded historical runtime metadata does not match its source")
        expected_files.add((root / "manifest.json").resolve())
        actual_report = verify_fixed_evidence(root, decode_video=False)
        saved_path = root / "report" / "evidence-verification.json"
        saved = json.loads(saved_path.read_text())
        saved_episodes = {item["episode_id"]: item for item in saved.get("episodes", [])}
        if (saved.get("passed") is not True or len(saved_episodes) != len(saved.get("episodes", []))
                or set(saved_episodes) != {item["episode_id"] for item in actual_report["episodes"]}):
            raise ValueError("Historical evidence audit is missing or incomplete")
        for item in actual_report["episodes"]:
            if item["sha256"] != saved_episodes[item["episode_id"]].get("sha256"):
                raise ValueError(f"Historical evidence changed since its audit: {item['episode_id']}")
        old_cfg, summaries, _ = load_results(root)
        task_limits = {(task.suite, task.name): task.max_steps for task in cfg.base.tasks}
        for summary in summaries:
            if summary["pair_id"] not in required_pairs:
                continue
            setting = (summary["pair_id"], summary["k"])
            if setting not in expected_settings or setting in observed_settings:
                raise ValueError("Duplicate or unexpected fixed control in combined comparison")
            if summary["max_steps"] != task_limits.get((summary["suite"], summary["task"])):
                raise ValueError("Historical task horizon differs")
            observed_settings.add(setting)
            directory = root / "episodes" / summary["episode_id"]
            expected_files.update((directory / name).resolve() for name in ("summary.json", "chunks.jsonl"))
            chunks = read_jsonl(directory / "chunks.jsonl")
            reference = {"initial_state_sha256": summary["initial_state_sha256"],
                         "initial_observation_sha256": summary["initial_observation_sha256"],
                         "first_prediction_sha256": first_prediction(chunks[0])}
            previous = references.setdefault(summary["pair_id"], reference)
            if previous != reference:
                raise ValueError("Historical first predictions or initial conditions differ within a pair")
            normalized, compact = normalized_fixed(summary, chunks)
            records.append({"summary": normalized, "chunks": compact,
                            "origin": "reused_fixed", "raw_source_root": str(root)})
            if decode_videos:
                verify_video(directory / "video.mp4", summary["steps"], old_cfg.control_hz)
        reports.append({"root": str(root), "recorded_root": str(recorded),
                        "manifest_sha256": sha256(root / "manifest.json"),
                        "saved_audit_sha256": sha256(saved_path),
                        "episodes": len(summaries), "evidence_unchanged": True,
                        "trace_replay_passed": True, "videos_decoded": decode_videos,
                        "runtime": old_manifest["runtime"], "hardware": old_manifest["hardware"]})
    if observed_settings != expected_settings or set(references) != required_pairs:
        raise ValueError("Historical controls do not completely cover the combined comparison")
    if set(declared_files) != expected_files:
        raise ValueError("Historical reference checksum inventory differs from the required controls")
    if audit.get("pairs") != len(required_pairs) or audit.get("episodes") != len(records):
        raise ValueError("Historical reference summary counts disagree")
    return records, references, reports


def load_and_verify(root, *, decode_videos=False, check_sources=True, reference_roots=None):
    root = Path(root).resolve()
    reference_roots = reference_roots or [PROJECT / "outputs/compact", PROJECT / "outputs/validation"]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = CappedExperiment.fromdict(manifest["config"])
    if (manifest.get("config_fingerprint") != cfg.fingerprint
            or manifest.get("data_kind") != "real_libero"
            or manifest.get("experiment_kind") != "capped_dvac_comparison"):
        raise ValueError("Invalid capped comparison manifest")
    expected = {episode.id: episode for episode in cfg.episodes()}
    if (manifest.get("full_plan_episode_ids") != list(expected)
            or manifest.get("planned_episode_ids") != list(expected)):
        raise ValueError("Final comparison requires the complete configuration plan, not a selected subset")
    actual = {path.name for path in (root / "episodes").iterdir() if path.is_dir()}
    if actual != set(expected):
        raise ValueError(f"Incomplete or unexpected capped episode inventory: {len(actual)}/{len(expected)}")
    for field in ("runtime", "hardware", "capped_source", "upstream_sampler"):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise ValueError(f"Missing collection metadata: {field}")
    invariance_path = root / "invariance.json"
    if sha256(invariance_path) != manifest.get("invariance_check_sha256"):
        raise ValueError("Invariance-check evidence checksum mismatch")
    invariance = json.loads(invariance_path.read_text())
    checks = invariance.get("checks", [])
    expected_tasks = {(task.suite, task.name) for task in cfg.base.tasks}
    if (invariance.get("passed") is not True or invariance.get("config_fingerprint") != cfg.fingerprint
            or invariance.get("source_manifest") != manifest["capped_source"]
            or invariance.get("sampler_manifest") != manifest["upstream_sampler"]
            or invariance.get("runtime") != manifest["runtime"]
            or invariance.get("hardware") != manifest["hardware"]
            or len(checks) != len(expected_tasks)
            or {(item.get("suite"), item.get("task")) for item in checks} != expected_tasks
            or any(item.get("exact_action_equality") is not True
                   or item.get("max_absolute_action_error") != 0 for item in checks)):
        raise ValueError("Invariance evidence does not match the recorded setup")
    if check_sources:
        for relative, checksum in manifest["capped_source"].items():
            source = (PROJECT / relative).resolve()
            if not source.is_relative_to(PROJECT) or not source.is_file() or sha256(source) != checksum:
                raise ValueError(f"Collection source checksum mismatch: {relative}")
    records, references, fixed_report = load_fixed_controls(cfg, manifest, reference_roots, decode_videos)
    fixed_count = len(records)
    first_diagnostics, starts = defaultdict(set), defaultdict(set)
    checked, totals = [], defaultdict(int)
    max_variance_error = 0.0
    manifest_sha = sha256(manifest_path)
    for identity, episode in expected.items():
        directory = root / "episodes" / identity
        summary = json.loads((directory / "summary.json").read_text())
        if (summary.get("episode_id") != identity or summary.get("pair_id") != episode.pair_id
                or any(summary.get(key) != value or type(summary.get(key)) is not type(value)
                       for key, value in episode.asdict().items())):
            raise ValueError(f"Capped episode identity mismatch: {identity}")
        if (summary.get("config_fingerprint") != cfg.fingerprint
                or summary.get("manifest_sha256") != manifest_sha
                or summary.get("data_kind") != "real_libero" or summary.get("env_seed") != episode.env_seed):
            raise ValueError("Capped episode does not match its manifest")
        reference = references[episode.pair_id]
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            if summary.get(field) != reference[field]:
                raise ValueError("Capped initial conditions differ from historical controls")
            starts[(episode.suite, episode.task, episode.state, field)].add(summary[field])
        if (summary.get("reference_pair_id") != episode.pair_id
                or summary.get("reference_initial_state_match") is not True
                or summary.get("reference_first_prediction_equal") is not True
                or summary.get("reference_max_absolute_action_error") != 0):
            raise ValueError("Missing per-episode historical reference gate evidence")
        filenames = ["steps.jsonl", "chunks.jsonl"] + (["video.mp4"] if cfg.base.record_video else [])
        hashes = summary.get("evidence_sha256", {})
        if set(hashes) != set(filenames):
            raise ValueError("Capped evidence hash inventory mismatch")
        for filename in filenames:
            path = directory / filename
            if not path.is_file() or not path.stat().st_size or sha256(path) != hashes[filename]:
                raise ValueError(f"Capped evidence checksum mismatch: {path}")
        rows, chunks = read_jsonl(directory / "steps.jsonl"), read_jsonl(directory / "chunks.jsonl")
        compact, result = verify_episode(summary, rows, chunks, cfg)
        if first_prediction(chunks[0]) != reference["first_prediction_sha256"]:
            raise ValueError("First complete prediction differs from historical fixed controls")
        first_diagnostics[episode.pair_id].add(digest(chunks[0]["diagnostics"]["clean_estimates"]))
        record = {"episode_id": identity, "summary_sha256": sha256(directory / "summary.json"),
                  "files": hashes, **result}
        if decode_videos and cfg.base.record_video:
            record["video"] = verify_video(directory / "video.mp4", summary["steps"], cfg.base.control_hz)
        for key in ("steps", "model_calls", "clean_estimate_values", "executed_action_values"):
            totals[key] += result[key]
        max_variance_error = max(max_variance_error, result["max_variance_error"])
        checked.append(record)
        records.append({"summary": summary, "chunks": compact,
                        "origin": "capped", "raw_source_root": str(root)})
    if any(len(values) != 1 for values in starts.values()) or any(len(values) != 1 for values in first_diagnostics.values()):
        raise ValueError("Capped methods disagree on initial conditions or first diagnostic trajectory")
    settings = [(r["summary"]["pair_id"], r["summary"]["method"]) for r in records]
    if len(set(settings)) != len(settings):
        raise ValueError("A combined task/state/seed/method setting appears twice")
    report = {
        "complete": True, "episodes": len(records), "fixed_episodes": fixed_count,
        "capped_episodes": len(checked), "planned_capped_episodes": len(expected),
        "matched_task_state_seed_blocks": len(first_diagnostics),
        "physical_task_state_settings": len(starts) // 2,
        "manifest_sha256": manifest_sha, "config_fingerprint": cfg.fingerprint,
        "invariance_sha256": sha256(invariance_path), "verifier_source_sha256": sha256(__file__),
        "verification_dependency_sha256": {name: sha256(PROJECT / name) for name in (
            "scripts/verify_dvac.py", "scripts/verify_evidence.py", "vla_replan/analysis.py")},
        "collection_source_checks": check_sources, "videos_decoded": decode_videos,
        "first_prediction_equality_all_methods": True,
        "max_variance_error": max_variance_error, "capped_totals": dict(totals),
        "historical_controls": fixed_report, "evidence": checked,
        "timing_note": "Historical fixed controls did not collect FM diagnostics and used a different CPU thread count. Their raw times are retained but are not a controlled latency comparison with new capped runs.",
    }
    return cfg, records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference-roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-videos", action="store_true")
    args = parser.parse_args()
    _, _, result = load_and_verify(args.input, decode_videos=args.decode_videos,
                                  reference_roots=args.reference_roots)
    if args.output:
        protected = [args.input.resolve()] + [Path(root).resolve() for root in (args.reference_roots or [PROJECT / "outputs/compact", PROJECT / "outputs/validation"])]
        if any(args.output.resolve().is_relative_to(root) for root in protected):
            raise ValueError("Verification output must be outside immutable raw evidence")
        atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("evidence", "historical_controls")}, indent=2))


if __name__ == "__main__":
    main()


