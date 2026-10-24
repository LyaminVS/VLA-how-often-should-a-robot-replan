"""Independently verify action execution and the controlled pre-motion prefix.

This checks saved evidence only; it does not rerun the policy or infer grasp events.
The optional video check decodes every frame sequentially with one decoder thread.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from moving_target.config import CONDITIONS, MovingExperiment
from moving_target.runner import source_metadata
from scripts.analyze_moving_target import DATA_KIND, load_run, read_jsonl, sha256
from vla_replan.config import digest
from vla_replan.core import sampling_seed
from vla_replan.storage import atomic_json


def numeric_vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Expected {length} numbers: {label}")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"Nonfinite or nonnumeric value: {label}")
    return value


def verify_action_log(summary, rows, chunks, chunk_size=50, control_hz=20):
    """Check all predicted actions, executed prefixes, seeds and outcome bookkeeping."""
    identity = summary.get("episode_id", "episode")
    k, count = summary["k"], summary["steps"]
    expected_steps = list(range(0, count, k))
    if [chunk.get("step") for chunk in chunks] != expected_steps:
        raise ValueError(f"Invalid chunk schedule: {identity}")
    if [row.get("step") for row in rows] != list(range(count)):
        raise ValueError(f"Invalid step sequence: {identity}")
    if type(summary["model_calls"]) is not int or summary["model_calls"] != len(chunks):
        raise ValueError(f"Model call total disagrees with saved chunks: {identity}")
    for chunk in chunks:
        if chunk.get("sampling_seed") != sampling_seed(summary["policy_seed"], chunk["step"]):
            raise ValueError(f"Incorrect sampling seed: {identity}, step {chunk['step']}")
        actions = chunk.get("actions")
        if not isinstance(actions, list) or len(actions) != chunk_size:
            raise ValueError(f"Incorrect predicted chunk horizon: {identity}")
        for action in actions:
            numeric_vector(action, 7, "predicted action")
        latency = chunk.get("inference_s")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError(f"Invalid inference latency: {identity}")
    if not math.isclose(sum(chunk["inference_s"] for chunk in chunks), summary["inference_s"], abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"Inference-time total differs from saved chunks: {identity}")
    clipped_values = close_commands = 0
    previous_action = None
    max_action_error = 0.0
    for row in rows:
        step = row["step"]
        index = step % k
        chunk = chunks[step // k]
        predicted = chunk["actions"][index]
        expected = [max(-1.0, min(1.0, value)) for value in predicted]
        actual = numeric_vector(row.get("action"), 7, "executed action")
        error = max(abs(left - right) for left, right in zip(actual, expected, strict=True))
        max_action_error = max(max_action_error, error)
        if actual != expected:
            raise ValueError(f"Executed action is not the clipped predicted prefix: {identity}, step {step}, maximum error {error}")
        if type(row.get("chunk_index")) is not int or row["chunk_index"] != index:
            raise ValueError(f"Invalid executed chunk index: {identity}, step {step}")
        if row.get("replanned") is not (index == 0):
            raise ValueError(f"Replan marker differs from fixed K schedule: {identity}")
        if type(row.get("model_calls")) is not int or row["model_calls"] != step // k + 1:
            raise ValueError(f"Per-step model call total is incorrect: {identity}")
        if not math.isclose(row.get("sim_time_s", -1), step / control_hz, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"Step timestamp is not at the configured control rate: {identity}")
        if type(row.get("success_after_action")) is not bool:
            raise ValueError(f"Missing boolean post-action outcome: {identity}")
        if row.get("inference_s") != (chunk["inference_s"] if index == 0 else 0.0):
            raise ValueError(f"Step inference time differs from the prediction schedule: {identity}")
        if row["success_after_action"] and step != count - 1:
            raise ValueError(f"Actions continued after success: {identity}")
        clipped_values += sum(value != original for value, original in zip(expected, predicted, strict=True))
        if actual[6] > 0 and (previous_action is None or previous_action[6] <= 0):
            close_commands += 1
        previous_action = actual
    if type(summary.get("clipped_action_values")) is not int or summary["clipped_action_values"] != clipped_values:
        raise ValueError(f"Clipped action count mismatch: {identity}")
    if type(summary.get("gripper_close_commands")) is not int or summary["gripper_close_commands"] != close_commands:
        raise ValueError(f"Gripper close-command count mismatch: {identity}")
    success = any(row["success_after_action"] for row in rows)
    if type(summary.get("success")) is not bool or summary["success"] != success:
        raise ValueError(f"Summary success disagrees with post-action outcomes: {identity}")
    termination = summary.get("termination")
    if success and termination != "success":
        raise ValueError(f"Successful episode has inconsistent termination: {identity}")
    if not success and termination not in ("time_limit", "terminated"):
        raise ValueError(f"Failed episode has inconsistent termination: {identity}")
    if termination == "time_limit" and count != summary["max_steps"]:
        raise ValueError(f"Time limit reported before the configured horizon: {identity}")
    return {"steps": count, "model_calls": len(chunks), "predicted_action_values": len(chunks) * chunk_size * 7,
            "executed_action_values": count * 7, "max_executed_action_error": max_action_error}


def verify_force_timing(summary, rows, delay_s, offset_z_m, max_force_n):
    """Require exactly zero intervention before onset/static/after latched contact."""
    motions = [(row["sim_time_s"], row["motion"]) for row in rows]
    motions.append((summary["sim_time_s"], summary["motion_final"]))
    latched = None
    for sim_time, motion in motions:
        force, torque = motion["horizontal_force_n"], motion["applied_torque_y_nm"]
        if type(force) not in (int, float) or type(torque) not in (int, float) or not math.isfinite(force + torque):
            raise ValueError("Intervention force and torque must be finite")
        if abs(force) > max_force_n + 1e-9 or not math.isclose(torque, offset_z_m * force, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Intervention violates its force cap or application point")
        contact = motion.get("first_contact_sim_time")
        if contact is not None:
            if latched is not None and contact != latched:
                raise ValueError("The first-contact latch changed")
            latched = contact
        elif latched is not None:
            raise ValueError("The first-contact latch disappeared")
        must_be_zero = summary["condition"] == "static" or sim_time < delay_s or latched is not None
        if must_be_zero and (force != 0.0 or torque != 0.0):
            raise ValueError("Nonzero intervention before onset, in static control, or after contact")
        if (summary["condition"] == "static" or sim_time < delay_s or latched is not None) and motion.get("motion_active") is not False:
            raise ValueError("Intervention active flag contradicts the protocol")
    if summary["motion_final"]["max_force_after_contact_n"] != 0.0:
        raise ValueError("Physics-substep evidence contains force after contact")


def pre_motion_prefix(summary, rows, chunks, delay_s, control_hz):
    """Keep only actions/predictions made strictly before motion onset."""
    return {
        "identity": summary["episode_id"],
        "condition": summary["condition"],
        "actions": [row["action"] for row in rows if row["step"] / control_hz < delay_s],
        "predictions": [{"step": chunk["step"], "sampling_seed": chunk["sampling_seed"], "actions": chunk["actions"]}
                        for chunk in chunks if chunk["step"] / control_hz < delay_s],
    }


def compare_prefixes(treatments):
    """Exact equality is required; no tolerance can conceal a pre-treatment change."""
    if set(treatments) != set(CONDITIONS):
        raise ValueError("A controlled prefix block needs static, slow and fast episodes")
    static = treatments["static"]
    compared_actions = compared_predictions = 0
    max_action_error = max_prediction_error = 0.0
    for condition in ("slow", "fast"):
        candidate = treatments[condition]
        if len(static["actions"]) != len(candidate["actions"]):
            raise ValueError("Episode lengths differ before intervention onset")
        for index, (left, right) in enumerate(zip(static["actions"], candidate["actions"], strict=True)):
            error = max(abs(x - y) for x, y in zip(left, right, strict=True))
            max_action_error = max(max_action_error, error)
            if left != right:
                raise ValueError(f"Pre-motion executed actions differ: {candidate['identity']}, prefix step {index}, maximum error {error}")
            compared_actions += 1
        first, second = static["predictions"], candidate["predictions"]
        if len(first) != len(second):
            raise ValueError("Prediction schedules differ before intervention onset")
        for left, right in zip(first, second, strict=True):
            if left["step"] != right["step"] or left["sampling_seed"] != right["sampling_seed"]:
                raise ValueError("Prediction schedule or seed differs before intervention onset")
            for action_left, action_right in zip(left["actions"], right["actions"], strict=True):
                error = max(abs(x - y) for x, y in zip(action_left, action_right, strict=True))
                max_prediction_error = max(max_prediction_error, error)
            if left["actions"] != right["actions"]:
                raise ValueError(f"Pre-motion full predictions differ: {candidate['identity']}, step {left['step']}, maximum error {max_prediction_error}")
            compared_predictions += 1
    return {"condition_vs_static_comparisons": 2, "compared_action_vectors": compared_actions,
            "compared_prediction_chunks": compared_predictions, "max_action_prefix_error": max_action_error,
            "max_prediction_prefix_error": max_prediction_error}


def parsed_utc(value, label):
    if not isinstance(value, str):
        raise ValueError(f"Missing creation timestamp: {label}")
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError(f"Timestamp lacks a timezone: {label}")
    return timestamp.astimezone(timezone.utc)


def checked_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its frozen calibration directory")
    return path


def verify_frozen(cfg, manifest, frozen_path):
    frozen_path = Path(frozen_path).resolve()
    frozen = json.loads(frozen_path.read_text())
    if (frozen.get("config_fingerprint") != cfg.fingerprint or digest(frozen.get("main_config")) != cfg.fingerprint
            or frozen.get("sources") != manifest["sources"]
            or frozen.get("sources") != source_metadata()
            or frozen.get("baseline_runtime") != manifest["runtime"]
            or frozen.get("planned_episode_ids") != manifest["planned_episode_ids"]):
        raise ValueError("Main run differs from frozen configuration, source code or baseline runtime")
    freeze_time = parsed_utc(frozen.get("created_utc"), "frozen protocol")
    run_time = parsed_utc(manifest.get("created_utc"), "main manifest")
    if freeze_time >= run_time:
        raise ValueError("Protocol was not frozen before the main run was created")
    current_config = PROJECT / "configs/moving-target.json"
    if MovingExperiment.load(current_config).fingerprint != cfg.fingerprint:
        raise ValueError("Current main configuration differs from the registered experiment")
    baseline = hashlib.sha256()
    for path in sorted((PROJECT / "vla_replan").glob("*.py")):
        baseline.update(path.name.encode())
        baseline.update(path.read_bytes())
    if baseline.hexdigest() != manifest["runtime"]["code_sha256"]:
        raise ValueError("The frozen baseline source code changed")
    lock_path = PROJECT / "requirements/linux.lock.txt"
    if sha256(lock_path) != manifest["runtime"]["dependency_lock_sha256"]:
        raise ValueError("The frozen dependency lock changed")
    protocol_path = PROJECT / "docs/moving-target-protocol.md"
    if sha256(protocol_path) != frozen.get("protocol_sha256"):
        raise ValueError("The frozen protocol document changed")
    files = frozen.get("calibration_files", {})
    selected = frozen.get("selected_physics_calibration", "physics-b/calibration.json")
    if selected not in files:
        raise ValueError("Physics-B calibration is missing from the frozen evidence")
    for relative, expected_hash in files.items():
        if sha256(checked_child(frozen_path.parent, relative)) != expected_hash:
            raise ValueError(f"Frozen calibration evidence changed: {relative}")
    model_manifest_relative = frozen.get("model_check_manifest", "model-check/manifest.json")
    if model_manifest_relative not in files:
        raise ValueError("Model-check manifest is missing from the frozen evidence")
    model_manifest = json.loads(checked_child(frozen_path.parent, model_manifest_relative).read_text())
    if model_manifest.get("hardware") != manifest.get("hardware"):
        raise ValueError("Main run hardware differs from the frozen integration checks")
    calibration = json.loads(checked_child(frozen_path.parent, selected).read_text())
    if calibration.get("motion") != cfg.motion.asdict() or calibration.get("state") != 20:
        raise ValueError("Selected physical calibration uses a different intervention or held-out state")
    checks = frozen.get("model_checks", {})
    if checks.get("episodes") != 9 or checks.get("contacts") != 9 or checks.get("all_intervention_and_schedule_checks_passed") is not True:
        raise ValueError("Pre-run model and intervention checks were not completed")
    return {"frozen_protocol_sha256": sha256(frozen_path), "frozen_created_utc": frozen["created_utc"],
            "main_created_utc": manifest["created_utc"], "current_config_sha256": sha256(current_config),
            "protocol_document_sha256": sha256(protocol_path), "baseline_code_sha256": baseline.hexdigest(),
            "dependency_lock_sha256": sha256(lock_path), "physics_b_calibration_sha256": files[selected],
            "calibration_files": files}


def decode_video(path, steps, fps):
    import imageio.v2 as imageio

    frames = 0
    with imageio.get_reader(str(path), input_params=["-threads", "1"]) as reader:
        metadata = reader.get_meta_data()
        if tuple(metadata.get("size", ())) != (768, 576) or not math.isclose(metadata.get("fps", 0), fps, abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"Video dimensions or frame rate differ from the experiment: {path}")
        for frame in reader:
            if frame.shape != (576, 768, 3):
                raise ValueError(f"Decoded frame dimensions changed: {path}")
            frames += 1
    if frames != steps + fps:
        raise ValueError(f"Video has {frames} frames, expected {steps + fps}: {path}")
    return frames


def verify(input_path, frozen_protocol, output_path, *, decode=False):
    root, output, frozen_path = Path(input_path).resolve(), Path(output_path).resolve(), Path(frozen_protocol).resolve()
    if output.is_relative_to(root) or output.is_relative_to(frozen_path.parent):
        raise ValueError("Verification output must be outside raw and frozen calibration directories")
    cfg, validated_rows, provenance = load_run(root)
    base = cfg.base_config
    if (len(validated_rows) != 135 or tuple(sorted(base.initial_states)) != (10, 11, 12, 13, 14)
            or tuple(sorted(base.policy_seeds)) != (11, 22, 33) or tuple(sorted(base.intervals)) != (5, 10, 50)
            or base.control_hz != 20 or base.chunk_size != 50 or not base.record_video):
        raise ValueError("Independent verification requires the complete registered 135-episode plan")
    manifest = json.loads((root / "manifest.json").read_text())
    frozen_evidence = verify_frozen(cfg, manifest, frozen_path)
    prefixes = defaultdict(dict)
    totals = {"verified_episodes": 0, "steps": 0, "model_calls": 0, "predicted_action_values": 0,
              "executed_action_values": 0, "decoded_videos": 0, "decoded_frames": 0}
    max_execution_error = 0.0
    for record in provenance["episodes"]:
        directory = root / "episodes" / record["episode_id"]
        summary = json.loads((directory / "summary.json").read_text())
        rows, chunks = read_jsonl(directory / "steps.jsonl"), read_jsonl(directory / "chunks.jsonl")
        actions = verify_action_log(summary, rows, chunks, base.chunk_size, base.control_hz)
        verify_force_timing(summary, rows, cfg.motion.start_delay_s, cfg.motion.force_application_offset_z_m, cfg.motion.max_force_n)
        prefix = pre_motion_prefix(summary, rows, chunks, cfg.motion.start_delay_s, base.control_hz)
        key = summary["state"], summary["policy_seed"], summary["k"]
        if summary["condition"] in prefixes[key]:
            raise ValueError("Duplicate treatment in a controlled prefix block")
        prefixes[key][summary["condition"]] = prefix
        max_execution_error = max(max_execution_error, actions["max_executed_action_error"])
        for field in ("steps", "model_calls", "predicted_action_values", "executed_action_values"):
            totals[field] += actions[field]
        totals["verified_episodes"] += 1
        if decode:
            totals["decoded_frames"] += decode_video(directory / "video.mp4", summary["steps"], base.control_hz)
            totals["decoded_videos"] += 1
        if totals["verified_episodes"] % 15 == 0:
            print(f"Verified {totals['verified_episodes']}/135 episode action logs" + (" and decoded videos" if decode else ""), flush=True)
    if len(prefixes) != 45:
        raise ValueError("Expected exactly 45 state/seed/K prefix blocks")
    prefix_checks = []
    for (state, policy_seed, k), treatments in sorted(prefixes.items()):
        prefix_checks.append({"state": state, "policy_seed": policy_seed, "k": k, **compare_prefixes(treatments)})
    result = {
        "schema_version": 1, "artifact_kind": "moving_target_evidence_verification", "data_kind": DATA_KIND,
        "verified_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "config_fingerprint": cfg.fingerprint, **totals,
        "prefix_pairs": len(prefix_checks), "prefix_block_definition": "Matched static/slow/fast triplet for one state, policy seed and K",
        "condition_vs_static_prefix_comparisons": sum(item["condition_vs_static_comparisons"] for item in prefix_checks),
        "max_executed_action_error": max_execution_error,
        "max_action_prefix_error": max(item["max_action_prefix_error"] for item in prefix_checks),
        "max_prediction_prefix_error": max(item["max_prediction_prefix_error"] for item in prefix_checks),
        "prefix_action_steps": list(range(math.ceil(cfg.motion.start_delay_s * base.control_hz))),
        "prefix_checks": prefix_checks, "frozen_evidence": frozen_evidence, "provenance": provenance,
        "verifier_source_sha256": sha256(__file__),
        "analysis_source_sha256": sha256(PROJECT / "scripts/analyze_moving_target.py"),
        "video_verification_mode": "decoded_all_frames" if decode else "checksum_only",
        "checks": {"complete_plan_and_raw_checksums": True, "current_sources_match_frozen_run": True,
                   "freeze_precedes_main_creation": True, "physics_b_bound_to_freeze": True,
                   "full_50_by_7_chunks_finite": True, "executed_actions_equal_clipped_prefix": True,
                   "sampling_seeds_and_fixed_k_schedule": True, "success_and_gripper_bookkeeping": True,
                   "zero_force_before_onset_static_and_after_contact": True,
                   "identical_pre_motion_actions_and_full_predictions": True,
                   "all_video_frames_decoded": bool(decode)},
        "limitations": "Log verification checks instrumentation consistency; it cannot independently prove model outputs or physical task success. Finger contact is not a grasp label. Simulator terminal flags are not logged separately from summary termination.",
    }
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "verified_episodes": totals["verified_episodes"], "prefix_pairs": len(prefix_checks), "max_action_prefix_error": result["max_action_prefix_error"], "model_calls": totals["model_calls"], "decoded_videos": totals["decoded_videos"]}, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/moving-target"))
    parser.add_argument("--frozen-protocol", type=Path, default=Path("outputs/moving-calibration/frozen-protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/moving-target-report/evidence-verification.json"))
    parser.add_argument("--decode-video", action="store_true", help="Sequentially decode and count every frame of all 135 raw videos")
    args = parser.parse_args(argv)
    verify(args.input, args.frozen_protocol, args.output, decode=args.decode_video)


if __name__ == "__main__":
    main()
