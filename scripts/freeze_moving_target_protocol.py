"""Validate calibration and integration evidence, then create a new pre-run record.

Run with the measured rollout environment before creating the main output directory.
Existing records are never overwritten. This utility does not select parameters by
task success; the supplied physical controller and its checks must already be fixed.
"""

import argparse
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
from scripts.analyze_moving_target import load_run, read_jsonl, sha256
from scripts.verify_moving_target import compare_prefixes, pre_motion_prefix, verify_action_log, verify_force_timing
from vla_replan.storage import runtime_metadata


def ensure_fresh_destination(output, main_output):
    if Path(output).exists():
        raise FileExistsError(f"Frozen protocol already exists and will not be replaced: {output}")
    if Path(main_output).exists():
        raise ValueError("Freeze the protocol before creating the intended main output directory")


def register_file(path, root, hashes):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Calibration inputs must lie inside the frozen record's parent directory")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty calibration file: {path}")
    relative = str(path.relative_to(root))
    hashes[relative] = sha256(path)
    return relative


def validate_physics(cfg, physics, evidence_root, hashes):
    physics = Path(physics).resolve()
    path = physics / "calibration.json"
    selected = register_file(path, evidence_root, hashes)
    calibration = json.loads(path.read_text())
    if (calibration.get("motion") != cfg.motion.asdict() or calibration.get("state") != 20
            or calibration.get("steps") != 240 or set(calibration.get("results", {})) != set(CONDITIONS)):
        raise ValueError("Selected physics calibration must use the main intervention on excluded state 20 for 240 steps")
    starts = set()
    for condition in CONDITIONS:
        record = calibration["results"][condition]
        reset, final = record["reset"], record["final"]
        starts.add((reset["initial_state_sha256"], reset["initial_observation_sha256"]))
        if final.get("condition") != condition or final.get("first_contact_step") is not None or final.get("first_contact_sim_time") is not None:
            raise ValueError("Physics calibration must have no robot finger contact")
        for name in ("peak_force_n", "precontact_x_range_m", "precontact_y_range_m", "precontact_z_range_m", "max_precontact_tilt_deg"):
            value = final.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid physical calibration diagnostic: {condition}/{name}")
        if final["peak_force_n"] > cfg.motion.max_force_n + 1e-9:
            raise ValueError("Calibration exceeded the intervention force limit")
        if condition == "static":
            if final.get("active_physics_samples") != 0 or final["peak_force_n"] != 0:
                raise ValueError("Static physical calibration contains active intervention")
        elif final.get("active_physics_samples", 0) <= 0 or final["precontact_x_range_m"] <= 0:
            raise ValueError("Moving physical calibration produced no measured movement")
        log_path = physics / f"{condition}.jsonl"
        register_file(log_path, evidence_root, hashes)
        register_file(physics / f"{condition}.mp4", evidence_root, hashes)
        records = read_jsonl(log_path)
        if [item.get("step") for item in records] != list(range(241)):
            raise ValueError("Physical calibration trajectory is incomplete")
        if {key: value for key, value in records[-1].items() if key != "step"} != final:
            raise ValueError("Calibration final diagnostics differ from the trajectory log")
        summary = {"condition": condition, "sim_time_s": 12.0, "motion_final": final}
        rows = [{"sim_time_s": item["step"] / cfg.base_config.control_hz, "motion": item} for item in records[:-1]]
        verify_force_timing(summary, rows, cfg.motion.start_delay_s, cfg.motion.force_application_offset_z_m, cfg.motion.max_force_n)
    if len(starts) != 1:
        raise ValueError("Physical calibration conditions did not share the same initial observation")
    return selected


def validate_model_checks(cfg, check_path, evidence_root, hashes, current_runtime):
    check_path = Path(check_path).resolve()
    check_cfg, rows, provenance = load_run(check_path)
    expected_base = cfg.base_config.asdict()
    actual_base = check_cfg.base_config.asdict()
    for name in ("name", "initial_states", "policy_seeds"):
        expected_base.pop(name)
        actual_base.pop(name)
    if (len(rows) != 9 or tuple(check_cfg.base_config.initial_states) != (20,)
            or tuple(check_cfg.base_config.policy_seeds) != (101,) or expected_base != actual_base
            or check_cfg.motion.asdict() != cfg.motion.asdict()):
        raise ValueError("Integration checks must cover the same fixed controller/model with state 20, seed 101, all three K values and conditions")
    if provenance["sources"] != source_metadata() or provenance["runtime"] != current_runtime:
        raise ValueError("Integration checks do not match the current source code and rollout environment")
    register_file(check_path / "manifest.json", evidence_root, hashes)
    prefixes = defaultdict(dict)
    contacts = 0
    for record in provenance["episodes"]:
        directory = check_path / "episodes" / record["episode_id"]
        for filename in (*record["files"], "evidence.json"):
            register_file(directory / filename, evidence_root, hashes)
        summary = json.loads((directory / "summary.json").read_text())
        steps, chunks = read_jsonl(directory / "steps.jsonl"), read_jsonl(directory / "chunks.jsonl")
        verify_action_log(summary, steps, chunks, cfg.base_config.chunk_size, cfg.base_config.control_hz)
        verify_force_timing(summary, steps, cfg.motion.start_delay_s, cfg.motion.force_application_offset_z_m, cfg.motion.max_force_n)
        prefixes[summary["k"]][summary["condition"]] = pre_motion_prefix(summary, steps, chunks, cfg.motion.start_delay_s, cfg.base_config.control_hz)
        contacts += summary["motion_final"]["first_contact_step"] is not None
    if contacts != 9:
        raise ValueError("All nine integration checks must exercise contact-triggered release")
    for treatments in prefixes.values():
        compare_prefixes(treatments)
    return {"episodes": 9, "contacts": contacts, "all_intervention_and_schedule_checks_passed": True,
            "controlled_prefix_blocks": len(prefixes)}, provenance


def freeze(args):
    output, main_output = args.output.resolve(), args.main_output.resolve()
    ensure_fresh_destination(output, main_output)
    cfg = MovingExperiment.load(args.config)
    base = cfg.base_config
    if (tuple(sorted(base.initial_states)) != (10, 11, 12, 13, 14)
            or tuple(sorted(base.policy_seeds)) != (11, 22, 33)
            or tuple(sorted(base.intervals)) != (5, 10, 50) or len(cfg.episodes()) != 135):
        raise ValueError("This freeze utility registers the five-state, three-seed, 135-episode main protocol")
    hashes = {}
    selected = validate_physics(cfg, args.physics, output.parent, hashes)
    metadata = runtime_metadata()
    checks, provenance = validate_model_checks(cfg, args.model_checks, output.parent, hashes, metadata)
    for directory in args.additional_physics:
        directory = directory.resolve()
        for filename in ("calibration.json", *(f"{condition}.{suffix}" for condition in CONDITIONS for suffix in ("jsonl", "mp4"))):
            register_file(directory / filename, output.parent, hashes)
    protocol_hash = sha256(args.protocol)
    ensure_fresh_destination(output, main_output)
    result = {
        "schema_version": 1,
        "role": "Fixed motion protocol and validated implementation before the main evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_config": cfg.asdict(), "config_fingerprint": cfg.fingerprint,
        "planned_episode_ids": [episode.id for episode in cfg.episodes()],
        "sources": provenance["sources"], "baseline_runtime": metadata,
        "protocol_sha256": protocol_hash, "calibration_files": hashes,
        "selected_physics_calibration": selected, "model_checks": checks,
        "model_check_manifest": str((args.model_checks.resolve() / "manifest.json").relative_to(output.parent)),
        "hardware": provenance["hardware"],
        "intended_main_output": str(main_output),
        "parameter_selection": "Controller fixed before the main collection; physical calibration and integration checks are excluded from task-completion results.",
        "freeze_generator_sha256": sha256(__file__),
        "verification_source_sha256": sha256(PROJECT / "scripts/verify_moving_target.py"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "planned_episodes": len(cfg.episodes()), "config_fingerprint": cfg.fingerprint,
                      "calibration_files": len(hashes), "created_utc": result["created_utc"]}, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/moving-target.json"))
    parser.add_argument("--physics", type=Path, default=Path("outputs/moving-calibration/physics-b"))
    parser.add_argument("--model-checks", type=Path, default=Path("outputs/moving-calibration/model-check"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/moving-target-protocol.md"))
    parser.add_argument("--main-output", type=Path, default=Path("outputs/moving-target"))
    parser.add_argument("--output", type=Path, default=Path("outputs/moving-calibration/frozen-protocol.json"))
    parser.add_argument("--additional-physics", action="append", type=Path, default=[], help="Also bind a retained rejected calibration directory")
    freeze(parser.parse_args(argv))


if __name__ == "__main__":
    main()
