"""Verify commanded and actually applied translations under deterministic dropout.

The frozen capped verifier replays policy commands and DVAC decisions. This
module independently replays the disturbance stream and checks applied actions.
Logged success is the environment outcome after those applied actions; it is
not evidence that the unmodified command was executed. Partial collection is
reported explicitly and is never relabeled a complete balanced experiment.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.verify_capped import first_prediction, verify_episode as verify_commanded_episode
from scripts.verify_dvac import read_jsonl, sha256, vector, verify_video
from vla_replan.storage import atomic_json

MODEL = "translation_dropout_v1"


def reference_uniform(pair_id, disturbance_seed, step):
    """Independent specification replay; never call the collection's RNG helper."""
    if (not isinstance(pair_id, str) or not pair_id or type(disturbance_seed) is not int
            or disturbance_seed < 0 or type(step) is not int or step < 0):
        raise ValueError("Invalid deterministic dropout schedule identity")
    payload = json.dumps([MODEL, pair_id, disturbance_seed, step],
                         separators=(",", ":"), ensure_ascii=True).encode("utf8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
    return integer / 2**53


def reference_mask_sha256(pair_id, seed, probability, max_steps):
    return hashlib.sha256(bytes(int(reference_uniform(pair_id, seed, step) < probability)
                                for step in range(max_steps))).hexdigest()


def verify_applied_actions(summary, rows):
    probability = summary.get("dropout_p")
    seed = summary.get("disturbance_seed")
    maximum = summary.get("max_steps")
    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid dropout probability")
    if type(seed) is not int or seed < 0 or type(maximum) is not int or maximum < 1:
        raise ValueError("Invalid disturbance seed or episode limit")
    if summary.get("dropout_model") != MODEL:
        raise ValueError("Unknown action disturbance model")
    if not rows or len(rows) != summary.get("steps") or [row.get("step") for row in rows] != list(range(len(rows))):
        raise ValueError("Applied-action evidence must cover every executed step")
    expected_hash = reference_mask_sha256(summary["pair_id"], seed, probability, maximum)
    if summary.get("dropout_mask_sha256") != expected_hash:
        raise ValueError("Full-horizon dropout mask checksum differs")
    if type(summary.get("dropout_mask_length")) is not int or summary["dropout_mask_length"] != maximum:
        raise ValueError("Dropout mask must cover the full configured action limit")
    count = changed = 0
    for row in rows:
        step = row["step"]
        uniform = reference_uniform(summary["pair_id"], seed, step)
        value = row.get("dropout_u")
        if type(value) not in (int, float) or not math.isfinite(value) or value != uniform:
            raise ValueError(f"Dropout uniform differs from the step-indexed shared schedule at step {step}")
        dropped = uniform < probability
        if row.get("translation_dropped") is not dropped:
            raise ValueError(f"Translation-drop flag disagrees with u < p at step {step}")
        command = vector(row.get("action"), 7, "commanded policy action")
        if vector(row.get("commanded_action"), 7, "explicit commanded action") != command:
            raise ValueError("Commanded-action alias disagrees with the frozen executor's command")
        applied = vector(row.get("applied_action"), 7, "applied simulator action")
        expected = ([0.0, 0.0, 0.0] + command[3:]) if dropped else command
        if applied != expected:
            raise ValueError(f"Applied action does not match translation-only dropout at step {step}")
        count += int(dropped)
        changed += int(applied[:3] != command[:3])
    if type(summary.get("translation_drop_count")) is not int or summary["translation_drop_count"] != count:
        raise ValueError("Summary dropout count differs from applied-step evidence")
    if type(summary.get("translation_changed_count")) is not int or summary["translation_changed_count"] != changed:
        raise ValueError("Summary changed-translation count differs from applied-step evidence")
    return {"translation_drop_count": count, "translation_changed_steps": changed,
            "verified_applied_action_values": len(rows) * 7, "dropout_mask_sha256": expected_hash,
            "uniform_schedule_replayed": True, "rotation_and_gripper_preserved": True,
            "outcome_interpretation": "Success flags are logged environment outcomes after applied actions; no simulator replay was performed"}


def _verify_sources(recorded, *, check_sources):
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Missing action-dropout collection source hashes")
    if check_sources:
        for relative, checksum in recorded.items():
            path = (PROJECT / relative).resolve()
            if not path.is_relative_to(PROJECT) or not path.is_file() or sha256(path) != checksum:
                raise ValueError(f"Collection source checksum mismatch: {relative}")


def inventory_report(cfg, completed):
    """Report missing settings and balance against the complete frozen plan."""
    episodes = cfg.episodes()
    expected = {episode.id: episode for episode in episodes}
    if len(expected) != len(episodes) or not set(completed).issubset(expected):
        raise ValueError("Duplicate or unexpected dropout episode inventory")
    planned = Counter((episode.dropout_p, episode.method) for episode in episodes)
    observed = Counter((expected[identity].dropout_p, expected[identity].method) for identity in completed)
    return {
        "complete": set(completed) == set(expected), "completed_new_episodes": len(completed),
        "planned_new_episodes": len(expected),
        "missing_episode_ids": [identity for identity in expected if identity not in completed],
        "coverage": [{"dropout_p": probability, "method": method, "completed": observed[(probability, method)],
                      "planned": count, "complete": observed[(probability, method)] == count}
                     for (probability, method), count in sorted(planned.items())],
    }


def registered_plan(cfg):
    """Keep all nine methods and 30 starts while honoring configured probabilities."""
    from action_dropout.config import ALL_METHODS

    cfg.validate()
    ordered = sorted(cfg.episodes(), key=lambda episode: episode.method == "fixed_1")
    expected = {episode.id: episode for episode in ordered}
    expected_count = 30 * len(ALL_METHODS) * len(cfg.probabilities)
    if (set(cfg.methods) != set(ALL_METHODS) or any(p <= 0 for p in cfg.probabilities)
            or len(expected) != expected_count
            or len({episode.pair_id for episode in expected.values()}) != 30):
        raise ValueError("The registered plan must retain nine methods, 30 starts and all configured positive probabilities, even for a partial collection")
    return expected


def load_and_verify(root, *, baseline_root=None, reference_roots=None, check_sources=True, decode_videos=False):
    from action_dropout.config import DropoutExperiment
    from scripts.verify_capped import MATCH_FIELDS, load_and_verify as verify_baseline

    root = Path(root).resolve()
    baseline_root = Path(baseline_root or PROJECT / "outputs/capped").resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = DropoutExperiment.fromdict(manifest["config"])
    expected = registered_plan(cfg)
    if (manifest.get("config_fingerprint") != cfg.fingerprint or manifest.get("data_kind") != "real_libero"
            or manifest.get("experiment_kind") != "translation_dropout"
            or manifest.get("planned_episode_ids") != list(expected)
            or manifest.get("full_plan_episode_ids") != list(expected)):
        raise ValueError("Invalid or reduced action-dropout experiment manifest")
    _verify_sources(manifest.get("source_manifest"), check_sources=check_sources)
    base_cfg, base_records, baseline_audit = verify_baseline(
        baseline_root, decode_videos=decode_videos, check_sources=check_sources, reference_roots=reference_roots)
    if baseline_audit.get("complete") is not True or len(base_records) != 270:
        raise ValueError("The retained p=0 baseline must be the complete verified 270-episode comparison")
    original_manifest = json.loads((baseline_root / "manifest.json").read_text())
    new_base = json.loads(json.dumps(cfg.base.asdict()))
    old_base = json.loads(json.dumps(base_cfg.base.asdict()))
    if any(new_base[field] != old_base[field] for field in (*MATCH_FIELDS, "tasks", "initial_states", "policy_seeds")):
        raise ValueError("Dropout collection does not preserve baseline model, task or starting conditions")
    for field in ("runtime", "hardware", "upstream_sampler"):
        if manifest.get(field) != original_manifest.get(field):
            raise ValueError(f"Dropout and retained adaptive baseline differ in {field}")
    audit = manifest.get("baseline_sources", {})
    declared = audit.get("files", {})
    reference_pairs = {}
    expected_reference_files = {"manifest.json", "invariance.json"}
    for record in base_records:
        summary = record["summary"]
        if record["origin"] != "capped" or summary["method"] != "dvac_cap_50":
            continue
        directory = baseline_root / "episodes" / summary["episode_id"]
        with (directory / "chunks.jsonl").open() as stream:
            first = json.loads(next(stream))
        reference_pairs[summary["pair_id"]] = {"summary": summary, "first_prediction": first_prediction(first)}
        expected_reference_files.update(str((directory / name).relative_to(baseline_root))
                                        for name in ("summary.json", "chunks.jsonl", "steps.jsonl", "video.mp4"))
    if set(declared) != expected_reference_files or len(reference_pairs) != 30:
        raise ValueError("Missing or unexpected immutable baseline reference inventory")
    for relative, checksum in declared.items():
        path = (baseline_root / relative).resolve()
        if not path.is_relative_to(baseline_root) or not path.is_file() or sha256(path) != checksum:
            raise ValueError(f"Baseline reference checksum differs: {relative}")
    check_path = root / "invariance.json"
    if sha256(check_path) != manifest.get("invariance_check_sha256"):
        raise ValueError("Passthrough check checksum differs")
    check = json.loads(check_path.read_text())
    tasks = {(task.suite, task.name) for task in cfg.base.tasks}
    checks = check.get("checks", [])
    if (check.get("passed") is not True or check.get("check_kind") != "action_dropout_passthrough"
            or check.get("config_fingerprint") != cfg.fingerprint
            or check.get("source_manifest") != manifest["source_manifest"]
            or check.get("runtime") != manifest["runtime"] or check.get("hardware") != manifest["hardware"]
            or check.get("sampler_manifest") != manifest["upstream_sampler"]
            or check.get("baseline_sources") != audit or len(checks) != len(tasks)
            or {(row.get("suite"), row.get("task")) for row in checks} != tasks
            or any(row.get("exact_action_equality") is not True or row.get("max_absolute_action_error") != 0
                   or row.get("exact_observation_equality") is not True or row.get("steps_checked") != 8
                   for row in checks)):
        raise ValueError("p=0 wrapper passthrough check does not match this collection")
    episode_root = root / "episodes"
    actual = {path.name for path in episode_root.iterdir() if path.is_dir()} if episode_root.exists() else set()
    inventory = inventory_report(cfg, actual)
    records = []
    for record in base_records:
        summary = {**record["summary"], "dropout_p": 0.0, "disturbance_seed": None,
                   "translation_drop_count": 0, "translation_changed_count": 0,
                   "dropout_model": "none_retained_baseline", "dropout_baseline_reused": True}
        records.append({**record, "summary": summary, "origin": "reused_p0"})
    manifest_sha = sha256(manifest_path)
    evidence = []
    for identity, episode in expected.items():
        if identity not in actual:
            continue
        directory = episode_root / identity
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        if (summary.get("episode_id") != identity or summary.get("pair_id") != episode.pair_id
                or any(summary.get(key) != value or type(summary.get(key)) is not type(value)
                       for key, value in episode.asdict().items())
                or summary.get("config_fingerprint") != cfg.fingerprint
                or summary.get("manifest_sha256") != manifest_sha
                or summary.get("data_kind") != "real_libero" or summary.get("env_seed") != episode.env_seed):
            raise ValueError(f"Dropout episode identity differs: {identity}")
        reference = reference_pairs[episode.pair_id]
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            if summary.get(field) != reference["summary"][field]:
                raise ValueError("Dropout initial state or observation differs from the retained baseline")
        if (summary.get("reference_pair_id") != episode.pair_id
                or summary.get("reference_initial_state_match") is not True
                or summary.get("reference_first_prediction_equal") is not True
                or summary.get("reference_max_absolute_action_error") != 0):
            raise ValueError("Missing per-episode baseline gate evidence")
        hashes = summary.get("evidence_sha256", {})
        filenames = {"steps.jsonl", "chunks.jsonl", "video.mp4"}
        if set(hashes) != filenames:
            raise ValueError("Dropout evidence checksum inventory differs")
        for name, checksum in hashes.items():
            path = directory / name
            if not path.is_file() or not path.stat().st_size or sha256(path) != checksum:
                raise ValueError(f"Dropout evidence checksum differs: {path}")
        rows = read_jsonl(directory / "steps.jsonl")
        chunks = read_jsonl(directory / "chunks.jsonl")
        compact, command_result = verify_commanded_episode(summary, rows, chunks, cfg)
        applied_result = verify_applied_actions(summary, rows)
        if first_prediction(chunks[0]) != reference["first_prediction"]:
            raise ValueError("First full prediction differs from the retained baseline")
        command_result["verified_commanded_action_values"] = command_result.pop("executed_action_values")
        item = {"episode_id": identity, "summary_sha256": sha256(summary_path), "files": hashes,
                **command_result, **applied_result}
        if decode_videos:
            item["video"] = verify_video(directory / "video.mp4", summary["steps"], cfg.base.control_hz)
        evidence.append(item)
        records.append({"summary": summary, "chunks": compact, "origin": "dropout", "raw_source_root": str(root)})
    return cfg, records, {
        **inventory, "passed": True, "episodes": len(records), "baseline_episodes": 270,
        "dropout_episodes": len(actual), "manifest_sha256": manifest_sha,
        "config_fingerprint": cfg.fingerprint, "invariance_sha256": sha256(check_path),
        "verifier_source_sha256": sha256(__file__), "collection_source_checks": check_sources,
        "videos_decoded": decode_videos, "baseline_verification": baseline_audit, "evidence": evidence,
        "interpretation": "Verification covers the observed inventory; complete=false does not license treating unfinished cells as failures or pooling unbalanced panels",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline-root", type=Path, default=PROJECT / "outputs/capped")
    parser.add_argument("--reference-roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-videos", action="store_true")
    args = parser.parse_args()
    _, _, audit = load_and_verify(args.input, baseline_root=args.baseline_root,
                                 reference_roots=args.reference_roots, decode_videos=args.decode_videos)
    if args.output:
        protected = [args.input, args.baseline_root, *(args.reference_roots or [PROJECT / "outputs/compact", PROJECT / "outputs/validation"])]
        if any(args.output.resolve().is_relative_to(path.resolve()) for path in protected):
            raise ValueError("Verification output must be outside immutable evidence")
        atomic_json(args.output, audit)
    print(json.dumps({key: value for key, value in audit.items()
                      if key not in ("evidence", "baseline_verification", "missing_episode_ids")}, indent=2))


if __name__ == "__main__":
    main()
