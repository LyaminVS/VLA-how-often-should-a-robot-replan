#!/usr/bin/env python3
"""Collect and verify registered portable dropout cohorts on one GPU."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, value in {
    "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "MPLBACKEND": "Agg", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "LIBERO_CONFIG_PATH": str(ROOT / ".cache/libero"),
}.items():
    os.environ[key] = value

from action_dropout.config import DropoutExperiment
from action_dropout.core import dropout_mask_sha256, transform_action, uniform_for_step
from vla_replan.storage import atomic_json, exclusive_run, runtime_metadata, write_jsonl

ALL_METHODS = ("fixed_1", "fixed_5", "fixed_10", "fixed_25", "fixed_50",
               "dvac_cap_5", "dvac_cap_10", "dvac_cap_25", "dvac_cap_50")


def cohort(cfg):
    if cfg.methods == ("fixed_1",) and set(cfg.probabilities) == {0.1, 0.3}:
        return "kaggle_fixed_1"
    if set(cfg.methods) == set(ALL_METHODS) and cfg.probabilities == (0.2,):
        return "kaggle_all_methods_p020"
    raise ValueError("Registered portable cohorts are K=1 p=.1,.3 (60 episodes) and all nine methods p=.2 (270 episodes)")


def make_selector(cfg, episode):
    if not episode.adaptive:
        return None
    from capped_replan.selector import CappedDVACSelector
    return CappedDVACSelector(episode.execution_cap, horizon=cfg.base.chunk_size,
        tail_length=cfg.dvac.tail_length, window_size=cfg.dvac.window_size,
        alpha=cfg.dvac.alpha, min_steps=cfg.dvac.min_steps)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def bundle_evidence():
    record = read_json(ROOT / "bundle-manifest.json")
    for name, expected in record["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Bundle source/reference changed: {name}")
    return {"manifest_sha256": sha(ROOT / "bundle-manifest.json"), "files": record["files"]}


def plan(cfg):
    profile = cohort(cfg)
    episodes = cfg.episodes()
    total = 60 if profile == "kaggle_fixed_1" else 270
    if len(episodes) != total or len({episode.pair_id for episode in episodes}) != 30:
        raise ValueError(f"Require exactly 30 matched starts and {total} episodes")
    # Keep shuffled matched-start blocks; defer the expensive K=1 control.
    return sorted(episodes, key=lambda episode: episode.method == "fixed_1")


def verify_manifest(output, cfg, manifest):
    expected = {"config": json.loads(json.dumps(cfg.asdict())), "config_fingerprint": cfg.fingerprint,
                "bundle": bundle_evidence(), "planned_episode_ids": [episode.id for episode in plan(cfg)],
                "home_reference_sha256": sha(ROOT / "references/home-first-predictions.json")}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest configuration, sources or planned episodes differ")
    if manifest.get("experiment_kind") != "kaggle_translation_dropout" or manifest.get("data_kind") != "real_libero":
        raise ValueError("Unexpected portable cohort")
    if sha(output / "invariance.json") != manifest.get("invariance_sha256"):
        raise ValueError("Stored p=0 check was modified")
    invariance = read_json(output / "invariance.json")
    if (invariance.get("passed") is not True or invariance.get("kind") != "kaggle_p0_passthrough"
            or len(invariance.get("checks", [])) != 3 or invariance.get("bundle") != expected["bundle"]
            or invariance.get("runtime") != manifest["runtime"] or invariance.get("hardware") != manifest["hardware"]
            or any(row.get("exact_action_equality") is not True or row.get("exact_observation_equality") is not True
                   for row in invariance["checks"])):
        raise ValueError("Invalid local wrapper invariance record")


class HomeStateGate:
    def __init__(self, environment, references):
        self.environment, self.references = environment, references

    def reset(self, episode):
        observation, info = self.environment.reset(episode)
        if info["initial_state_sha256"] != self.references[episode.pair_id]["initial_state_sha256"]:
            raise ValueError("Physical LIBERO initial-state bytes differ from the home experiment")
        return observation, info

    def step(self, action):
        return self.environment.step(action)


def compare_home(reference, summary, first):
    a, b = first["actions"], reference["actions"]
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b, strict=True)):
        raise ValueError("Home and Kaggle action shapes differ")
    errors = [abs(x - y) for row_a, row_b in zip(a, b, strict=True) for x, y in zip(row_a, row_b, strict=True)]
    within = all(abs(x - y) <= 1e-4 + 1e-4 * abs(y)
                 for row_a, row_b in zip(a, b, strict=True) for x, y in zip(row_a, row_b, strict=True))
    return {"same_initial_state": summary["initial_state_sha256"] == reference["initial_state_sha256"],
            "same_initial_observation": summary["initial_observation_sha256"] == reference["initial_observation_sha256"],
            "same_first_sampling_seed": first["sampling_seed"] == reference["sampling_seed"],
            "first_chunk_exact_equal": a == b, "first_chunk_max_abs_error": max(errors),
            "first_chunk_mean_abs_error": sum(errors) / len(errors),
            "first_chunk_allclose_atol_1e4_rtol_1e4": within,
            "interpretation": "Diagnostic only: cross-GPU images/numerics may differ. The historical p=0 cohort ran on another GPU."}


def local_reference(summary, first):
    return {key: summary[key] for key in ("pair_id", "initial_state_sha256", "initial_observation_sha256")} | {
        "sampling_seed": first["sampling_seed"], "actions": first["actions"]}


def check_passthrough(cfg, policy, environment):
    from action_dropout.config import DropoutEpisode
    from action_dropout.core import run_episode
    from capped_replan.config import CappedEpisode
    from capped_replan.core import run_episode as original_run
    from vla_replan.backend import observation_hash

    class Observed:
        def reset(self, episode):
            obs, info = environment.reset(episode)
            self.hashes = [observation_hash(obs.payload["raw"])]
            return obs, info

        def step(self, action):
            obs, success, terminal = environment.step(action)
            self.hashes.append(observation_hash(obs.payload["raw"]))
            return obs, success, terminal

    observed, checks = Observed(), []
    for task in cfg.base.tasks:
        args = (task.suite, task.name, 21, 101, "fixed_5", 8)
        old, old_rows, old_chunks = original_run(observed, policy, CappedEpisode(*args),
            chunk_size=cfg.base.chunk_size, control_hz=cfg.base.control_hz)
        old_hashes = list(observed.hashes)
        new, rows, chunks = run_episode(observed, policy, DropoutEpisode(*args, 0, cfg.disturbance_seed),
            chunk_size=cfg.base.chunk_size, control_hz=cfg.base.control_hz)
        if ([x["actions"] for x in old_chunks] != [x["actions"] for x in chunks]
                or [x["diagnostics"]["clean_estimates"] for x in old_chunks]
                != [x["diagnostics"]["clean_estimates"] for x in chunks]
                or [r["action"] for r in old_rows] != [r["applied_action"] for r in rows]
                or old_hashes != observed.hashes or old["success"] != new["success"]
                or old["steps"] != new["steps"]):
            raise ValueError("Local p=0 wrapper invariance failed")
        checks.append({"suite": task.suite, "task": task.name, "state": 21, "policy_seed": 101,
                       "exact_action_equality": True, "exact_observation_equality": True,
                       "observation_hashes": old_hashes, "steps_checked": len(rows)})
        print("PASS local p=0 invariance:", task.name, flush=True)
    return {"passed": True, "kind": "kaggle_p0_passthrough", "checks": checks}


def verify_completed(output, cfg, manifest):
    """Verify committed evidence, masks, control commands and paired starts before reuse."""
    from scripts.verify_capped import verify_episode
    verify_manifest(output, cfg, manifest)
    home = read_json(ROOT / "references/home-first-predictions.json")
    manifest_hash = sha(output / "manifest.json")
    expected = {episode.id: episode for episode in plan(cfg)}
    records, references = {}, {}
    for directory in (output / "episodes").glob("*"):
        if not directory.is_dir() or not (directory / "summary.json").is_file():
            raise ValueError(f"Incomplete committed episode directory: {directory}")
    for path in output.glob("episodes/*/summary.json"):
        summary = read_json(path)
        episode = expected.get(path.parent.name)
        if episode is None or summary.get("episode_id") != episode.id:
            raise ValueError(f"Unexpected completed episode {path}")
        if (summary.get("manifest_sha256") != manifest_hash or summary.get("config_fingerprint") != cfg.fingerprint
                or any(summary.get(key) != value for key, value in episode.asdict().items())):
            raise ValueError(f"Episode metadata mismatch: {path}")
        files = summary.get("evidence_sha256", {})
        if set(files) != {"steps.jsonl", "chunks.jsonl", "video.mp4"}:
            raise ValueError("Missing episode evidence")
        for name, digest in files.items():
            if sha(path.parent / name) != digest:
                raise ValueError(f"Evidence checksum mismatch: {path.parent / name}")
        steps = [json.loads(line) for line in (path.parent / "steps.jsonl").read_text().splitlines()]
        chunks = [json.loads(line) for line in (path.parent / "chunks.jsonl").read_text().splitlines()]
        if (len(steps) != summary["steps"] or len(chunks) != summary["model_calls"]
                or not 1 <= len(steps) <= episode.max_steps
                or summary["pair_id"] != episode.pair_id or summary["env_seed"] != episode.env_seed):
            raise ValueError("Step/call count mismatch")
        # Independently reconstruct FM variance, DVAC thresholds/caps, every
        # selected prefix and all commanded actions from the saved evidence.
        verify_episode(summary, steps, chunks, cfg)
        drops, changes = 0, 0
        for index, row in enumerate(steps):
            uniform = uniform_for_step(episode.pair_id, episode.disturbance_seed, index)
            applied, dropped = transform_action(row["action"], uniform, episode.dropout_p)
            command = row["action"]
            if (row["step"] != index
                    or row["commanded_action"] != command or row["dropout_u"] != uniform
                    or type(row["translation_dropped"]) is not bool
                    or row["translation_dropped"] != dropped or row["applied_action"] != applied):
                raise ValueError("Saved policy command or dropout mask is inconsistent")
            drops += int(dropped)
            changes += int(command[:3] != applied[:3])
        if (drops != summary["translation_drop_count"] or changes != summary["translation_changed_count"]
                or summary.get("dropout_mask_length") != episode.max_steps
                or summary.get("dropout_model") != "translation_dropout_v1"
                or summary["dropout_mask_sha256"] != dropout_mask_sha256(
                    episode.pair_id, episode.disturbance_seed, episode.dropout_p, episode.max_steps)):
            raise ValueError("Saved dropout counts or full mask hash are inconsistent")
        if (summary["success"] != steps[-1]["success_after_action"]
                or any(row["success_after_action"] for row in steps[:-1])):
            raise ValueError("Saved success or stopping event is inconsistent")
        comparison = compare_home(home["references"][episode.pair_id], summary, chunks[0])
        if (not comparison["same_initial_state"] or not comparison["same_first_sampling_seed"]
                or summary.get("home_comparison") != comparison):
            raise ValueError("Saved historical comparison differs from evidence")
        records[episode.id] = summary
        current_reference = local_reference(summary, chunks[0])
        if episode.pair_id in references and references[episode.pair_id] != current_reference:
            raise ValueError("Paired Kaggle initial observations or first actions differ")
        references[episode.pair_id] = current_reference
    for summary in records.values():
        ref = references.get(summary["pair_id"])
        if not ref:
            raise ValueError("Missing local first-prediction reference")
        first = json.loads((output / "episodes" / summary["episode_id"] / "chunks.jsonl").read_text().splitlines()[0])
        if local_reference(summary, first) != ref:
            raise ValueError("Paired Kaggle initial observations or first actions differ")
    return records, references


def summarize(output, cfg, records):
    rows = list(records.values())
    if cohort(cfg) == "kaggle_all_methods_p020":
        return summarize_all_methods(output, cfg, records)
    complete_pairs = {row["pair_id"] for row in rows
                      if sum(other["pair_id"] == row["pair_id"] for other in rows) == 2}
    paired = [row for row in rows if row["pair_id"] in complete_pairs]
    result = {"cohort": "kaggle_fixed_1", "completed_episodes": len(rows), "planned_episodes": len(plan(cfg)),
              "complete_matched_starts": len(complete_pairs), "planned_matched_starts": 30,
              "scope": "Only starts completed at both p values enter this table. No new p=0 episodes. Historical home baseline and GPU timings remain separate.",
              "conditions": []}
    for probability in cfg.probabilities:
        selected = [row for row in paired if row["dropout_p"] == probability]
        if not selected:
            continue
        result["conditions"].append({"p": probability, "episodes": len(selected),
            "successes": sum(row["success"] for row in selected),
            "success_rate": sum(row["success"] for row in selected) / len(selected),
            "mean_model_calls": sum(row["model_calls"] for row in selected) / len(selected),
            "mean_episode_wall_s": sum(row["episode_wall_s"] for row in selected) / len(selected),
            "drop_fraction": sum(row["translation_drop_count"] for row in selected) / sum(row["steps"] for row in selected)})
    atomic_json(output / "progress.json", result)
    print(json.dumps(result, indent=2), flush=True)


def summarize_all_methods(output, cfg, records):
    """Only complete 30-start method cohorts enter the comparison table."""
    planned = plan(cfg)
    expected_pairs = {episode.pair_id for episode in planned}
    rows = list(records.values())
    result = {"cohort": cohort(cfg), "completed_episodes": len(rows),
        "planned_episodes": len(planned), "planned_matched_starts": len(expected_pairs),
        "complete_matched_starts": sum(
            {row["method"] for row in rows if row["pair_id"] == pair} == set(cfg.methods)
            for pair in expected_pairs),
        "scope": "Only methods with all 30 matched starts enter conditions. Incomplete methods are inventoried separately. Historical home p=0 and other GPUs remain separate.",
        "conditions": [], "method_inventory": [], "complete_methods": [],
        "missing_episode_ids": [episode.id for episode in planned if episode.id not in records]}
    for method in cfg.methods:
        selected = [row for row in rows if row["method"] == method]
        complete = len(selected) == 30 and {row["pair_id"] for row in selected} == expected_pairs
        result["method_inventory"].append({"method": method, "completed_episodes": len(selected),
            "planned_episodes": 30, "complete": complete})
        if not complete:
            continue
        result["complete_methods"].append(method)
        result["conditions"].append({"method": method, "p": .2, "episodes": len(selected),
            "successes": sum(row["success"] for row in selected),
            "success_rate": sum(row["success"] for row in selected) / len(selected),
            "mean_model_calls": sum(row["model_calls"] for row in selected) / len(selected),
            "mean_episode_wall_s": sum(row["episode_wall_s"] for row in selected) / len(selected),
            "drop_fraction": sum(row["translation_drop_count"] for row in selected) / sum(row["steps"] for row in selected)})
    atomic_json(output / "progress.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "missing_episode_ids"}, indent=2), flush=True)


def export_zip(output, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for name in ("manifest.json", "invariance.json", "progress.json"):
            if (output / name).exists():
                archive.write(output / name, f"{output.name}/{name}")
        for summary in sorted(output.glob("episodes/*/summary.json")):
            for path in sorted(summary.parent.iterdir()):
                if path.is_file():
                    archive.write(path, output.name + "/" + str(path.relative_to(output)))
    temporary.replace(destination)


def run(args):
    from adaptive_replan.policy import InstrumentedSmolVLAPolicy
    from action_dropout.core import run_episode
    from action_dropout.video import EpisodeVideo
    from vla_replan.backend import LiberoEnvironment
    cfg = DropoutExperiment.load(args.config)
    episodes, output = plan(cfg), Path(args.output)
    home = read_json(ROOT / "references/home-first-predictions.json")
    source_base = home["manifest"]["config"]["base"]
    base = json.loads(json.dumps(cfg.base.asdict()))
    if any(base[key] != value for key, value in source_base.items() if key != "name"):
        raise ValueError("Kaggle policy, simulator or matched-start configuration differs from home")
    if json.loads(json.dumps(cfg.asdict()["dvac"])) != home["manifest"]["config"]["dvac"]:
        raise ValueError("Kaggle DVAC settings differ from the frozen home comparison")
    sources = bundle_evidence()
    with exclusive_run(output):
        policy = InstrumentedSmolVLAPolicy(cfg.base, tail_length=cfg.dvac.tail_length)
        metadata = runtime_metadata()
        old = home["manifest"]
        # Pin actual policy/simulator versions, solver implementation and assets;
        # retain hardware/OS provenance without requiring an RTX2060 on Kaggle.
        if (metadata["packages"] != old["runtime"]["packages"]
                or metadata["simulator_assets"] != old["runtime"]["simulator_assets"]
                or policy.sampler_manifest() != old["upstream_sampler"]):
            raise ValueError("Pinned policy/simulator packages, assets or FM sampler differ")
        stable = {"config": json.loads(json.dumps(cfg.asdict())), "config_fingerprint": cfg.fingerprint,
            "bundle": sources, "runtime": metadata, "hardware": policy.runtime,
            "upstream_sampler": policy.sampler_manifest(), "home_reference_sha256": sha(ROOT / "references/home-first-predictions.json"),
            "planned_episode_ids": [episode.id for episode in episodes]}
        manifest_path = output / "manifest.json"
        environment = LiberoEnvironment(cfg.base)
        try:
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                if any(manifest.get(key) != value for key, value in stable.items()):
                    raise ValueError("Resume requires unchanged sources, configuration, packages and GPU type")
                if sha(output / "invariance.json") != manifest["invariance_sha256"]:
                    raise ValueError("Stored p=0 check was modified")
            else:
                invariance = check_passthrough(cfg, policy, environment)
                invariance.update(runtime=metadata, hardware=policy.runtime, bundle=sources)
                atomic_json(output / "invariance.json", invariance)
                manifest = {**stable, "schema_version": 1, "experiment_kind": "kaggle_translation_dropout",
                    "data_kind": "real_libero", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "invariance_sha256": sha(output / "invariance.json"),
                    "comparison": "Separate GPU cohort: " + cohort(cfg) + ". Historical p=0 is reused from home; cross-hardware comparability is limited. Do not pool timings or infer cross-hardware equality.",
                    "collection_order": "Matched starts shuffled, then method/probability variants shuffled within each start; K=1 last.",
                    "intervention": "Zero clipped XYZ with probability p; preserve rotation/gripper. Settling is unperturbed."}
                atomic_json(manifest_path, manifest)
            records, references = verify_completed(output, cfg, manifest)
            remaining = [episode for episode in episodes if episode.id not in records]
            if remaining:
                observation, _ = environment.reset(remaining[0])
                for i in range(cfg.base.warmup_calls):
                    policy.reset()
                    policy.predict(observation, i)
            deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None
            manifest_hash = sha(manifest_path)
            for number, episode in enumerate(remaining):
                if (args.limit and number >= args.limit) or (deadline and time.monotonic() >= deadline):
                    break
                attempt = output / "attempts" / f"{episode.id}__{time.time_ns()}"
                attempt.mkdir(parents=True)
                video = EpisodeVideo(attempt / "video.mp4", episode, cfg.base.control_hz)
                started = time.perf_counter()
                try:
                    policy.reset_peak_memory()
                    summary, rows, chunks = run_episode(HomeStateGate(environment, home["references"]), policy, episode,
                        selector=make_selector(cfg, episode),
                        reference=references.get(episode.pair_id),
                        chunk_size=cfg.base.chunk_size, control_hz=cfg.base.control_hz, record=video)
                    video.close()
                    comparison = compare_home(home["references"][episode.pair_id], summary, chunks[0])
                    if not comparison["same_first_sampling_seed"]:
                        raise ValueError("First sampling seed differs from home")
                    summary.update(config_fingerprint=cfg.fingerprint, manifest_sha256=manifest_hash,
                        data_kind="real_libero", collection_cohort=cohort(cfg), home_comparison=comparison,
                        episode_wall_s=time.perf_counter() - started, peak_gpu_memory_bytes=policy.peak_memory())
                    write_jsonl(attempt / "steps.jsonl", rows)
                    write_jsonl(attempt / "chunks.jsonl", chunks)
                    summary["evidence_sha256"] = {name: sha(attempt / name) for name in ("steps.jsonl", "chunks.jsonl", "video.mp4")}
                    atomic_json(attempt / "summary.json", summary)
                    destination = output / "episodes" / episode.id
                    destination.parent.mkdir(exist_ok=True)
                    attempt.rename(destination)
                    records[episode.id] = summary
                    references[episode.pair_id] = local_reference(summary, chunks[0])
                    print(f"[{len(records)}/{len(episodes)}] p={episode.dropout_p} {episode.method} {episode.task} state={episode.state} "
                          f"success={summary['success']} calls={summary['model_calls']} wall={summary['episode_wall_s']:.1f}s", flush=True)
                    summarize(output, cfg, records)
                    if len(records) % args.export_every == 0:
                        export_zip(output, Path(args.archive))
                except BaseException:
                    atomic_json(attempt / "error.json", {"traceback": traceback.format_exc(), "episode": episode.asdict()})
                    raise
                finally:
                    video.close()
            verify_completed(output, cfg, manifest)
            summarize(output, cfg, records)
        finally:
            environment.close()
            if manifest_path.exists():
                export_zip(output, Path(args.archive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/action-dropout-kaggle.json"))
    parser.add_argument("--output", default="/kaggle/working/action-dropout-kaggle")
    parser.add_argument("--archive", default="/kaggle/working/action-dropout-kaggle-results.zip")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-hours", type=float, default=10)
    parser.add_argument("--export-every", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.export_every < 1 or (args.limit is not None and args.limit < 1) or args.max_hours < 0:
        parser.error("Bounds must be positive; --max-hours 0 disables the time bound")
    if args.verify_only:
        cfg = DropoutExperiment.load(args.config)
        output = Path(args.output)
        records, _ = verify_completed(output, cfg, read_json(output / "manifest.json"))
        summarize(output, cfg, records)
        export_zip(output, Path(args.archive))
    else:
        run(args)


if __name__ == "__main__":
    main()
