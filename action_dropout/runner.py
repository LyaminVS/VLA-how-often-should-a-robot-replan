"""Bounded, sequential dropout collection with resumable immutable evidence."""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from vla_replan.storage import atomic_json, exclusive_run, git_metadata, runtime_metadata, write_jsonl
from .integrity import file_sha256, source_manifest
from .reference import load_references, validate_runtime


def check(cfg, output, baseline_root):
    from adaptive_replan.policy import InstrumentedSmolVLAPolicy
    from capped_replan.config import CappedEpisode
    from capped_replan.core import run_episode as original_run
    from vla_replan.backend import LiberoEnvironment, observation_hash
    from .config import DropoutEpisode
    from .core import run_episode

    class ObservedEnvironment:
        def __init__(self, env):
            self.env, self.observations = env, []

        def reset(self, episode):
            obs, info = self.env.reset(episode)
            self.observations = [observation_hash(obs.payload["raw"])]
            return obs, info

        def step(self, action):
            obs, success, terminal = self.env.step(action)
            self.observations.append(observation_hash(obs.payload["raw"]))
            return obs, success, terminal

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _, audit, old = load_references(baseline_root, cfg)
    sources, metadata = source_manifest(), runtime_metadata()
    policy = InstrumentedSmolVLAPolicy(cfg.base, tail_length=cfg.dvac.tail_length)
    validate_runtime(old, metadata, policy)
    env = LiberoEnvironment(cfg.base)
    observer = ObservedEnvironment(env)
    checks = []
    try:
        for task in cfg.base.tasks:
            plain_episode = CappedEpisode(task.suite, task.name, 21, 101, "fixed_5", 8)
            dropped_episode = DropoutEpisode(task.suite, task.name, 21, 101, "fixed_5", 8,
                                             dropout_p=0.0, disturbance_seed=cfg.disturbance_seed)
            plain, plain_rows, plain_chunks = original_run(observer, policy, plain_episode,
                                                           chunk_size=cfg.base.chunk_size, control_hz=cfg.base.control_hz)
            plain_hashes = list(observer.observations)
            observed, rows, chunks = run_episode(observer, policy, dropped_episode,
                                                 chunk_size=cfg.base.chunk_size, control_hz=cfg.base.control_hz)
            if ([chunk["actions"] for chunk in chunks] != [chunk["actions"] for chunk in plain_chunks]
                    or [chunk["diagnostics"]["clean_estimates"] for chunk in chunks]
                    != [chunk["diagnostics"]["clean_estimates"] for chunk in plain_chunks]
                    or [row["action"] for row in rows] != [row["action"] for row in plain_rows]
                    or any(row["applied_action"] != row["action"] for row in rows)
                    or observer.observations != plain_hashes
                    or observed["success"] != plain["success"] or observed["steps"] != plain["steps"]):
                raise ValueError("Zero-dropout wrapper changed original inference or simulation")
            checks.append({"suite": task.suite, "task": task.name, "initial_state": 21,
                           "policy_seed": 101, "sampling_seed": plain_chunks[0]["sampling_seed"], "steps_checked": len(rows),
                           "exact_action_equality": True, "max_absolute_action_error": 0.0,
                           "exact_observation_equality": True, "observation_hashes": plain_hashes})
            print(f"PASS {task.name}: p=0 matches all actions and {len(plain_hashes)} observations", flush=True)
    finally:
        env.close()
    if source_manifest() != sources:
        raise ValueError("Collection sources changed during invariance check")
    atomic_json(output, {"passed": True, "check_kind": "action_dropout_passthrough",
                        "checks": checks, "config": cfg.asdict(), "config_fingerprint": cfg.fingerprint,
                        "source_manifest": sources, "runtime": metadata, "hardware": policy.runtime,
                        "sampler_manifest": policy.sampler_manifest(), "baseline_sources": audit})


def run(cfg, output, integrity_check, baseline_root, *, methods=None, limit=None, deadline=None):
    from adaptive_replan.policy import InstrumentedSmolVLAPolicy
    from capped_replan.selector import CappedDVACSelector
    from vla_replan.backend import LiberoEnvironment
    from .core import run_episode
    from .video import EpisodeVideo

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Limit must be a positive integer")
    if methods is not None and (not methods or not set(methods).issubset(cfg.methods)):
        raise ValueError("Unknown selected methods")
    stop_time = datetime.fromisoformat(deadline).timestamp() if deadline else None
    if stop_time is not None and datetime.fromisoformat(deadline).tzinfo is None:
        raise ValueError("Deadline requires an explicit UTC offset")
    # Preserve complete balanced comparisons before the particularly expensive K=1.
    episodes = sorted(cfg.episodes(), key=lambda episode: episode.method == "fixed_1")
    references, audit, old = load_references(baseline_root, cfg)
    output, check_path = Path(output), Path(integrity_check)
    with exclusive_run(output):
        sources, metadata = source_manifest(), runtime_metadata()
        evidence = json.loads(check_path.read_text())
        expected_tasks = {(task.suite, task.name) for task in cfg.base.tasks}
        if (evidence.get("passed") is not True or evidence.get("check_kind") != "action_dropout_passthrough"
                or evidence.get("source_manifest") != sources
                or evidence.get("config_fingerprint") != cfg.fingerprint
                or evidence.get("runtime") != metadata or evidence.get("baseline_sources") != audit
                or len(evidence["checks"]) != len(expected_tasks)
                or {(row["suite"], row["task"]) for row in evidence["checks"]} != expected_tasks
                or any(row["exact_action_equality"] is not True or row["exact_observation_equality"] is not True
                       or row["max_absolute_action_error"] != 0 for row in evidence["checks"])):
            raise ValueError("Matching p=0 action/observation invariance evidence is required")
        policy = InstrumentedSmolVLAPolicy(cfg.base, tail_length=cfg.dvac.tail_length)
        validate_runtime(old, metadata, policy)
        if evidence["hardware"] != policy.runtime or evidence["sampler_manifest"] != policy.sampler_manifest():
            raise ValueError("Invariance hardware or sampler changed")
        check_sha = file_sha256(check_path)
        stable = {"config": json.loads(json.dumps(cfg.asdict())), "config_fingerprint": cfg.fingerprint,
                  "source_manifest": sources, "runtime": metadata, "hardware": policy.runtime,
                  "upstream_sampler": policy.sampler_manifest(), "baseline_sources": audit,
                  "invariance_check_sha256": check_sha,
                  "planned_episode_ids": [episode.id for episode in episodes],
                  "full_plan_episode_ids": [episode.id for episode in episodes]}
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if any(manifest.get(key) != value for key, value in stable.items()):
                raise ValueError("Resume configuration, sources, reference or runtime changed")
        else:
            manifest = {**stable, "schema_version": 1, "experiment_kind": "translation_dropout",
                        "data_kind": "real_libero", "created_utc": datetime.now(timezone.utc).isoformat(),
                        "git": git_metadata(),
                        "intervention": "After policy action clipping, replace XYZ by zero with probability p; preserve rotation/gripper. Fresh actual-state observations remain unchanged.",
                        "noise_pairing": "Same deterministic uniform per task/state/policy-seed/control-step for all methods and all p; one disturbance seed.",
                        "collection_order": "Shuffled matched blocks; fixed K=1 is last because of deadline cost.",
                        "timing": "New methods share FM diagnostics. Historical fixed controls differ in instrumentation/threading; use calls for cross-family cost."}
            (output / "invariance.json").write_bytes(check_path.read_bytes())
            atomic_json(manifest_path, manifest)
        if file_sha256(output / "invariance.json") != check_sha:
            raise ValueError("Stored invariance evidence changed")
        manifest_sha = file_sha256(manifest_path)
        expected = {episode.id: episode for episode in episodes}
        completed = set()
        for path in output.glob("episodes/*/summary.json"):
            row = json.loads(path.read_text())
            episode = expected.get(path.parent.name)
            if (episode is None or any(row.get(key) != value for key, value in episode.asdict().items())
                    or row.get("episode_id") != episode.id or row.get("pair_id") != episode.pair_id
                    or row.get("config_fingerprint") != cfg.fingerprint or row.get("manifest_sha256") != manifest_sha
                    or set(row.get("evidence_sha256", {})) != {"steps.jsonl", "chunks.jsonl", "video.mp4"}
                    or any(file_sha256(path.parent / name) != checksum for name, checksum in row["evidence_sha256"].items())):
                raise ValueError(f"Invalid saved episode: {path}")
            completed.add(episode.id)
        remaining = [episode for episode in episodes if episode.id not in completed
                     and (methods is None or episode.method in methods)]
        env = LiberoEnvironment(cfg.base)
        try:
            if not remaining:
                print("Selected episodes complete; evidence hashes match.", flush=True)
                return
            observation, _ = env.reset(remaining[0])
            for i in range(cfg.base.warmup_calls):
                policy.reset()
                policy.predict(observation, i)
            for number, episode in enumerate(remaining, 1):
                if (limit is not None and number > limit) or (stop_time is not None and time.time() >= stop_time):
                    print(f"Bounded stop: {len(completed)+number-1}/{len(episodes)} complete", flush=True)
                    break
                attempt = output / "attempts" / f"{episode.id}__{time.time_ns()}"
                attempt.mkdir(parents=True)
                video = EpisodeVideo(attempt / "video.mp4", episode, cfg.base.control_hz)
                selector = CappedDVACSelector(execution_cap=episode.execution_cap,
                    horizon=cfg.base.chunk_size, tail_length=cfg.dvac.tail_length,
                    window_size=cfg.dvac.window_size, alpha=cfg.dvac.alpha,
                    min_steps=cfg.dvac.min_steps) if episode.adaptive else None
                policy.reset_peak_memory()
                started = time.perf_counter()
                try:
                    summary, rows, chunks = run_episode(env, policy, episode, selector=selector,
                        reference=references[episode.pair_id], chunk_size=cfg.base.chunk_size,
                        control_hz=cfg.base.control_hz, record=video)
                    video.close()
                    summary.update(config_fingerprint=cfg.fingerprint, manifest_sha256=manifest_sha,
                                   data_kind="real_libero", peak_gpu_memory_bytes=policy.peak_memory(),
                                   episode_wall_s=time.perf_counter() - started)
                    write_jsonl(attempt / "steps.jsonl", rows)
                    write_jsonl(attempt / "chunks.jsonl", chunks)
                    summary["evidence_sha256"] = {name: file_sha256(attempt / name)
                                                 for name in ("steps.jsonl", "chunks.jsonl", "video.mp4")}
                    atomic_json(attempt / "summary.json", summary)
                    destination = output / "episodes" / episode.id
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    attempt.rename(destination)
                except BaseException:
                    atomic_json(attempt / "error.json", {"episode": episode.asdict(), "traceback": traceback.format_exc()})
                    raise
                finally:
                    video.close()
                print(f"[{len(completed)+number}/{len(episodes)}] p={episode.dropout_p:.2f} {episode.method} "
                      f"task={episode.task} state={episode.state} success={summary['success']} "
                      f"calls={summary['model_calls']} drops={summary['translation_drop_count']}/{summary['steps']} "
                      f"wall={summary['episode_wall_s']:.1f}s", flush=True)
        finally:
            env.close()
