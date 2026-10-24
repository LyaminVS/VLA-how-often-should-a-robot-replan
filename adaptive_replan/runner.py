"""Sequential collection with immutable manifests, trace hashes and paired resets."""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from vla_replan.storage import atomic_json, exclusive_run, git_metadata, runtime_metadata, write_jsonl

from .core import run_episode
from .integrity import file_sha256, source_manifest


def _validate_existing(directory, episode, config, manifest_sha):
    summary = json.loads((directory / "summary.json").read_text())
    for key, value in episode.asdict().items():
        if summary.get(key) != value:
            raise ValueError(f"Stored episode differs at {key}: {episode.id}")
    if summary.get("episode_id") != episode.id or summary.get("pair_id") != episode.pair_id:
        raise ValueError(f"Invalid stored episode identity: {episode.id}")
    if summary.get("config_fingerprint") != config.fingerprint or summary.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"Stored episode has a different manifest: {episode.id}")
    expected = {"steps.jsonl", "chunks.jsonl"}
    if config.base.record_video:
        expected.add("video.mp4")
    hashes = summary.get("evidence_sha256", {})
    if set(hashes) != expected:
        raise ValueError(f"Missing evidence hashes: {episode.id}")
    for filename, expected_sha in hashes.items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError(f"Stored evidence changed: {episode.id}/{filename}")
    return summary


def run(config, output, device="cuda", limit=None, integrity_check="outputs/dvac-check/invariance.json"):
    from vla_replan.backend import LiberoEnvironment

    from .policy import InstrumentedSmolVLAPolicy
    from .selector import DVACSelector
    from .video import EpisodeVideo

    config.validate()
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    output = Path(output)
    with exclusive_run(output):
        metadata = runtime_metadata()
        adaptive_source = source_manifest()
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest and (
            manifest["config_fingerprint"] != config.fingerprint
            or manifest["runtime"] != metadata
            or manifest["adaptive_source"] != adaptive_source
        ):
            raise ValueError("Configuration, code or software changed. Use a new output directory.")
        policy = InstrumentedSmolVLAPolicy(config.base, device=device, tail_length=config.dvac.tail_length)
        sampler = policy.sampler_manifest()
        check_path = Path(integrity_check)
        check = json.loads(check_path.read_text())
        expected_tasks = {(task.suite, task.name) for task in config.base.tasks}
        checks = check.get("checks", [])
        if (
            check.get("passed") is not True
            or len(checks) != len(expected_tasks)
            or {(row.get("suite"), row.get("task")) for row in checks} != expected_tasks
            or any(row.get("exact_action_equality") is not True
                   or row.get("max_absolute_action_error") != 0 for row in checks)
            or check.get("config_fingerprint") != config.fingerprint
            or check.get("source_manifest") != adaptive_source
            or check.get("sampler_manifest") != sampler
            or check.get("runtime") != metadata
            or check.get("hardware") != policy.runtime
        ):
            raise ValueError("A passing invariance check from the same sources, configuration and runtime is required")
        check_sha = file_sha256(check_path)
        if manifest and (manifest["hardware"] != policy.runtime or manifest["upstream_sampler"] != sampler):
            raise ValueError("Hardware or upstream sampler changed. Use a new output directory.")
        if manifest and manifest.get("invariance_check_sha256") != check_sha:
            raise ValueError("The invariance-check evidence changed. Use a new output directory.")
        episodes = config.episodes()
        if not manifest:
            manifest = {
                "schema_version": 1,
                "experiment_kind": "dvac_comparison",
                "data_kind": "real_libero",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_fingerprint": config.fingerprint,
                "config": config.asdict(),
                "runtime": metadata,
                "adaptive_source": adaptive_source,
                "upstream_sampler": sampler,
                "hardware": policy.runtime,
                "invariance_check_sha256": check_sha,
                "git": git_metadata(),
                "planned_episode_ids": [e.id for e in episodes],
                "initialization": "Reset DVAC history per episode; first call executes K=1 with no threshold.",
                "timing": "All methods collect identical FM diagnostics; selector_s measures DVAC selection only.",
            }
            (output / "invariance.json").write_bytes(check_path.read_bytes())
            atomic_json(manifest_path, manifest)
        if file_sha256(output / "invariance.json") != check_sha:
            raise ValueError("Stored invariance-check evidence changed")
        manifest_sha = file_sha256(manifest_path)
        expected_episodes = {episode.id: episode for episode in episodes}
        fingerprints = {}
        completed = set()
        for summary_path in sorted(output.glob("episodes/*/summary.json")):
            episode_id = summary_path.parent.name
            if episode_id not in expected_episodes:
                raise ValueError(f"Unexpected completed episode: {episode_id}")
            existing = _validate_existing(summary_path.parent, expected_episodes[episode_id], config, manifest_sha)
            fingerprint = (existing["initial_state_sha256"], existing["initial_observation_sha256"])
            physical_key = (existing["suite"], existing["task"], existing["state"])
            if physical_key in fingerprints and fingerprints[physical_key] != fingerprint:
                raise ValueError(f"Stored paired initial observations differ: {physical_key}")
            fingerprints[physical_key] = fingerprint
            completed.add(episode_id)
        env = LiberoEnvironment(config.base)
        try:
            remaining = [episode for episode in episodes if episode.id not in completed]
            if not remaining:
                print(f"All {len(episodes)} episodes already complete and evidence hashes match.", flush=True)
                return
            observation, _ = env.reset(remaining[0])
            for i in range(config.base.warmup_calls):
                policy.reset()
                policy.predict(observation, i)
            count = 0
            for episode in remaining:
                if limit is not None and count >= limit:
                    break
                destination = output / "episodes" / episode.id
                attempt = output / "attempts" / f"{episode.id}__{time.time_ns()}"
                attempt.mkdir(parents=True)
                video = (
                    EpisodeVideo(attempt / "video.mp4", episode, config.base.control_hz)
                    if config.base.record_video else None
                )
                selector = DVACSelector(
                    horizon=config.base.chunk_size,
                    tail_length=config.dvac.tail_length,
                    window_size=config.dvac.window_size,
                    alpha=config.dvac.alpha,
                    min_steps=config.dvac.min_steps,
                ) if episode.method == "dvac" else None
                policy.reset_peak_memory()
                started = time.perf_counter()
                try:
                    summary, steps, chunks = run_episode(
                        env, policy, episode, selector=selector,
                        chunk_size=config.base.chunk_size,
                        control_hz=config.base.control_hz, record=video,
                    )
                    fingerprint = (summary["initial_state_sha256"], summary["initial_observation_sha256"])
                    physical_key = (episode.suite, episode.task, episode.state)
                    if physical_key in fingerprints and fingerprints[physical_key] != fingerprint:
                        raise RuntimeError("Paired initial observations differ; refusing an unfair comparison")
                    fingerprints[physical_key] = fingerprint
                    if video:
                        video.close()
                        video = None
                    summary.update(
                        config_fingerprint=config.fingerprint,
                        manifest_sha256=manifest_sha,
                        data_kind="real_libero",
                        peak_gpu_memory_bytes=policy.peak_memory(),
                        episode_wall_s=time.perf_counter() - started,
                    )
                    write_jsonl(attempt / "steps.jsonl", steps)
                    write_jsonl(attempt / "chunks.jsonl", chunks)
                    filenames = ["steps.jsonl", "chunks.jsonl"]
                    if config.base.record_video:
                        filenames.append("video.mp4")
                    summary["evidence_sha256"] = {name: file_sha256(attempt / name) for name in filenames}
                    atomic_json(attempt / "summary.json", summary)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    attempt.rename(destination)
                except BaseException:
                    atomic_json(attempt / "error.json", {
                        "episode": episode.asdict(), "traceback": traceback.format_exc(),
                    })
                    raise
                finally:
                    if video:
                        video.close()
                count += 1
                print(
                    f"[{len(completed) + count}/{len(episodes)}] {episode.method} "
                    f"task={episode.task} state={episode.state} seed={episode.policy_seed} "
                    f"success={summary['success']} calls={summary['model_calls']} "
                    f"mean_K={summary['mean_chosen_k']:.2f} compute={summary['compute_s']:.1f}s",
                    flush=True,
                )
        finally:
            env.close()
