import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .accelerator import prepare_platform
from .core import run_episode
from .storage import atomic_json, exclusive_run, git_metadata, runtime_metadata, write_jsonl


def run(config, output, device="auto", limit=None):
    from .backend import LiberoEnvironment, SmolVLAPolicy
    from .video import EpisodeVideo

    output = Path(output)
    with exclusive_run(output):
        prepare_platform()
        metadata = runtime_metadata()
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest and (
            manifest["config_fingerprint"] != config.fingerprint or manifest["runtime"] != metadata
        ):
            raise ValueError("Configuration, code or software changed. Use a new output directory.")
        policy = SmolVLAPolicy(config, device)
        if manifest and manifest["hardware"] != policy.runtime:
            raise ValueError("Hardware changed. Use a new output directory for comparable timing.")
        if not manifest:
            manifest = {
                "schema_version": 1,
                "data_kind": "real_libero",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_fingerprint": config.fingerprint,
                "config": config.asdict(),
                "runtime": metadata,
                "hardware": policy.runtime,
                "git": git_metadata(),
                "planned_episode_ids": [e.id for e in config.episodes()],
            }
            atomic_json(manifest_path, manifest)
        env = LiberoEnvironment(config)
        try:
            episodes = config.episodes()
            observation, _ = env.reset(episodes[0])
            for i in range(config.warmup_calls):
                policy.reset()
                policy.predict(observation, i)
            fingerprints = {}
            for summary_path in output.glob("episodes/*/summary.json"):
                existing = json.loads(summary_path.read_text())
                fingerprints[existing["pair_id"]] = (
                    existing["initial_state_sha256"],
                    existing["initial_observation_sha256"],
                )
            count = 0
            for episode in episodes:
                destination = output / "episodes" / episode.id
                if (destination / "summary.json").exists():
                    continue
                if limit is not None and count >= limit:
                    break
                # Separate attempt directories preserve interrupted evidence; only complete runs are published.
                attempt = output / "attempts" / f"{episode.id}__{time.time_ns()}"
                attempt.mkdir(parents=True)
                video = (
                    EpisodeVideo(attempt / "video.mp4", episode, config.control_hz)
                    if config.record_video
                    else None
                )
                policy.reset_peak_memory()
                started = time.perf_counter()
                try:
                    summary, steps, chunks = run_episode(
                        env,
                        policy,
                        episode,
                        chunk_size=config.chunk_size,
                        control_hz=config.control_hz,
                        record=video,
                    )
                    fingerprint = (summary["initial_state_sha256"], summary["initial_observation_sha256"])
                    if episode.pair_id in fingerprints and fingerprints[episode.pair_id] != fingerprint:
                        raise RuntimeError(
                            "Paired initial observations differ; refusing an unfair comparison"
                        )
                    fingerprints[episode.pair_id] = fingerprint
                    if video:
                        video.close()
                        video = None
                    summary.update(
                        config_fingerprint=config.fingerprint,
                        data_kind="real_libero",
                        peak_gpu_memory_bytes=policy.peak_memory(),
                        episode_wall_s=time.perf_counter() - started,
                    )
                    write_jsonl(attempt / "steps.jsonl", steps)
                    write_jsonl(attempt / "chunks.jsonl", chunks)
                    atomic_json(attempt / "summary.json", summary)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    attempt.rename(destination)
                except BaseException:
                    atomic_json(
                        attempt / "error.json",
                        {"episode": episode.asdict(), "traceback": traceback.format_exc()},
                    )
                    raise
                finally:
                    if video:
                        video.close()
                count += 1
                print(
                    f"[{count}] K={episode.k} state={episode.state} success={summary['success']} "
                    f"calls={summary['model_calls']} compute={summary['compute_s']:.1f}s",
                    flush=True,
                )
        finally:
            env.close()
