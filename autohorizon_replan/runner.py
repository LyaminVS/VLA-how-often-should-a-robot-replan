"""Sequential, resumable AutoHorizon collection against immutable DVAC references."""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from capped_replan.core import run_episode
from capped_replan.reference import MATCH_FIELDS
from vla_replan.storage import atomic_json, exclusive_run, git_metadata, runtime_metadata, write_jsonl

from .integrity import file_sha256, source_manifest


def load_references(root, cfg):
    from capped_replan.integrity import source_manifest as capped_sources

    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    canonical_base = json.loads(json.dumps(cfg.base.asdict()))
    if manifest.get("data_kind") != "real_libero" or manifest["capped_source"] != capped_sources():
        raise ValueError("Historical model/rollout sources changed")
    for field in MATCH_FIELDS:
        if manifest["config"]["base"][field] != canonical_base[field]:
            raise ValueError(f"Historical inference setting differs: {field}")
    if manifest["config"]["base"]["tasks"] != canonical_base["tasks"]:
        raise ValueError("Historical tasks or episode limits differ")
    files = {name: file_sha256(root / name) for name in ("manifest.json", "invariance.json")}
    required = {episode.pair_id for episode in cfg.episodes()}
    references = {}
    for path in sorted(root.glob("episodes/*/summary.json")):
        row = json.loads(path.read_text())
        if row.get("method") != "dvac_cap_50" or row.get("pair_id") not in required:
            continue
        if (row["manifest_sha256"] != files["manifest.json"]
                or row["config_fingerprint"] != manifest["config_fingerprint"]
                or row["episode_id"] != path.parent.name or row["data_kind"] != "real_libero"):
            raise ValueError("Historical identity differs from its manifest")
        files[str(path.relative_to(root))] = file_sha256(path)
        if set(row.get("evidence_sha256", {})) != {"chunks.jsonl", "steps.jsonl", "video.mp4"}:
            raise ValueError("Historical episode must include all three evidence files")
        for filename, sha in row["evidence_sha256"].items():
            artifact = path.parent / filename
            if filename not in ("chunks.jsonl", "steps.jsonl", "video.mp4") or file_sha256(artifact) != sha:
                raise ValueError(f"Historical evidence changed: {artifact}")
            files[str(artifact.relative_to(root))] = sha
        with (path.parent / "chunks.jsonl").open() as stream:
            first = json.loads(next(stream))
        if row["pair_id"] in references or first["step"] != 0:
            raise ValueError("Duplicate or invalid reference")
        references[row["pair_id"]] = {
            "pair_id": row["pair_id"], "initial_state_sha256": row["initial_state_sha256"],
            "initial_observation_sha256": row["initial_observation_sha256"],
            "sampling_seed": first["sampling_seed"], "actions": first["actions"],
            "reference_episode_ids": [row["episode_id"]],
        }
    if set(references) != required:
        raise ValueError("References must cover exactly every requested pair")
    return references, {"root": str(root), "files": dict(sorted(files.items()))}, manifest


def _reference_runtime(old, metadata, policy):
    if old["runtime"] != metadata or old["hardware"] != policy.runtime:
        raise ValueError("Historical adaptive reference runtime/hardware differs")
    sampler = policy.sampler_manifest()
    if any(sampler.get(key) != value for key, value in old["upstream_sampler"].items()):
        raise ValueError("Historical upstream sampler changed")


def check(cfg, output, reference_root):
    from vla_replan.backend import LiberoEnvironment, SmolVLAPolicy
    from vla_replan.config import Episode
    from .policy import AutoHorizonSmolVLAPolicy
    from .selector import AutoHorizonSelector

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _, audit, old = load_references(reference_root, cfg)
    before = source_manifest()
    metadata = runtime_metadata()
    policy = AutoHorizonSmolVLAPolicy(cfg.base, tail_length=cfg.dvac.tail_length,
                                    selected_fm_step=cfg.autohorizon.selected_fm_step)
    _reference_runtime(old, metadata, policy)
    env = LiberoEnvironment(cfg.base)
    checks = []
    try:
        for task in cfg.base.tasks:
            episode = Episode(task.suite, task.name, 21, 101, 1, task.max_steps)
            obs, reset = env.reset(episode)
            policy.reset()
            plain = SmolVLAPolicy.predict(policy, obs, 101)
            policy.reset()
            observed = policy.predict(obs, 101)
            error = max(abs(x-y) for a, b in zip(plain, observed, strict=True)
                        for x, y in zip(a, b, strict=True))
            if plain != observed or error != 0:
                raise ValueError("Attention capture changed the complete predicted chunk")
            selector = AutoHorizonSelector(horizon=cfg.base.chunk_size,
                                          execution_cap=cfg.base.chunk_size)
            decision = selector.select(policy.last_diagnostics["attention"])
            checks.append({"suite": task.suite, "task": task.name, "initial_state": 21,
                           "sampling_seed": 101, "exact_action_equality": True,
                           "max_absolute_action_error": error,
                           "attention_capture": policy.last_diagnostics["attention_capture"],
                           "attention_shape": [cfg.base.chunk_size, cfg.base.chunk_size],
                           "raw_chosen_k": decision["raw_chosen_k"], **reset})
            print(f"PASS {task.name}: identical full chunk; attention layers="
                  f"{checks[-1]['attention_capture']['layer_indices']}; raw K={decision['raw_chosen_k']}", flush=True)
    finally:
        env.close()
    if before != source_manifest():
        raise ValueError("Collection sources changed during the check")
    atomic_json(output, {"passed": True, "check_kind": "autohorizon_invariance",
                        "robot_actions_executed": 0, "checks": checks, "config": cfg.asdict(),
                        "config_fingerprint": cfg.fingerprint, "source_manifest": before,
                        "runtime": metadata, "hardware": policy.runtime,
                        "sampler_manifest": policy.sampler_manifest(), "reference_sources": audit})


def run(cfg, output, integrity_check, reference_root, limit=None):
    from vla_replan.backend import LiberoEnvironment
    from .policy import AutoHorizonSmolVLAPolicy, SelectorAdapter
    from .selector import AutoHorizonSelector
    from .video import EpisodeVideo

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    episodes = cfg.episodes()
    references, audit, old = load_references(reference_root, cfg)
    output, check_path = Path(output), Path(integrity_check)
    with exclusive_run(output):
        sources, metadata = source_manifest(), runtime_metadata()
        evidence = json.loads(check_path.read_text())
        if (evidence.get("passed") is not True or evidence.get("source_manifest") != sources
                or evidence.get("config_fingerprint") != cfg.fingerprint
                or evidence.get("runtime") != metadata or evidence.get("reference_sources") != audit
                or {(row["suite"], row["task"]) for row in evidence["checks"]}
                != {(task.suite, task.name) for task in cfg.base.tasks}
                or any(row["exact_action_equality"] is not True or row["max_absolute_action_error"] != 0
                       for row in evidence["checks"])):
            raise ValueError("A passing matching AutoHorizon invariance check is required")
        policy = AutoHorizonSmolVLAPolicy(cfg.base, tail_length=cfg.dvac.tail_length,
                                        selected_fm_step=cfg.autohorizon.selected_fm_step)
        _reference_runtime(old, metadata, policy)
        if evidence["hardware"] != policy.runtime or evidence["sampler_manifest"] != policy.sampler_manifest():
            raise ValueError("Invariance hardware or sampler changed")
        check_sha = file_sha256(check_path)
        stable = {"config_fingerprint": cfg.fingerprint, "config": json.loads(json.dumps(cfg.asdict())),
                  "source_manifest": sources, "runtime": metadata, "hardware": policy.runtime,
                  "upstream_sampler": policy.sampler_manifest(), "reference_sources": audit,
                  "invariance_check_sha256": check_sha, "planned_episode_ids": [e.id for e in episodes]}
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if any(manifest.get(key) != value for key, value in stable.items()):
                raise ValueError("Resume configuration/code/runtime changed; use a new directory")
        else:
            manifest = {**stable, "schema_version": 1, "experiment_kind": "autohorizon_comparison",
                        "data_kind": "real_libero", "created_utc": datetime.now(timezone.utc).isoformat(),
                        "git": git_metadata(),
                        "initialization": "AutoHorizon acts from the first call; no forced K=1 initialization.",
                        "timing": "Additional attention diagnostics are recorded; cross-family wall time is not a fair efficiency metric."}
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
                    or row.get("config_fingerprint") != cfg.fingerprint
                    or row.get("manifest_sha256") != manifest_sha
                    or set(row.get("evidence_sha256", {})) != {"steps.jsonl", "chunks.jsonl", "video.mp4"}
                    or any(file_sha256(path.parent / name) != sha for name, sha in row["evidence_sha256"].items())):
                raise ValueError(f"Invalid existing episode: {path}")
            completed.add(episode.id)
        remaining = [episode for episode in episodes if episode.id not in completed]
        env = LiberoEnvironment(cfg.base)
        try:
            if not remaining:
                print("All episodes complete; hashes match.", flush=True)
                return
            obs, _ = env.reset(remaining[0])
            for i in range(cfg.base.warmup_calls):
                policy.reset()
                policy.predict(obs, i)
            for number, episode in enumerate(remaining, 1):
                if limit is not None and number > limit:
                    break
                attempt = output / "attempts" / f"{episode.id}__{time.time_ns()}"
                attempt.mkdir(parents=True)
                video = EpisodeVideo(attempt / "video.mp4", episode, cfg.base.control_hz)
                selector = SelectorAdapter(AutoHorizonSelector(
                    horizon=cfg.base.chunk_size, execution_cap=episode.execution_cap,
                    hold_thr=cfg.autohorizon.hold_thr, max_entropy_q=cfg.autohorizon.max_entropy_q,
                    selected_fm_step=cfg.autohorizon.selected_fm_step), policy)
                policy.reset_peak_memory()
                started = time.perf_counter()
                try:
                    summary, rows, chunks = run_episode(
                        env, policy, episode, selector=selector, reference=references[episode.pair_id],
                        chunk_size=cfg.base.chunk_size, control_hz=cfg.base.control_hz, record=video)
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
                print(f"[{len(completed)+number}/{len(episodes)}] {episode.method} task={episode.task} "
                      f"state={episode.state} seed={episode.policy_seed} success={summary['success']} "
                      f"calls={summary['model_calls']} mean_K={summary['mean_chosen_k']:.2f} "
                      f"wall={summary['episode_wall_s']:.1f}s", flush=True)
        finally:
            env.close()
