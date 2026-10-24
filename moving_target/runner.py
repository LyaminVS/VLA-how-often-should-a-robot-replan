"""Auditable, resumable moving-target runs using the unchanged baseline executor."""

import hashlib
import json
import math
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from vla_replan.config import digest
from vla_replan.core import run_episode
from vla_replan.storage import atomic_json, exclusive_run, git_metadata, runtime_metadata, write_jsonl

DATA_KIND = "moving_target_stress_test"


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_metadata():
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / "moving_target").glob("*.py"))
    paths.append(project / "scripts/run_moving_target.py")
    files = {str(path.relative_to(project)): sha256_file(path) for path in paths}
    return {"files": files, "sha256": digest(files)}


def _fingerprint_key(summary):
    # The physical initial state must also agree across policy-sampling seeds.
    return summary["suite"], summary["task"], summary["state"]


def _check_initial_observation(summary, fingerprints):
    key = _fingerprint_key(summary)
    value = (summary["initial_state_sha256"], summary["initial_observation_sha256"])
    if key in fingerprints and fingerprints[key] != value:
        raise RuntimeError("Paired physical initial observations differ; refusing an unfair comparison")
    fingerprints[key] = value


def verify_episode(directory, episode, config, sources):
    """Verify every published payload before allowing a completed episode to be skipped."""
    directory = Path(directory)
    evidence_path = directory / "evidence.json"
    if not evidence_path.is_file():
        raise ValueError(f"Missing episode integrity record: {directory}")
    evidence = json.loads(evidence_path.read_text())
    expected = {"summary.json", "steps.jsonl", "chunks.jsonl"}
    if config.base_config.record_video:
        expected.add("video.mp4")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("episode_id") != episode.id
        or evidence.get("config_fingerprint") != config.fingerprint
        or evidence.get("sources_sha256") != sources["sha256"]
        or set(evidence.get("files", {})) != expected
    ):
        raise ValueError(f"Episode integrity identity changed: {directory}")
    for name, expected_hash in evidence["files"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Episode payload checksum mismatch: {path}")
    summary = json.loads((directory / "summary.json").read_text())
    expected_fields = {
        **episode.asdict(),
        "episode_id": episode.id,
        "pair_id": episode.pair_id,
        "env_seed": episode.env_seed,
        "config_fingerprint": config.fingerprint,
        "data_kind": DATA_KIND,
    }
    if any(summary.get(key) != value for key, value in expected_fields.items()):
        raise ValueError(f"Episode summary identity changed: {directory}")
    return summary


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize_motion(rows, final, control_hz):
    """Control-rate diagnostics; contact is a physical event, not a grasp-success label."""
    samples = [row["motion"] for row in rows]
    active = [motion for motion in samples if motion.get("motion_active")]
    errors = [float(m["tracking_error_m"]) for m in active if m.get("tracking_error_m") is not None]
    forces = []
    for motion in active:
        value = motion.get("horizontal_force_n")
        if value is not None:
            forces.append(math.sqrt(sum(float(x) ** 2 for x in value)) if isinstance(value, (list, tuple)) else abs(float(value)))
    positions = [m["object_pos"] for m in samples + [final] if m.get("object_pos") is not None]
    contact_step = final.get("first_contact_step")
    return {
        "first_contact_step": contact_step,
        "first_contact_sim_time": final.get("first_contact_sim_time"),
        "contact_observed": contact_step is not None,
        "active_samples": len(active),
        "active_duration_s": len(active) / control_hz,
        "tracking_error_mean_m": _mean(errors),
        "tracking_error_max_m": max(errors) if errors else None,
        "horizontal_force_mean_n": _mean(forces),
        "horizontal_force_max_n": max(forces) if forces else None,
        "force_saturated_fraction": _mean([float(bool(m["force_saturated"])) for m in active if "force_saturated" in m]),
        "all_episode_object_x_range_m": max(p[0] for p in positions) - min(p[0] for p in positions) if positions else None,
        "diagnostic_sampling_hz": control_hz,
        "physics_diagnostics": {key: final.get(key) for key in (
            "active_physics_samples", "active_sim_time_s", "tracking_mae_m", "tracking_rmse_m",
            "force_saturation_fraction", "peak_force_n", "max_force_after_contact_n",
            "precontact_x_range_m", "precontact_y_range_m", "precontact_z_range_m",
            "max_precontact_tilt_deg",
        )},
    }


class MovingEpisodeVideo:
    """Stream frames with instruction, treatment, force status and replan markers."""

    def __init__(self, path, episode, fps):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=7, macro_block_size=16,
            ffmpeg_log_level="error", ffmpeg_params=["-threads", "1"],
        )
        self.episode = episode
        self.fps = fps
        self.instruction = episode.base_episode.task.replace("_", " ")
        self.last_replan = 0

    def __call__(self, frame, row, outcome):
        import textwrap

        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        if frame is None:
            raise ValueError("Recording requested but simulator returned no frame")
        self.instruction = row.get("instruction", self.instruction)
        if row["replanned"]:
            self.last_replan = row["step"]
        canvas = Image.new("RGB", (768, 576), "#101a2c")
        canvas.paste(Image.fromarray(frame).resize((768, 384)), (0, 192))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=16)
        small = ImageFont.load_default(size=13)
        for i, line in enumerate(textwrap.wrap(self.instruction, width=82)[:2]):
            draw.text((16, 10 + i * 21), line, font=font, fill="white")
        color = "#f6c550" if row["replanned"] else "#97abc7"
        status = "REPLAN" if row["replanned"] else f"Executing | last replan: step {self.last_replan}"
        if outcome is not None:
            status = "SUCCESS" if outcome else "FAILURE / EPISODE ENDED"
            color = "#71e4ac" if outcome else "#ff8d91"
        draw.text((16, 60), f"Condition: {self.episode.condition.upper()} | K={self.episode.base_episode.k} | step {row['step']} | calls={row['model_calls']}", font=font, fill="white")
        draw.text((16, 86), status, font=font, fill=color)
        motion = row["motion"]
        contact = motion.get("first_contact_step")
        motion_status = "FORCE ACTIVE" if motion.get("motion_active") else "FORCE INACTIVE"
        if contact is not None:
            motion_status = f"FINGER CONTACT at step {contact} | force permanently disabled"
        elif self.episode.condition == "static":
            motion_status = "STATIC CONTROL | no external force"
        draw.text((16, 112), motion_status, font=small, fill="white")
        error = motion.get("tracking_error_m")
        if error is not None and motion.get("motion_active"):
            draw.text((16, 138), f"Target tracking error: {1000 * error:.1f} mm", font=small, fill="#97abc7")
        draw.text((16, 165), f"Simulation time: {row['sim_time_s']:.2f} s | 20 Hz | contact is not proof of a grasp", font=small, fill="#97abc7")
        draw.rectangle((0, 188, 767, 191), fill=color)
        encoded = np.asarray(canvas)
        for _ in range(self.fps if outcome is not None else 1):
            self.writer.append_data(encoded)

    def close(self):
        self.writer.close()


def run(config, output, device="cuda", limit=None):
    from vla_replan.backend import SmolVLAPolicy

    from .environment import MovingTargetEnvironment

    config.validate()
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    output = Path(output)
    base = config.base_config
    with exclusive_run(output):
        metadata = runtime_metadata()
        sources = source_metadata()
        episodes = config.episodes()
        planned_ids = [episode.id for episode in episodes]
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest and (
            manifest.get("data_kind") != DATA_KIND
            or manifest.get("config_fingerprint") != config.fingerprint
            or digest(manifest.get("config")) != config.fingerprint
            or manifest.get("runtime") != metadata
            or manifest.get("sources") != sources
            or manifest.get("planned_episode_ids") != planned_ids
        ):
            raise ValueError("Configuration, plan, code or software changed. Use a new output directory.")
        published = list((output / "episodes").iterdir()) if (output / "episodes").exists() else []
        if published and not manifest:
            raise ValueError("Published episodes exist without a manifest")
        if any(path.name not in planned_ids or not path.is_dir() for path in published):
            raise ValueError("Unexpected published episode outside the registered plan")
        fingerprints = {}
        completed = set()
        by_id = {episode.id: episode for episode in episodes}
        for directory in published:
            summary = verify_episode(directory, by_id[directory.name], config, sources)
            _check_initial_observation(summary, fingerprints)
            completed.add(directory.name)
        if len(completed) == len(episodes):
            print(f"All {len(episodes)} moving-target episodes are complete and integrity-checked.", flush=True)
            return {"completed": len(completed), "new_episodes": 0, "planned": len(episodes)}
        policy = SmolVLAPolicy(base, device)
        if manifest and manifest.get("hardware") != policy.runtime:
            raise ValueError("Hardware changed. Use a new output directory for comparable timing.")
        if not manifest:
            manifest = {
                "schema_version": 1,
                "data_kind": DATA_KIND,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_fingerprint": config.fingerprint,
                "config": config.asdict(),
                "runtime": metadata,
                "sources": sources,
                "hardware": policy.runtime,
                "git": git_metadata(),
                "planned_episode_ids": planned_ids,
                "design": "Condition and K randomized within each task/state/policy-seed block; world-x target driven in simulation time; force removed at first finger contact.",
            }
            atomic_json(manifest_path, manifest)
        env = MovingTargetEnvironment(base, config.motion)
        count = 0
        try:
            env.set_condition(episodes[0].condition)
            observation, _ = env.reset(episodes[0].base_episode)
            for i in range(base.warmup_calls):
                policy.reset()
                policy.predict(observation, i)
            for episode in episodes:
                if episode.id in completed:
                    continue
                if limit is not None and count >= limit:
                    break
                env.set_condition(episode.condition)
                destination = output / "episodes" / episode.id
                attempt = output / "attempts" / f"{episode.id}__{time.time_ns()}"
                attempt.mkdir(parents=True)
                video = None
                final_motion = {}

                def record(frame, row, outcome):
                    nonlocal final_motion
                    row["motion"] = deepcopy(env.telemetry())
                    if outcome is not None:
                        final_motion = row["motion"]
                    if video is not None:
                        video(frame, row, outcome)

                started = time.perf_counter()
                try:
                    if base.record_video:
                        video = MovingEpisodeVideo(attempt / "video.mp4", episode, base.control_hz)
                    policy.reset_peak_memory()
                    summary, rows, chunks = run_episode(
                        env, policy, episode.base_episode, chunk_size=base.chunk_size,
                        control_hz=base.control_hz, record=record,
                    )
                    _check_initial_observation(summary, fingerprints)
                    if video is not None:
                        video.close()
                        video = None
                    summary.update(
                        episode_id=episode.id,
                        pair_id=episode.pair_id,
                        condition=episode.condition,
                        config_fingerprint=config.fingerprint,
                        data_kind=DATA_KIND,
                        peak_gpu_memory_bytes=policy.peak_memory(),
                        episode_wall_s=time.perf_counter() - started,
                        motion_summary=summarize_motion(rows, final_motion, base.control_hz),
                        motion_final=final_motion,
                    )
                    write_jsonl(attempt / "steps.jsonl", rows)
                    write_jsonl(attempt / "chunks.jsonl", chunks)
                    atomic_json(attempt / "summary.json", summary)
                    names = ["summary.json", "steps.jsonl", "chunks.jsonl"]
                    if base.record_video:
                        names.append("video.mp4")
                    atomic_json(attempt / "evidence.json", {
                        "schema_version": 1,
                        "episode_id": episode.id,
                        "config_fingerprint": config.fingerprint,
                        "sources_sha256": sources["sha256"],
                        "files": {name: sha256_file(attempt / name) for name in names},
                    })
                    verify_episode(attempt, episode, config, sources)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    attempt.rename(destination)
                except BaseException:
                    atomic_json(attempt / "error.json", {"episode": episode.asdict(), "traceback": traceback.format_exc()})
                    raise
                finally:
                    if video is not None:
                        video.close()
                count += 1
                completed.add(episode.id)
                print(f"[{len(completed)}/{len(episodes)}] {episode.condition} K={episode.base_episode.k} state={episode.base_episode.state} seed={episode.base_episode.policy_seed} success={summary['success']} calls={summary['model_calls']} compute={summary['compute_s']:.1f}s", flush=True)
        finally:
            env.close()
        return {"completed": len(completed), "new_episodes": count, "planned": len(episodes)}
