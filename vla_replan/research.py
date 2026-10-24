"""Fixed/adaptive execution and paired continuations from a replayed K=1 prefix.

This extension has a separate manifest/schema. Original fixed-K results and their
analysis remain unchanged. The adaptive rule sees only predicted actions, not
simulator object positions, contacts, success labels, or future observations.
"""

import hashlib
import json
import math
import random
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .accelerator import prepare_platform
from .backend import LiberoEnvironment, SmolVLAPolicy, observation_hash
from .config import Episode, Experiment, digest
from .core import difference, norm, sampling_seed
from .storage import atomic_json, exclusive_run, git_metadata, runtime_metadata, write_jsonl
from .video import EpisodeVideo


def choose_interval(actions, method):
    """Frozen v1 rule: transition/turn -> 5, closing -> 10, otherwise -> 25."""
    if method.startswith("fixed"):
        return int(method.removeprefix("fixed")), "fixed"
    if method != "adaptive_v1":
        raise ValueError(f"Unknown execution method: {method}")
    first = actions[0]
    sign = first[6] > 0
    if any((a[6] > 0) != sign for a in actions[1:10]):
        return 5, "predicted_gripper_transition"
    a, b = first[:3], actions[9][:3]
    length = norm(a) * norm(b)
    if length > 1e-6 and sum(x * y for x, y in zip(a, b, strict=True)) / length < 0.5:
        return 5, "predicted_translation_turn"
    return (10, "closing_command") if sign else (25, "open_gripper_motion")


def validate_chunk(actions, horizon):
    if len(actions) != horizon or any(len(a) != 7 for a in actions):
        raise ValueError(f"Expected {horizon} x 7 actions")
    if any(not math.isfinite(v) for a in actions for v in a):
        raise ValueError("Nonfinite predicted action")


def simulation_hash(env):
    handle = getattr(env, "env", None)
    if handle is None:  # Unit-test doubles have no MuJoCo runtime.
        return None
    state = handle._env.env.sim.get_state().flatten()
    return hashlib.sha256(state.tobytes()).hexdigest()


def trial_id(episode, method):
    return f"{episode.suite}__{episode.task}__s{episode.state}__r{episode.policy_seed}__{method}"


def load_trial(path):
    path = Path(path)
    return {
        "summary": json.loads((path / "summary.json").read_text()),
        **{name: [json.loads(line) for line in (path / f"{name}.jsonl").read_text().splitlines()]
           for name in ("steps", "chunks")},
    }


def run_trial(env, policy, episode, method, config, record=None, reference=None, fork=0, progress=None):
    reset_start = time.perf_counter()
    obs, reset_info = env.reset(episode)
    policy.reset()
    reset_s = time.perf_counter() - reset_start
    replay_s = 0.0
    if reference is not None:
        if not 0 < fork < reference["summary"]["steps"]:
            raise ValueError("Fork must precede baseline termination")
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            if reset_info[field] != reference["summary"][field]:
                raise RuntimeError(f"Replay initial {field} differs")
        tick = time.perf_counter()
        for row in reference["steps"][:fork]:
            if observation_hash(obs.payload["raw"]) != row["observation_sha256"]:
                raise RuntimeError(f"Replay observation differs at step {row['step']}")
            if simulation_hash(env) != row["physics_state_sha256"]:
                raise RuntimeError(f"Replay physics state differs at step {row['step']}")
            obs, succeeded, terminal = env.step(row["action"])
            if succeeded or terminal:
                raise RuntimeError("Replay terminated before the fork")
        replay_s = time.perf_counter() - tick
        if observation_hash(obs.payload["raw"]) != reference["steps"][fork]["observation_sha256"]:
            raise RuntimeError("Fork observations differ")
        if simulation_hash(env) != reference["steps"][fork]["physics_state_sha256"]:
            raise RuntimeError("Fork physics states differ")
    elif fork:
        raise ValueError("A fork requires reference data")
    rows, chunks = [], []
    next_plan = fork
    chunk, index, interval = [], 0, 0
    previous_action = reference["steps"][fork - 1]["action"] if reference is not None else None
    inference_s = simulation_s = video_s = eef_path = 0.0
    first_chunk_error = None
    success = terminal = False
    started = time.perf_counter()
    for step in range(fork, episode.max_steps):
        replanned = step == next_plan
        latency = 0.0
        if replanned:
            seed = sampling_seed(episode.policy_seed, fork if method == "constant_noise" else step)
            tick = time.perf_counter()
            chunk = policy.predict(obs, seed)
            latency = time.perf_counter() - tick
            inference_s += latency
            validate_chunk(chunk, config.chunk_size)
            selected = "fixed1" if method == "constant_noise" else "fixed10" if method == "commit10" else method
            interval, reason = choose_interval(chunk, selected)
            if not 1 <= interval <= config.chunk_size:
                raise ValueError("Selected interval is outside prediction horizon")
            if reference is not None and step == fork:
                expected = next(c["actions"] for c in reference["chunks"] if c["step"] == fork)
                first_chunk_error = max(abs(x-y) for a,b in zip(chunk, expected, strict=True) for x,y in zip(a,b, strict=True))
                if first_chunk_error > 2e-6:
                    raise RuntimeError(f"First fork prediction differs: {first_chunk_error}")
            index = 0
            next_plan = step + interval
            chunks.append({"step": step, "sampling_seed": seed, "selected_k": interval,
                           "reason": reason, "actions": chunk, "inference_s": latency})
        predicted = chunk[index]
        action = [max(-1.0, min(1.0, x)) for x in predicted]
        row = {"step": step, "sim_time_s": step / config.control_hz, "replanned": replanned,
               "model_calls": len(chunks), "chunk_index": index, "active_k": interval,
               "plan_step": chunks[-1]["step"], "action": action, "eef": obs.eef,
               "gripper": obs.gripper, "observation_sha256": observation_hash(obs.payload["raw"]),
               "physics_state_sha256": simulation_hash(env),
               "action_translation_jump_l2": norm(difference(action[:3], previous_action[:3])) if previous_action else None,
               "inference_s": latency, "instruction": reset_info["instruction"]}
        if record is not None:
            tick = time.perf_counter()
            record(obs.frame, row, None)
            video_s += time.perf_counter() - tick
        tick = time.perf_counter()
        new_obs, success, terminal = env.step(action)
        simulation_s += time.perf_counter() - tick
        eef_path += norm(difference(new_obs.eef, obs.eef))
        row["success_after_action"] = bool(success)
        rows.append(row)
        obs = new_obs
        index += 1
        previous_action = action
        if progress and (replanned or success or terminal):
            progress(step + 1, len(chunks))
        if success or terminal:
            break
    total_steps = fork + len(rows)
    if record is not None:
        tick = time.perf_counter()
        record(obs.frame, {"step": total_steps, "sim_time_s": total_steps / config.control_hz,
                          "replanned": False, "model_calls": len(chunks), "active_k": interval}, success)
        video_s += time.perf_counter() - tick
    result = {**episode.asdict(), **reset_info, "method": method, "success": bool(success),
              "steps": total_steps, "executed_steps": len(rows), "fork_step": fork,
              "model_calls": len(chunks), "inference_s": inference_s, "simulation_s": simulation_s,
              "compute_s": inference_s + simulation_s, "reset_s": reset_s, "replay_s": replay_s,
              "video_encode_s": video_s, "rollout_wall_s": time.perf_counter() - started,
              "eef_path_m": eef_path, "first_chunk_max_abs_error": first_chunk_error,
              "termination": "success" if success else "terminated" if terminal else "time_limit",
              "mean_selected_k": sum(c["selected_k"] for c in chunks) / len(chunks),
              "gripper_sign_changes": sum((a["action"][6] > 0) != (b["action"][6] > 0) for a,b in zip(rows, rows[1:], strict=False)),
              "mean_action_jump": sum(r["action_translation_jump_l2"] or 0 for r in rows) / len(rows)}
    result["k"] = None if method == "adaptive_v1" else 1 if method == "constant_noise" else 10 if method == "commit10" else int(method.removeprefix("fixed"))
    return result, rows, chunks


def research_jobs(plan, config):
    rng = random.Random(plan["order_seed"])
    jobs = []
    for phase, states, methods in (
        ("development", plan["development_states"], plan["development_methods"]),
        ("heldout", plan["heldout_states"], plan["methods"]),
    ):
        blocks = [(task, state, seed) for task in config.tasks for state in states for seed in plan["policy_seeds"]]
        rng.shuffle(blocks)
        for task, state, seed in blocks:
            order = list(methods)
            rng.shuffle(order)
            for method in order:
                ep = Episode(task.suite, task.name, state, seed, 1, task.max_steps)
                jobs.append((phase, ep, method))
    blocks = [(task, state, seed) for task in config.tasks for state in plan["causal_states"] for seed in plan["policy_seeds"]]
    rng.shuffle(blocks)
    for task, state, seed in blocks:
        ep = Episode(task.suite, task.name, state, seed, 1, task.max_steps)
        order = ["constant_noise", "commit10"]
        rng.shuffle(order)
        jobs.extend(("causal", ep, method) for method in order)
    return jobs


def run_research(plan_path, output, device="auto", limit=None):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text())
    config = Experiment.load(plan_path.parent / plan["base_config"])
    config = replace(config, record_video=plan["record_video"])
    if set(plan["development_states"]) & set(plan["heldout_states"]):
        raise ValueError("Development and heldout states overlap")
    if not set(plan["causal_states"]) <= set(plan["heldout_states"]):
        raise ValueError("Causal states must be part of heldout sample")
    prepare_platform()
    with exclusive_run(output):
        signature = {"plan": plan, "base_config": config.asdict(), "runtime": runtime_metadata()}
        fingerprint = digest(signature)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest and manifest["fingerprint"] != fingerprint:
            raise ValueError("Research code, plan or runtime changed; use a new output directory")
        policy = SmolVLAPolicy(config, device)
        if manifest and manifest["hardware"] != policy.runtime:
            raise ValueError("Hardware changed")
        jobs = research_jobs(plan, config)
        if manifest is None:
            manifest = {**signature, "fingerprint": fingerprint, "schema": "replanning_research_v1",
                        "created_utc": datetime.now(timezone.utc).isoformat(), "hardware": policy.runtime,
                        "git": git_metadata(), "scheduled_jobs": len(jobs)}
            atomic_json(manifest_path, manifest)
        env = LiberoEnvironment(config)
        count = 0
        try:
            obs, _ = env.reset(jobs[0][1])
            for i in range(config.warmup_calls):
                policy.reset()
                policy.predict(obs, i)
            pairs = {}
            for p in output.glob("episodes/*/summary.json"):
                s = json.loads(p.read_text())
                pairs[s["pair_id"]] = (s["initial_state_sha256"], s["initial_observation_sha256"])
            for position, (phase, ep, method) in enumerate(jobs):
                identity = trial_id(ep, method)
                destination = output / "episodes" / identity
                if (destination / "summary.json").exists() or (destination / "skipped.json").exists():
                    continue
                if limit is not None and count >= limit:
                    break
                reference, fork = None, 0
                reference_path = output / "episodes" / trial_id(ep, "fixed1")
                if phase == "causal":
                    reference = load_trial(reference_path)
                    fork = plan["fork_step"]
                    if reference["summary"]["steps"] <= fork:
                        atomic_json(destination / "skipped.json", {"reason": "baseline_terminated_before_fork", "fork_step": fork})
                        continue
                attempt = output / "attempts" / f"{identity}__{time.time_ns()}"
                attempt.mkdir(parents=True)
                video = EpisodeVideo(attempt / "video.mp4", ep, config.control_hz, label=method) if config.record_video else None

                def progress(step, calls, phase=phase, position=position, identity=identity):
                    atomic_json(output / "status.json", {"phase": phase, "job": position + 1,
                                "total_jobs": len(jobs), "trial": identity, "step": step, "calls": calls})

                try:
                    progress(fork, 0)
                    summary, steps, chunks = run_trial(env, policy, ep, method, config, video, reference, fork, progress)
                    pair_id = ep.pair_id
                    pair = (summary["initial_state_sha256"], summary["initial_observation_sha256"])
                    if pair_id in pairs and pairs[pair_id] != pair:
                        raise RuntimeError("Paired initial conditions differ")
                    pairs[pair_id] = pair
                    if video:
                        video.close()
                        video = None
                    summary.update(phase=phase, pair_id=pair_id, trial_id=identity, fingerprint=fingerprint,
                                   data_kind="real_libero", reference_trial=trial_id(ep, "fixed1") if reference else None,
                                   reference_steps_sha256=hashlib.sha256((reference_path / "steps.jsonl").read_bytes()).hexdigest() if reference else None)
                    write_jsonl(attempt / "steps.jsonl", steps)
                    write_jsonl(attempt / "chunks.jsonl", chunks)
                    atomic_json(attempt / "summary.json", summary)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    attempt.rename(destination)
                    count += 1
                    print(f"[{position+1}/{len(jobs)}] {phase} {method} {ep.suite} s{ep.state} success={summary['success']} steps={summary['steps']} calls={summary['model_calls']} compute={summary['compute_s']:.1f}s", flush=True)
                except BaseException:
                    atomic_json(attempt / "error.json", {"traceback": traceback.format_exc(), "trial": identity})
                    raise
                finally:
                    if video:
                        video.close()
            complete = all((output / "episodes" / trial_id(ep, method) / "summary.json").exists() or
                           (output / "episodes" / trial_id(ep, method) / "skipped.json").exists() for _, ep, method in jobs)
            atomic_json(output / "status.json", {"phase": "completed" if complete else "paused_after_limit", "new_trials": count, "total_jobs": len(jobs)})
        finally:
            env.close()
