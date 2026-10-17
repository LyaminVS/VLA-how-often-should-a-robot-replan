"""Synchronous receding-horizon execution, independent of torch and LIBERO.

The simulator is paused while the policy predicts. A new chunk replaces the previous
one after exactly K control steps; termination takes priority over the next action.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass
class Observation:
    payload: Any
    eef: list[float]
    gripper: list[float]
    frame: Any = None


class Policy(Protocol):
    def reset(self): ...
    def predict(self, observation: Observation, seed: int) -> list[list[float]]: ...


class Environment(Protocol):
    def reset(self, episode) -> tuple[Observation, dict]: ...
    def step(self, action: list[float]) -> tuple[Observation, bool, bool]: ...


def norm(vector):
    return math.sqrt(sum(float(x) ** 2 for x in vector))


def difference(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


def sampling_seed(base, step):
    # Sampling noise is tied to simulation time, not run order or the number of calls.
    return (base * 1_000_003 + step * 97_409) % (2**63 - 1)


def run_episode(
    env: Environment,
    policy: Policy,
    episode,
    *,
    chunk_size=50,
    control_hz=20,
    record: Callable | None = None,
    clock=time.perf_counter,
):
    if not 1 <= episode.k <= chunk_size:
        raise ValueError("Execution interval must be in [1, chunk_size]")
    reset_start = clock()
    obs, reset_info = env.reset(episode)
    policy.reset()
    reset_s = clock() - reset_start
    rows, predictions = [], []
    chunk, chunk_index, calls = [], 0, 0
    inference_s = simulation_s = video_s = 0.0
    path_m = 0.0
    close_events = clipped_values = 0
    previous_action = None
    success = False
    terminal = False
    start = clock()
    for step in range(episode.max_steps):
        replanned = step % episode.k == 0
        latency = 0.0
        revision_l2 = None
        if replanned:
            old_next = chunk[chunk_index] if chunk_index < len(chunk) else None
            tick = clock()
            chunk = policy.predict(obs, sampling_seed(episode.policy_seed, step))
            latency = clock() - tick
            inference_s += latency
            if len(chunk) != chunk_size or any(len(a) != 7 for a in chunk):
                raise ValueError(f"Expected a {chunk_size} x 7 action chunk")
            if any(not math.isfinite(v) for a in chunk for v in a):
                raise ValueError("Model returned nonfinite actions")
            if old_next is not None:
                # Translation command coordinates, not Cartesian metres or proof of correction.
                revision_l2 = norm(difference(chunk[0][:3], old_next[:3]))
            chunk_index = 0
            calls += 1
            predictions.append(
                {
                    "step": step,
                    "sampling_seed": sampling_seed(episode.policy_seed, step),
                    "actions": chunk,
                    "inference_s": latency,
                }
            )
        predicted = chunk[chunk_index]
        action = [max(-1.0, min(1.0, x)) for x in predicted]
        clipped_values += sum(x != y for x, y in zip(predicted, action, strict=True))
        if action[6] > 0 and (previous_action is None or previous_action[6] <= 0):
            close_events += 1
        jump = norm(difference(action[:3], previous_action[:3])) if previous_action else None
        row = {
            "step": step,
            "sim_time_s": step / control_hz,
            "replanned": replanned,
            "model_calls": calls,
            "chunk_index": chunk_index,
            "action": action,
            "eef": obs.eef,
            "gripper": obs.gripper,
            "inference_s": latency,
            "plan_revision_translation_l2": revision_l2,
            "action_translation_jump_l2": jump,
        }
        if step == 0 and isinstance(obs.payload, dict) and "instruction" in obs.payload:
            row["instruction"] = obs.payload["instruction"]
        # Frame corresponds to the observation that produced this action and replan marker.
        if record is not None:
            tick = clock()
            record(obs.frame, row, None)
            video_s += clock() - tick
        tick = clock()
        new_obs, step_success, terminal = env.step(action)
        simulation_s += clock() - tick
        path_m += norm(difference(new_obs.eef, obs.eef))
        row["success_after_action"] = bool(step_success)
        rows.append(row)
        obs = new_obs
        success = success or bool(step_success)
        chunk_index += 1
        previous_action = action
        if terminal or success:
            break
    elapsed = clock() - start
    steps = len(rows)
    if record is not None:
        tick = clock()
        record(
            obs.frame,
            {"step": steps, "sim_time_s": steps / control_hz, "replanned": False, "model_calls": calls},
            success,
        )
        video_s += clock() - tick

    def mean_present(field, only_replans=False):
        values = [r[field] for r in rows if r[field] is not None and (not only_replans or r["replanned"])]
        return sum(values) / len(values) if values else None

    summary = {
        **episode.asdict(),
        "episode_id": episode.id,
        "pair_id": episode.pair_id,
        "env_seed": episode.env_seed,
        "success": success,
        "steps": steps,
        "termination": "success" if success else "terminated" if terminal else "time_limit",
        "model_calls": calls,
        "sim_time_s": steps / control_hz,
        "inference_s": inference_s,
        "simulation_s": simulation_s,
        "rollout_wall_s": elapsed,
        "reset_s": reset_s,
        "video_encode_s": video_s,
        "compute_s": inference_s + simulation_s,
        "eef_path_m": path_m,
        "mean_plan_revision_translation_l2": mean_present("plan_revision_translation_l2"),
        "mean_replan_jump_translation_l2": mean_present("action_translation_jump_l2", True),
        "gripper_close_commands": close_events,
        "clipped_action_values": clipped_values,
        **reset_info,
    }
    return summary, rows, predictions
