"""Execute the final sampled action block with a fixed or variance-selected prefix."""

import copy
import math
import time

from vla_replan.core import difference, norm, sampling_seed


def _validate_prediction(chunk, diagnostics, chunk_size):
    if len(chunk) != chunk_size or any(len(action) != 7 for action in chunk):
        raise ValueError(f"Expected a {chunk_size} x 7 final action chunk")
    if any(not math.isfinite(value) for action in chunk for value in action):
        raise ValueError("Model returned nonfinite actions")
    variance = diagnostics["variance"]
    if len(variance) != chunk_size or any(not math.isfinite(v) or v < 0 for v in variance):
        raise ValueError("Invalid per-action denoising variance")
    clean = diagnostics["clean_estimates"]
    times = diagnostics["fm_times"]
    if not clean or len(clean) != len(times) or any(not math.isfinite(t) for t in times):
        raise ValueError("Invalid FM diagnostic times")
    for estimate in clean:
        if len(estimate) != chunk_size or any(len(action) != 7 for action in estimate):
            raise ValueError("Invalid clean-action estimate shape")
        if any(not math.isfinite(v) for action in estimate for v in action):
            raise ValueError("Nonfinite clean-action estimate")


def run_episode(
    env, policy, episode, *, selector=None, chunk_size=50, control_hz=20,
    record=None, clock=time.perf_counter,
):
    if episode.method == "dvac":
        if selector is None:
            raise ValueError("DVAC requires a selector")
    elif episode.k is None or not 1 <= episode.k <= chunk_size:
        raise ValueError("Fixed execution interval must be in [1, chunk_size]")
    reset_start = clock()
    obs, reset_info = env.reset(episode)
    policy.reset()
    if selector is not None:
        selector.reset()
    reset_s = clock() - reset_start
    rows, predictions = [], []
    chunk, chunk_index, calls, next_replan_step = [], 0, 0, 0
    inference_s = selector_s = simulation_s = video_s = 0.0
    path_m = 0.0
    close_events = clipped_values = 0
    previous_action = None
    success = terminal = False
    decision = None
    start = clock()
    for step in range(episode.max_steps):
        replanned = step == next_replan_step
        latency = selection_latency = 0.0
        revision_l2 = None
        if replanned:
            old_next = chunk[chunk_index] if chunk_index < len(chunk) else None
            seed = sampling_seed(episode.policy_seed, step)
            tick = clock()
            chunk = policy.predict(obs, seed)
            latency = clock() - tick
            inference_s += latency
            diagnostics = copy.deepcopy(policy.last_diagnostics)
            _validate_prediction(chunk, diagnostics, chunk_size)
            if episode.method == "dvac":
                tick = clock()
                decision = dict(selector.select(diagnostics["variance"]))
                selection_latency = clock() - tick
                selector_s += selection_latency
            else:
                decision = {
                    "chosen_k": episode.k, "threshold": None, "first_crossing": None,
                    "window_calls": 0, "window_mean": None, "window_std": None,
                    "bootstrap": False,
                }
            chosen_k = decision["chosen_k"]
            if type(chosen_k) is not int or not 1 <= chosen_k <= chunk_size:
                raise ValueError("Selector returned an invalid execution horizon")
            next_replan_step = step + chosen_k
            if old_next is not None:
                revision_l2 = norm(difference(chunk[0][:3], old_next[:3]))
            chunk_index = 0
            calls += 1
            predictions.append({
                "step": step,
                "sampling_seed": seed,
                "actions": chunk,
                "diagnostics": diagnostics,
                "decision": decision,
                "inference_s": latency,
                "selector_s": selection_latency,
                "executed_steps": 0,
            })
        predicted = chunk[chunk_index]
        action = [max(-1.0, min(1.0, x)) for x in predicted]
        clipped_values += sum(x != y for x, y in zip(predicted, action, strict=True))
        if action[6] > 0 and (previous_action is None or previous_action[6] <= 0):
            close_events += 1
        jump = norm(difference(action[:3], previous_action[:3])) if previous_action else None
        row = {
            "step": step, "sim_time_s": step / control_hz, "replanned": replanned,
            "model_calls": calls, "chunk_index": chunk_index,
            "chosen_k": decision["chosen_k"], "next_replan_step": next_replan_step,
            "threshold": decision["threshold"], "action": action,
            "eef": obs.eef, "gripper": obs.gripper,
            "inference_s": latency, "selector_s": selection_latency,
            "plan_revision_translation_l2": revision_l2,
            "action_translation_jump_l2": jump,
        }
        if step == 0 and isinstance(obs.payload, dict) and "instruction" in obs.payload:
            row["instruction"] = obs.payload["instruction"]
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
        predictions[-1]["executed_steps"] += 1
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
        record(obs.frame, {
            "step": steps, "sim_time_s": steps / control_hz, "replanned": False,
            "model_calls": calls, "chosen_k": decision["chosen_k"],
            "threshold": decision["threshold"], "next_replan_step": next_replan_step,
        }, success)
        video_s += clock() - tick

    def mean_present(field, only_replans=False):
        values = [r[field] for r in rows if r[field] is not None and (not only_replans or r["replanned"])]
        return sum(values) / len(values) if values else None

    horizons = [p["decision"]["chosen_k"] for p in predictions]
    summary = {
        **episode.asdict(), "episode_id": episode.id, "pair_id": episode.pair_id,
        "env_seed": episode.env_seed, "success": success, "steps": steps,
        "termination": "success" if success else "terminated" if terminal else "time_limit",
        "model_calls": calls, "sim_time_s": steps / control_hz,
        "inference_s": inference_s, "selector_s": selector_s, "simulation_s": simulation_s,
        "rollout_wall_s": elapsed, "reset_s": reset_s, "video_encode_s": video_s,
        "compute_s": inference_s + selector_s + simulation_s,
        "eef_path_m": path_m,
        "mean_plan_revision_translation_l2": mean_present("plan_revision_translation_l2"),
        "mean_replan_jump_translation_l2": mean_present("action_translation_jump_l2", True),
        "gripper_close_commands": close_events, "clipped_action_values": clipped_values,
        "mean_chosen_k": sum(horizons) / len(horizons),
        "min_chosen_k": min(horizons), "max_chosen_k": max(horizons),
        "mean_executed_prefix": steps / calls,
        "bootstrap_calls": sum(p["decision"]["bootstrap"] for p in predictions),
        **reset_info,
    }
    return summary, rows, predictions
