"""Independently replay DVAC decisions and action execution from immutable evidence."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from adaptive_replan.config import AdaptiveExperiment, DVACSettings
from vla_replan.config import Experiment, Task, digest
from vla_replan.storage import atomic_json


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    result = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            result.append(item)
    return result


def finite(value, label, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid numeric value: {label}")
    return float(value)


def vector(value, length, label, minimum=-math.inf):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Invalid vector length: {label}")
    return [finite(item, label, minimum) for item in value]


def equal_number(actual, expected, label, *, rel_tol=1e-12, abs_tol=1e-14):
    if expected is None:
        if actual is not None:
            raise ValueError(f"Expected null: {label}")
    elif not math.isclose(finite(actual, label, -math.inf), expected,
                          rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(f"Numeric replay mismatch: {label}, {actual!r} != {expected!r}")


def configuration(data):
    if not isinstance(data, dict) or set(data) != {"base", "methods", "dvac"}:
        raise ValueError("Invalid adaptive configuration structure")
    base = dict(data["base"])
    base["tasks"] = tuple(Task(**task) for task in base["tasks"])
    for name in ("intervals", "initial_states", "policy_seeds"):
        base[name] = tuple(base[name])
    cfg = AdaptiveExperiment(Experiment(**base), tuple(data["methods"]), DVACSettings(**data["dvac"]))
    cfg.validate()
    return cfg


def verify_inherited_sources(runtime, project=PROJECT):
    """Match inherited backend and dependency lock without importing its runtime."""
    project = Path(project)
    files = sorted((project / "vla_replan").glob("*.py"))
    if not files:
        raise ValueError("Missing inherited vla_replan source files")
    combined = hashlib.sha256()
    for path in files:
        combined.update(path.name.encode())
        combined.update(path.read_bytes())
    lock = project / "requirements/linux.lock.txt"
    if not lock.is_file():
        raise ValueError("Missing inherited dependency lock")
    observed = {"code_sha256": combined.hexdigest(), "dependency_lock_sha256": sha256(lock)}
    for field, value in observed.items():
        if runtime.get(field) != value:
            raise ValueError(f"Inherited collection source checksum mismatch: {field}")
    return observed


def recompute_variance(estimates, *, tail_length=5, horizon=50, dimensions=7):
    """Replay population variance without importing the runtime implementation."""
    if not isinstance(estimates, list) or len(estimates) != tail_length:
        raise ValueError("Incorrect number of clean-action estimates")
    for estimate in estimates:
        if not isinstance(estimate, list) or len(estimate) != horizon:
            raise ValueError("Incorrect clean-action horizon")
        for action in estimate:
            vector(action, dimensions, "clean-action estimate")
    result = []
    for action_index in range(horizon):
        coordinate_variances = []
        for coordinate in range(dimensions):
            samples = [estimate[action_index][coordinate] for estimate in estimates]
            center = math.fsum(samples) / tail_length
            coordinate_variances.append(math.fsum((sample - center) ** 2 for sample in samples) / tail_length)
        result.append(math.fsum(coordinate_variances))
    return result


def replay_decision(variance, history, *, alpha=2, minimum=1):
    """Select a prefix using only preceding calls, then update a caller-owned history."""
    if not history:
        return {"chosen_k": minimum, "threshold": None, "first_crossing": None,
                "window_calls": 0, "window_mean": None, "window_std": None, "bootstrap": True}
    previous = [value for call in history for value in call]
    average = math.fsum(previous) / len(previous)
    deviation = math.sqrt(math.fsum((value - average) ** 2 for value in previous) / len(previous))
    threshold = average + alpha * deviation
    crossing = next((index for index, value in enumerate(variance) if value > threshold), None)
    return {"chosen_k": len(variance) if crossing is None else max(minimum, crossing),
            "threshold": threshold, "first_crossing": crossing, "window_calls": len(history),
            "window_mean": average, "window_std": deviation, "bootstrap": False}


def verify_episode(summary, rows, chunks, cfg):
    identity = summary["episode_id"]
    count = summary.get("steps")
    if type(count) is not int or not 1 <= count <= summary["max_steps"]:
        raise ValueError(f"Invalid episode length: {identity}")
    if [row.get("step") for row in rows] != list(range(count)) or not chunks:
        raise ValueError(f"Invalid step sequence or empty predictions: {identity}")
    if type(summary.get("model_calls")) is not int or summary["model_calls"] != len(chunks):
        raise ValueError(f"Invalid model call count: {identity}")
    horizon, tail = cfg.base.chunk_size, cfg.dvac.tail_length
    history, horizons, compact, next_step = [], [], [], 0
    max_variance_error = 0.0
    clipped = closes = 0
    previous_action = None
    for call_index, chunk in enumerate(chunks):
        if chunk.get("step") != next_step:
            raise ValueError(f"Invalid adaptive replan boundary: {identity}")
        expected_seed = (summary["policy_seed"] * 1_000_003 + next_step * 97_409) % (2**63 - 1)
        if chunk.get("sampling_seed") != expected_seed:
            raise ValueError(f"Incorrect sampling seed: {identity}")
        actions = chunk.get("actions")
        if not isinstance(actions, list) or len(actions) != horizon:
            raise ValueError(f"Incorrect prediction horizon: {identity}")
        for action in actions:
            vector(action, 7, "predicted action")
        diagnostic = chunk.get("diagnostics", {})
        calculated = recompute_variance(diagnostic.get("clean_estimates"), tail_length=tail, horizon=horizon)
        logged = vector(diagnostic.get("variance"), horizon, "logged variance", 0)
        for actual, expected in zip(logged, calculated, strict=True):
            equal_number(actual, expected, "denoising variance")
            max_variance_error = max(max_variance_error, abs(actual - expected))
        times = vector(diagnostic.get("fm_times"), tail, "flow times")
        expected_times = [1 - index / cfg.base.num_denoising_steps
                          for index in range(cfg.base.num_denoising_steps - tail, cfg.base.num_denoising_steps)]
        for actual, expected in zip(times, expected_times, strict=True):
            equal_number(actual, expected, "flow-time schedule", rel_tol=0, abs_tol=1e-6)
        if (diagnostic.get("velocity_evaluations") != cfg.base.num_denoising_steps
                or diagnostic.get("action_dimensions") != 7
                or diagnostic.get("statistics_dtype") != "float64"):
            raise ValueError("Invalid sampler diagnostic metadata")
        finite(diagnostic.get("diagnostics_finalize_s"), "diagnostics_finalize_s")
        if summary["method"] == "dvac":
            expected_decision = replay_decision(calculated, history, alpha=cfg.dvac.alpha, minimum=cfg.dvac.min_steps)
            history.append(calculated)
            history = history[-cfg.dvac.window_size:]
        else:
            expected_decision = {"chosen_k": summary["k"], "threshold": None, "first_crossing": None,
                                 "window_calls": 0, "window_mean": None, "window_std": None, "bootstrap": False}
        decision = chunk.get("decision", {})
        for field in ("chosen_k", "first_crossing", "window_calls", "bootstrap"):
            if decision.get(field) != expected_decision[field] or type(decision.get(field)) is not type(expected_decision[field]):
                raise ValueError(f"Decision replay mismatch for {field}: {identity}, call {call_index}")
        for field in ("threshold", "window_mean", "window_std"):
            equal_number(decision.get(field), expected_decision[field], field)
        k = expected_decision["chosen_k"]
        horizons.append(k)
        executed = min(k, count - next_step)
        if executed <= 0 or chunk.get("executed_steps") != executed:
            raise ValueError(f"Incorrect executed prefix length: {identity}")
        for field in ("inference_s", "selector_s"):
            finite(chunk.get(field), field)
        if summary["method"] != "dvac" and chunk["selector_s"] != 0:
            raise ValueError("Fixed methods should not run the DVAC selector")
        for offset in range(executed):
            step = next_step + offset
            row = rows[step]
            predicted = actions[offset]
            expected_action = [max(-1.0, min(1.0, value)) for value in predicted]
            actual = vector(row.get("action"), 7, "executed action")
            if actual != expected_action:
                raise ValueError(f"Executed action differs from final predicted prefix: {identity}, step {step}")
            for field, expected in (("chunk_index", offset), ("model_calls", call_index + 1),
                                    ("chosen_k", k), ("next_replan_step", next_step + k)):
                if row.get(field) != expected or type(row.get(field)) is not int:
                    raise ValueError(f"Invalid step {field}: {identity}")
            if row.get("replanned") is not (offset == 0):
                raise ValueError(f"Invalid replanning marker: {identity}")
            equal_number(row.get("sim_time_s"), step / cfg.base.control_hz, "simulation timestamp")
            equal_number(row.get("threshold"), decision["threshold"], "step threshold")
            for field in ("inference_s", "selector_s"):
                equal_number(row.get(field), chunk[field] if offset == 0 else 0.0, field)
            if type(row.get("success_after_action")) is not bool:
                raise ValueError("Missing boolean post-action success")
            if row["success_after_action"] and step != count - 1:
                raise ValueError("Actions continued after task completion")
            clipped += sum(left != right for left, right in zip(predicted, expected_action, strict=True))
            closes += int(actual[6] > 0 and (previous_action is None or previous_action[6] <= 0))
            previous_action = actual
        compact.append({key: value for key, value in chunk.items() if key not in ("actions", "diagnostics")}
                       | {"variance": calculated, "fm_times": times,
                          "diagnostics_finalize_s": diagnostic["diagnostics_finalize_s"]})
        next_step += executed
    if next_step != count:
        raise ValueError("Predictions do not cover all executed actions")
    for field in ("inference_s", "selector_s"):
        equal_number(summary.get(field), math.fsum(chunk[field] for chunk in chunks), "total " + field, abs_tol=1e-8)
    for field in ("simulation_s", "compute_s", "episode_wall_s", "sim_time_s", "eef_path_m"):
        finite(summary.get(field), field)
    equal_number(summary["compute_s"], summary["inference_s"] + summary["selector_s"] + summary["simulation_s"], "compute total", abs_tol=1e-8)
    equal_number(summary["sim_time_s"], count / cfg.base.control_hz, "episode simulation duration")
    for field, expected in (("mean_chosen_k", math.fsum(horizons) / len(horizons)),
                            ("min_chosen_k", min(horizons)), ("max_chosen_k", max(horizons)),
                            ("mean_executed_prefix", count / len(chunks))):
        equal_number(summary.get(field), expected, field)
    if (summary.get("clipped_action_values") != clipped or summary.get("gripper_close_commands") != closes
            or summary.get("bootstrap_calls") != int(summary["method"] == "dvac")):
        raise ValueError("Summary command or bootstrap counts disagree with evidence")
    success = any(row["success_after_action"] for row in rows)
    if type(summary.get("success")) is not bool or summary["success"] != success:
        raise ValueError("Summary success disagrees with executed action outcomes")
    if success and summary.get("termination") != "success":
        raise ValueError("Incorrect successful termination")
    if not success and summary.get("termination") not in ("terminated", "time_limit"):
        raise ValueError("Incorrect failed termination")
    if summary["termination"] == "time_limit" and count != summary["max_steps"]:
        raise ValueError("Time limit was reported before the configured horizon")
    return compact, {"steps": count, "model_calls": len(chunks), "max_variance_error": max_variance_error,
                     "clean_estimate_values": len(chunks) * tail * horizon * 7,
                     "executed_action_values": count * 7}


def verify_video(path, steps, hz):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), format="ffmpeg", input_params=["-threads", "1"])
    count, shape = 0, None
    try:
        metadata = reader.get_meta_data()
        equal_number(metadata.get("fps"), hz, "video frame rate", abs_tol=1e-6)
        for frame in reader:
            if shape is None:
                shape = list(frame.shape)
            if list(frame.shape) != shape:
                raise ValueError("Video frame dimensions changed")
            count += 1
    finally:
        reader.close()
    if count != steps + hz:
        raise ValueError(f"Incorrect video frame count: {path}, {count} != {steps + hz}")
    return {"decoded_frames": count, "shape": shape, "fps": hz}


def load_and_verify(root, *, decode_videos=False, check_sources=True):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = configuration(manifest["config"])
    if manifest.get("config_fingerprint") != cfg.fingerprint:
        raise ValueError("Invalid configuration fingerprint")
    if manifest.get("data_kind") != "real_libero" or manifest.get("experiment_kind") != "dvac_comparison":
        raise ValueError("Only recorded LIBERO DVAC comparison data are accepted")
    expected = {episode.id: episode for episode in cfg.episodes()}
    if manifest.get("planned_episode_ids") != list(expected):
        raise ValueError("Manifest planned episode IDs disagree with configuration")
    actual = {path.name for path in (root / "episodes").iterdir() if path.is_dir()}
    if actual != set(expected):
        raise ValueError(f"Incomplete or unexpected episode inventory: {len(actual)}/{len(expected)}")
    for field in ("runtime", "hardware", "adaptive_source", "upstream_sampler"):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise ValueError(f"Missing run metadata: {field}")
    invariance_path = root / "invariance.json"
    if not invariance_path.is_file() or sha256(invariance_path) != manifest.get("invariance_check_sha256"):
        raise ValueError("Invariance-check evidence checksum mismatch")
    invariance = json.loads(invariance_path.read_text())
    checks = invariance.get("checks", [])
    expected_tasks = {(task.suite, task.name) for task in cfg.base.tasks}
    if (invariance.get("passed") is not True or invariance.get("config_fingerprint") != cfg.fingerprint
            or invariance.get("source_manifest") != manifest["adaptive_source"]
            or invariance.get("sampler_manifest") != manifest["upstream_sampler"]
            or invariance.get("runtime") != manifest["runtime"]
            or invariance.get("hardware") != manifest["hardware"]
            or len(checks) != len(expected_tasks)
            or {(item.get("suite"), item.get("task")) for item in checks} != expected_tasks
            or any(item.get("exact_action_equality") is not True
                   or item.get("max_absolute_action_error") != 0 for item in checks)):
        raise ValueError("Invariance checks do not prove the recorded collection setup")
    inherited_sources = None
    if check_sources:
        inherited_sources = verify_inherited_sources(manifest["runtime"])
        sources = manifest["adaptive_source"]
        sources = sources.get("files", sources)
        for relative, checksum in sources.items():
            source = (PROJECT / relative).resolve()
            if not source.is_relative_to(PROJECT) or not source.is_file() or sha256(source) != checksum:
                raise ValueError(f"Collection source checksum mismatch: {relative}")
    starts, first_predictions = defaultdict(set), defaultdict(list)
    records, checked, totals = [], [], defaultdict(int)
    max_variance_error = 0.0
    for identity, episode in expected.items():
        directory = root / "episodes" / identity
        summary = json.loads((directory / "summary.json").read_text())
        if (summary.get("episode_id") != identity or summary.get("pair_id") != episode.pair_id
                or any(summary.get(key) != value for key, value in episode.asdict().items())):
            raise ValueError(f"Summary identity mismatch: {identity}")
        if (summary.get("config_fingerprint") != cfg.fingerprint
                or summary.get("manifest_sha256") != sha256(manifest_path)
                or summary.get("data_kind") != "real_libero"):
            raise ValueError("Episode configuration does not match manifest")
        if summary.get("env_seed") != episode.env_seed:
            raise ValueError("Incorrect environment seed")
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            value = summary.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Missing physical start hash: {field}")
            starts[(episode.suite, episode.task, episode.state, field)].add(value)
        filenames = ["steps.jsonl", "chunks.jsonl"] + (["video.mp4"] if cfg.base.record_video else [])
        hashes = summary.get("evidence_sha256", {})
        if set(hashes) != set(filenames):
            raise ValueError("Episode evidence hash inventory mismatch")
        for filename in filenames:
            path = directory / filename
            if not path.is_file() or not path.stat().st_size or sha256(path) != hashes[filename]:
                raise ValueError(f"Episode evidence checksum mismatch: {path}")
        rows = read_jsonl(directory / "steps.jsonl")
        chunks = read_jsonl(directory / "chunks.jsonl")
        compact, result = verify_episode(summary, rows, chunks, cfg)
        first_predictions[episode.pair_id].append(digest({
            "actions": chunks[0]["actions"], "sampling_seed": chunks[0]["sampling_seed"],
            "clean_estimates": chunks[0]["diagnostics"]["clean_estimates"],
        }))
        record = {"episode_id": identity, "summary_sha256": sha256(directory / "summary.json"),
                  "files": hashes, **result}
        if decode_videos and cfg.base.record_video:
            record["video"] = verify_video(directory / "video.mp4", summary["steps"], cfg.base.control_hz)
        for key in ("steps", "model_calls", "clean_estimate_values", "executed_action_values"):
            totals[key] += result[key]
        max_variance_error = max(max_variance_error, result["max_variance_error"])
        checked.append(record)
        records.append({"summary": summary, "chunks": compact})
    if any(len(values) != 1 for values in starts.values()):
        raise ValueError("Initial states or observations differ across matched methods and policy seeds")
    if any(len(values) != len(cfg.methods) or len(set(values)) != 1 for values in first_predictions.values()):
        raise ValueError("First sampled action chunks differ across paired methods")
    report = {
        "complete": True, "episodes": len(records), "planned_episodes": len(expected),
        "matched_task_state_seed_blocks": len(first_predictions),
        "physical_task_state_settings": len(starts) // 2,
        "manifest_sha256": sha256(manifest_path), "config_fingerprint": cfg.fingerprint,
        "invariance_sha256": sha256(invariance_path), "verifier_source_sha256": sha256(__file__),
        "collection_source_checks": check_sources, "videos_decoded": decode_videos,
        "inherited_source_sha256": inherited_sources,
        "first_prediction_equality": True, "max_variance_error": max_variance_error,
        "totals": dict(totals), "evidence": checked,
    }
    return cfg, records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-videos", action="store_true")
    args = parser.parse_args()
    _, _, result = load_and_verify(args.input, decode_videos=args.decode_videos)
    if args.output:
        if args.output.resolve().is_relative_to(args.input.resolve()):
            raise ValueError("Verification output must be outside immutable raw evidence")
        atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "evidence"}, indent=2))


if __name__ == "__main__":
    main()
