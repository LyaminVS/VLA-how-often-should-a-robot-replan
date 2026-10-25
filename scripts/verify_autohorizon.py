"""Independently replay AutoHorizon attention selection and frozen action execution.

The verifier does not import the production AutoHorizon selector. All decision
intermediates are reconstructed from the recorded attention matrix using the
pinned official float32 operations, including its mapped backward pointer.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from autohorizon_replan.config import AutoHorizonExperiment
from scripts.verify_dvac import (equal_number, finite, read_jsonl, recompute_variance,
                                 sha256, vector, verify_video)
from vla_replan.config import digest
from vla_replan.storage import atomic_json


UPSTREAM_COMMIT = "c7504f1756109103f2cfcc2e23f1b1a23841c885"
SELECTOR_VARIANT = "official_code_mapped_backward_join_v1"


def replay_attention(attention, *, horizon, execution_cap, selected_fm_step=3,
                     hold_thr=0.3, max_entropy_q=0.9):
    """Rebuild the published CPU-float32 rule without the collection selector.

    The backward count is measured in original coordinates. In addition to the
    count-sum condition, the upstream implementation requires a pointer join.
    A mathematical rewrite that omits either detail is a different scheduler.
    """
    import torch

    if (type(horizon) is not int or horizon < 1 or type(execution_cap) is not int
            or not 1 <= execution_cap <= horizon):
        raise ValueError("Invalid replay horizon or execution cap")
    if not isinstance(attention, list) or len(attention) != horizon:
        raise ValueError("Invalid attention matrix horizon")
    matrix = torch.tensor([vector(row, horizon, "attention row", 0) for row in attention],
                          dtype=torch.float32, device="cpu")
    masses = matrix.sum(dim=-1, keepdim=True)
    if (not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(masses).all())
            or bool((masses <= 0).any())):
        raise ValueError("Attention requires finite positive row mass in float32")
    normalized = matrix / (masses + 1e-12)
    if horizon == 1:
        diagnostic = {
            "normalized_attention": normalized.tolist(),
            "mu_forward": [0.0], "expected_index_forward": [0.0], "dmu_forward": [0.0],
            "entropy_forward": [0.0], "entropy_threshold_forward": 0.0,
            "reliable_forward": [True], "holds_forward": [True],
            "stop_forward": 0, "found_hold_forward": True, "N_forward": 1,
            "mu_backward": [0.0], "mu_backward_reversed": [0.0],
            "expected_index_backward_reversed": [0.0], "dmu_backward_reversed": [0.0],
            "entropy_backward_reversed": [0.0], "entropy_threshold_backward": 0.0,
            "reliable_backward_reversed": [True], "holds_backward_reversed": [True],
            "stop_backward_reversed": 0, "stop_backward": 0,
            "found_hold_backward": True, "N_backward": 1, "join_row": 0,
            "gap": [0.0], "full_coverage": True, "single_action_extension": True,
        }
        raw = 1
    else:
        directional = []
        for values in (normalized, torch.flip(normalized, dims=(0, 1))):
            values = values / (values.sum(-1, keepdim=True) + 1e-12)
            expected = (values * torch.arange(horizon, dtype=torch.float32)).sum(-1)
            pointer = torch.maximum(expected, torch.cummax(expected, dim=0).values)
            probabilities = values.clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(-1) / math.log(horizon)
            threshold = torch.quantile(entropy.float(), q=min(max_entropy_q, 0.999))
            reliable = entropy <= threshold
            increments = pointer - torch.cat([pointer.new_tensor([0.0]), pointer[:-1]])
            held = (increments < hold_thr) & reliable
            indices = torch.nonzero(held, as_tuple=False)
            stop = int(indices[0].item()) if len(indices) else horizon - 1
            directional.append((expected, pointer, entropy, threshold, reliable,
                                increments, held, stop, bool(len(indices))))
        fe, fp, fh, ft, fr, fd, fm, fs, ff = directional[0]
        be, bp, bh, bt, br, bd, bm, bs, bf = directional[1]
        mapped_backward = torch.flip(horizon - 1 - bp, dims=(0,))
        mapped_stop = horizon - 1 - bs
        forward_count = int(torch.clamp(torch.floor(fp[fs]) + 1, 1, horizon).item())
        backward_count = int(torch.clamp(
            torch.floor(mapped_backward[mapped_stop]) + 1, 1, horizon).item())
        gap = mapped_backward - fp
        joins = torch.nonzero(gap <= 1.0, as_tuple=False)
        join = int(joins[0].item()) if len(joins) else None
        full = forward_count + backward_count >= horizon and join is not None
        raw = horizon if full else forward_count
        diagnostic = {
            "normalized_attention": normalized.tolist(),
            "mu_forward": fp.tolist(), "expected_index_forward": fe.tolist(),
            "dmu_forward": fd.tolist(), "entropy_forward": fh.tolist(),
            "entropy_threshold_forward": float(ft.item()),
            "reliable_forward": fr.tolist(), "holds_forward": fm.tolist(),
            "stop_forward": fs, "found_hold_forward": ff, "N_forward": forward_count,
            "mu_backward": mapped_backward.tolist(), "mu_backward_reversed": bp.tolist(),
            "expected_index_backward_reversed": be.tolist(),
            "dmu_backward_reversed": bd.tolist(), "entropy_backward_reversed": bh.tolist(),
            "entropy_threshold_backward": float(bt.item()),
            "reliable_backward_reversed": br.tolist(), "holds_backward_reversed": bm.tolist(),
            "stop_backward_reversed": bs, "stop_backward": mapped_stop,
            "found_hold_backward": bf, "N_backward": backward_count,
            "join_row": join, "gap": gap.tolist(), "full_coverage": full,
            "single_action_extension": False,
        }
    return {
        "chosen_k": min(raw, execution_cap), "raw_chosen_k": raw,
        "execution_cap": execution_cap, "cap_applied": raw > execution_cap,
        "threshold": None, "first_crossing": None, "window_calls": 0,
        "window_mean": None, "window_std": None, "bootstrap": False,
        "method": "autohorizon", "method_id": f"autohorizon_cap_{execution_cap}",
        "selector_variant": SELECTOR_VARIANT, "upstream_commit": UPSTREAM_COMMIT,
        "selected_fm_step": selected_fm_step, "hold_thr": float(hold_thr),
        "max_entropy_q": float(max_entropy_q), "effective_entropy_q": min(max_entropy_q, 0.999),
        "run_len": 1, "selector_device": "cpu", "selector_dtype": "float32",
        "diagnostics": diagnostic,
    }


def compare_replay(actual, expected, label="decision"):
    """Check every intermediate and exact key inventory, including bool/int types."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Replay field inventory differs: {label}")
        for field, value in expected.items():
            compare_replay(actual[field], value, f"{label}.{field}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Replay vector length differs: {label}")
        for index, value in enumerate(expected):
            compare_replay(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        # Saved JSON floats reproduce the same float32 values. A small tolerance
        # permits platform reduction rounding in diagnostics; discrete choices
        # and all integer/bool indices must still agree exactly.
        equal_number(actual, expected, label, rel_tol=2e-6, abs_tol=2e-6)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Replay mismatch: {label}")


def validate_capture(diagnostic, cfg):
    """Replay layer averaging and validate the extraction identity."""
    import torch

    capture = diagnostic.get("attention_capture")
    if not isinstance(capture, dict):
        raise ValueError("Missing attention capture metadata")
    horizon = cfg.base.chunk_size
    step = cfg.autohorizon.selected_fm_step
    if (type(capture.get("selected_fm_step")) is not int
            or capture["selected_fm_step"] != step or not 1 <= step <= cfg.base.num_denoising_steps):
        raise ValueError("Attention captured at a different FM step")
    if type(capture.get("horizon")) is not int or capture["horizon"] != horizon:
        raise ValueError("Attention capture horizon differs")
    equal_number(capture.get("fm_time"), 1 - (step - 1) / cfg.base.num_denoising_steps,
                 "selected attention flow time", rel_tol=0, abs_tol=1e-6)
    if (capture.get("statistics_dtype") != "float32"
            or capture.get("extraction") != "self_attention_layers_only; mean_heads_then_layers_before_row_normalization"):
        raise ValueError("Attention extraction protocol differs")
    indices = capture.get("layer_indices")
    if (not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError("Invalid attention layer inventory")
    if type(capture.get("head_count")) is not int or capture["head_count"] < 1:
        raise ValueError("Invalid attention head count")
    layers = diagnostic.get("attention_layers")
    if not isinstance(layers, list) or len(layers) != len(indices):
        raise ValueError("Attention layer inventory differs from captured matrices")
    matrices = []
    for layer in layers:
        if not isinstance(layer, list) or len(layer) != horizon:
            raise ValueError("Invalid attention layer horizon")
        matrices.append([vector(row, horizon, "attention layer row", 0) for row in layer])
    layer_tensor = torch.tensor(matrices, dtype=torch.float32, device="cpu")
    if (not bool(torch.isfinite(layer_tensor).all()) or bool((layer_tensor.sum(-1) <= 0).any())
            or bool((layer_tensor.sum(-1) > 1.00001).any())):
        raise ValueError("Invalid action-attention probability mass")
    expected = layer_tensor.mean(dim=0).tolist()
    actual = diagnostic.get("attention")
    if not isinstance(actual, list) or len(actual) != horizon:
        raise ValueError("Invalid averaged attention horizon")
    for row, wanted in zip(actual, expected, strict=True):
        row = vector(row, horizon, "averaged attention row", 0)
        for value, target in zip(row, wanted, strict=True):
            equal_number(value, target, "attention layer mean", rel_tol=2e-6, abs_tol=2e-8)


def verify_episode(summary, rows, chunks, cfg):
    identity = summary["episode_id"]
    if not summary["method"].startswith("autohorizon_cap_"):
        raise ValueError("Unrecognized AutoHorizon method")
    declared_cap = int(summary["method"].rsplit("_", 1)[1])
    if (type(summary.get("execution_cap")) is not int
            or summary["execution_cap"] != declared_cap
            or not 1 <= declared_cap <= cfg.base.chunk_size
            or summary.get("k") is not None):
        raise ValueError("AutoHorizon identity disagrees with its execution cap")
    count = summary.get("steps")
    if type(count) is not int or not 1 <= count <= summary["max_steps"]:
        raise ValueError(f"Invalid episode length: {identity}")
    if [row.get("step") for row in rows] != list(range(count)) or not chunks:
        raise ValueError(f"Invalid step sequence or empty predictions: {identity}")
    if type(summary.get("model_calls")) is not int or summary["model_calls"] != len(chunks):
        raise ValueError(f"Invalid model call count: {identity}")
    horizon, tail = cfg.base.chunk_size, cfg.dvac.tail_length
    horizons, compact, next_step = [], [], 0
    raw_horizons, capped_calls = [], 0
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
        validate_capture(diagnostic, cfg)
        expected_decision = replay_attention(
            diagnostic.get("attention"), horizon=horizon, execution_cap=declared_cap,
            selected_fm_step=cfg.autohorizon.selected_fm_step,
            hold_thr=cfg.autohorizon.hold_thr, max_entropy_q=cfg.autohorizon.max_entropy_q,
        )
        raw_horizons.append(expected_decision["raw_chosen_k"])
        capped_calls += int(expected_decision["cap_applied"])
        decision = chunk.get("decision", {})
        compare_replay(decision, expected_decision, f"decision call {call_index}")
        k = expected_decision["chosen_k"]
        horizons.append(k)
        executed = min(k, count - next_step)
        if executed <= 0 or chunk.get("executed_steps") != executed:
            raise ValueError(f"Incorrect executed prefix length: {identity}")
        for field in ("inference_s", "selector_s"):
            finite(chunk.get(field), field)
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
            for field in ("raw_chosen_k", "execution_cap", "cap_applied"):
                if row.get(field) != decision[field] or type(row.get(field)) is not type(decision[field]):
                    raise ValueError(f"Invalid capped step metadata {field}: {identity}")
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
                          "diagnostics_finalize_s": diagnostic["diagnostics_finalize_s"],
                          "attention_capture": diagnostic["attention_capture"]})
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
            or type(summary.get("bootstrap_calls")) is not int or summary.get("bootstrap_calls") != 0):
        raise ValueError("Summary command or bootstrap counts disagree with evidence")
    equal_number(summary.get("mean_raw_chosen_k"), math.fsum(raw_horizons) / len(raw_horizons),
                 "mean raw chosen horizon")
    if type(summary.get("cap_applied_calls")) is not int or summary["cap_applied_calls"] != capped_calls:
        raise ValueError("Summary cap count disagrees with replay")
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
                     "executed_action_values": count * 7,
                     "attention_values": len(chunks) * horizon * horizon}


MATCH_FIELDS = (
    "model_id", "model_revision", "backbone_id", "backbone_revision", "chunk_size",
    "num_denoising_steps", "control_hz", "settle_steps", "precision", "record_video", "warmup_calls",
)


def first_prediction(chunk):
    return digest({"actions": chunk["actions"], "sampling_seed": chunk["sampling_seed"]})


def relocate_reference(audit, reference_root):
    """Allow relocation of one root while retaining its complete checksum inventory."""
    if not isinstance(audit, dict) or set(audit) != {"root", "files"}:
        raise ValueError("Missing AutoHorizon reference provenance")
    if not isinstance(audit["root"], str) or not Path(audit["root"]).is_absolute():
        raise ValueError("Recorded reference root must be absolute")
    root = Path(reference_root).resolve()
    files = audit["files"]
    if not isinstance(files, dict) or "manifest.json" not in files or "invariance.json" not in files:
        raise ValueError("Reference inventory omits manifest or invariance evidence")
    for relative, checksum in files.items():
        path = Path(relative)
        actual = (root / path).resolve()
        if (path.is_absolute() or ".." in path.parts or str(path) != relative
                or not actual.is_relative_to(root) or not actual.is_file()
                or not actual.stat().st_size or sha256(actual) != checksum):
            raise ValueError(f"Reference checksum or path mismatch: {relative}")
    return root, set(files)


def load_references(cfg, manifest, reference_root, *, decode_videos=False):
    """Validate all retained C=50 references, including their original DVAC decisions."""
    from capped_replan.config import CappedExperiment
    from scripts.verify_capped import verify_episode as verify_dvac_cap_episode

    root, declared = relocate_reference(manifest.get("reference_sources"), reference_root)
    old_manifest = json.loads((root / "manifest.json").read_text())
    old = CappedExperiment.fromdict(old_manifest["config"])
    if (old_manifest.get("experiment_kind") != "capped_dvac_comparison"
            or old_manifest.get("data_kind") != "real_libero"
            or old_manifest.get("config_fingerprint") != old.fingerprint):
        raise ValueError("Invalid retained DVAC manifest")
    if any(getattr(old.base, field) != getattr(cfg.base, field) for field in MATCH_FIELDS):
        raise ValueError("Retained references have different inference conditions")
    if (old_manifest.get("runtime") != manifest["runtime"]
            or old_manifest.get("hardware") != manifest["hardware"]):
        raise ValueError("Retained reference runtime or hardware differs")
    if (not isinstance(old_manifest.get("capped_source"), dict) or not old_manifest["capped_source"]
            or any(manifest["source_manifest"].get(key) != checksum
                   for key, checksum in old_manifest["capped_source"].items())):
        raise ValueError("Frozen rollout sources differ from retained references")
    if not isinstance(old_manifest.get("upstream_sampler"), dict) or not old_manifest["upstream_sampler"]:
        raise ValueError("Missing retained sampler source identity")
    if any(manifest["upstream_sampler"].get(key) != checksum
           for key, checksum in old_manifest.get("upstream_sampler", {}).items()):
        raise ValueError("Retained upstream sampler sources differ")
    if sha256(root / "invariance.json") != old_manifest.get("invariance_check_sha256"):
        raise ValueError("Retained invariance evidence checksum differs")
    old_check = json.loads((root / "invariance.json").read_text())
    if (old_check.get("passed") is not True or old_check.get("config_fingerprint") != old.fingerprint
            or old_check.get("source_manifest") != old_manifest.get("capped_source")
            or old_check.get("runtime") != old_manifest["runtime"]
            or old_check.get("hardware") != old_manifest["hardware"]
            or old_check.get("sampler_manifest") != old_manifest["upstream_sampler"]):
        raise ValueError("Retained invariance evidence does not match its manifest")
    required = {episode.pair_id for episode in cfg.episodes()}
    old_episodes = {episode.pair_id: episode for episode in old.episodes()
                    if episode.method == "dvac_cap_50" and episode.pair_id in required}
    if set(old_episodes) != required:
        raise ValueError("Retained DVAC references do not cover all requested settings")
    task_limits = {(task.suite, task.name): task.max_steps for task in cfg.base.tasks}
    expected_files = {"manifest.json", "invariance.json"}
    references, checks = {}, []
    manifest_sha = sha256(root / "manifest.json")
    for pair_id, episode in old_episodes.items():
        directory = root / "episodes" / episode.id
        summary = json.loads((directory / "summary.json").read_text())
        if (summary.get("episode_id") != episode.id or summary.get("pair_id") != pair_id
                or any(summary.get(key) != value or type(summary.get(key)) is not type(value)
                       for key, value in episode.asdict().items())
                or summary.get("max_steps") != task_limits.get((episode.suite, episode.task))
                or summary.get("config_fingerprint") != old.fingerprint
                or summary.get("manifest_sha256") != manifest_sha
                or summary.get("data_kind") != "real_libero"
                or summary.get("env_seed") != episode.env_seed):
            raise ValueError("Retained DVAC episode identity or configuration differs")
        expected_files.add(str((directory / "summary.json").relative_to(root)))
        filenames = {"steps.jsonl", "chunks.jsonl"} | ({"video.mp4"} if old.base.record_video else set())
        if set(summary.get("evidence_sha256", {})) != filenames:
            raise ValueError("Retained DVAC evidence inventory differs")
        for filename in filenames:
            relative = str((directory / filename).relative_to(root))
            expected_files.add(relative)
            if manifest["reference_sources"]["files"].get(relative) != summary["evidence_sha256"][filename]:
                raise ValueError("Retained evidence hash differs from its summary")
        rows = read_jsonl(directory / "steps.jsonl")
        chunks = read_jsonl(directory / "chunks.jsonl")
        _, verified = verify_dvac_cap_episode(summary, rows, chunks, old)
        references[pair_id] = {
            "initial_state_sha256": summary["initial_state_sha256"],
            "initial_observation_sha256": summary["initial_observation_sha256"],
            "first_prediction_sha256": first_prediction(chunks[0]),
            "first_clean_estimates_sha256": digest(chunks[0]["diagnostics"]["clean_estimates"]),
        }
        row = {"episode_id": episode.id, "summary_sha256": sha256(directory / "summary.json"),
               "files": summary["evidence_sha256"], **verified}
        if decode_videos and old.base.record_video:
            row["video"] = verify_video(directory / "video.mp4", summary["steps"], old.base.control_hz)
        checks.append(row)
    if declared != expected_files:
        raise ValueError("Retained reference provenance inventory is incomplete or contains unrelated evidence")
    return references, {"root": str(root), "recorded_root": manifest["reference_sources"]["root"],
                        "episodes": len(checks), "manifest_sha256": manifest_sha,
                        "trace_replay_passed": True, "videos_decoded": decode_videos, "evidence": checks}


def load_and_verify(root, *, decode_videos=False, check_sources=True, reference_roots=None):
    root = Path(root).resolve()
    reference_roots = reference_roots or [PROJECT / "outputs/capped"]
    if len(reference_roots) != 1:
        raise ValueError("AutoHorizon expects one retained C=50 reference root")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = AutoHorizonExperiment.fromdict(manifest["config"])
    if (manifest.get("config_fingerprint") != cfg.fingerprint
            or manifest.get("data_kind") != "real_libero"
            or manifest.get("experiment_kind") != "autohorizon_comparison"):
        raise ValueError("Invalid AutoHorizon manifest")
    expected = {episode.id: episode for episode in cfg.episodes()}
    if manifest.get("planned_episode_ids") != list(expected):
        raise ValueError("Final AutoHorizon report requires the complete configuration plan")
    if "full_plan_episode_ids" in manifest and manifest["full_plan_episode_ids"] != list(expected):
        raise ValueError("AutoHorizon full plan differs from configuration")
    actual = {path.name for path in (root / "episodes").iterdir() if path.is_dir()}
    if actual != set(expected):
        raise ValueError(f"Incomplete or unexpected AutoHorizon inventory: {len(actual)}/{len(expected)}")
    for field in ("runtime", "hardware", "source_manifest", "upstream_sampler"):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise ValueError(f"Missing collection metadata: {field}")
    invariance_path = root / "invariance.json"
    if sha256(invariance_path) != manifest.get("invariance_check_sha256"):
        raise ValueError("AutoHorizon invariance checksum mismatch")
    invariance = json.loads(invariance_path.read_text())
    checks = invariance.get("checks", [])
    expected_tasks = {(task.suite, task.name) for task in cfg.base.tasks}
    if (invariance.get("passed") is not True or invariance.get("check_kind") != "autohorizon_invariance"
            or invariance.get("robot_actions_executed") != 0
            or invariance.get("config") != cfg.asdict()
            or invariance.get("reference_sources") != manifest.get("reference_sources")
            or invariance.get("config_fingerprint") != cfg.fingerprint
            or invariance.get("source_manifest") != manifest["source_manifest"]
            or invariance.get("sampler_manifest") != manifest["upstream_sampler"]
            or invariance.get("runtime") != manifest["runtime"]
            or invariance.get("hardware") != manifest["hardware"]
            or len(checks) != len(expected_tasks)
            or {(item.get("suite"), item.get("task")) for item in checks} != expected_tasks
            or any(item.get("exact_action_equality") is not True
                   or item.get("max_absolute_action_error") != 0
                   or item.get("attention_shape") != [cfg.base.chunk_size, cfg.base.chunk_size]
                   or item.get("initial_state") != 21
                   or item.get("sampling_seed") != 101
                   or type(item.get("raw_chosen_k")) is not int
                   or not 1 <= item["raw_chosen_k"] <= cfg.base.chunk_size
                   for item in checks)):
        raise ValueError("AutoHorizon invariance evidence does not match the recorded setup")
    if check_sources:
        for relative, checksum in manifest["source_manifest"].items():
            path = Path(relative)
            source = (PROJECT / path).resolve()
            if (path.is_absolute() or ".." in path.parts or not source.is_relative_to(PROJECT)
                    or not source.is_file() or sha256(source) != checksum):
                raise ValueError(f"Collection source checksum mismatch: {relative}")
    references, reference_report = load_references(
        cfg, manifest, reference_roots[0], decode_videos=decode_videos)
    capture_protocol = {(item["suite"], item["task"]): item.get("attention_capture") for item in checks}
    first_diagnostics, starts = defaultdict(set), defaultdict(set)
    records, checked, totals = [], [], defaultdict(int)
    max_variance_error = 0.0
    manifest_sha = sha256(manifest_path)
    for identity, episode in expected.items():
        directory = root / "episodes" / identity
        summary = json.loads((directory / "summary.json").read_text())
        if (summary.get("episode_id") != identity or summary.get("pair_id") != episode.pair_id
                or any(summary.get(key) != value or type(summary.get(key)) is not type(value)
                       for key, value in episode.asdict().items())):
            raise ValueError(f"AutoHorizon episode identity mismatch: {identity}")
        if (summary.get("config_fingerprint") != cfg.fingerprint
                or summary.get("manifest_sha256") != manifest_sha
                or summary.get("data_kind") != "real_libero" or summary.get("env_seed") != episode.env_seed):
            raise ValueError("AutoHorizon episode does not match its manifest")
        reference = references[episode.pair_id]
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            if summary.get(field) != reference[field]:
                raise ValueError("AutoHorizon initial conditions differ from retained controls")
            starts[(episode.suite, episode.task, episode.state, field)].add(summary[field])
        if (summary.get("reference_pair_id") != episode.pair_id
                or summary.get("reference_initial_state_match") is not True
                or summary.get("reference_first_prediction_equal") is not True
                or summary.get("reference_max_absolute_action_error") != 0):
            raise ValueError("Missing per-episode historical reference gate")
        filenames = ["steps.jsonl", "chunks.jsonl"] + (["video.mp4"] if cfg.base.record_video else [])
        hashes = summary.get("evidence_sha256", {})
        if set(hashes) != set(filenames):
            raise ValueError("AutoHorizon evidence inventory differs")
        for filename in filenames:
            path = directory / filename
            if not path.is_file() or not path.stat().st_size or sha256(path) != hashes[filename]:
                raise ValueError(f"AutoHorizon evidence checksum mismatch: {path}")
        rows, chunks = read_jsonl(directory / "steps.jsonl"), read_jsonl(directory / "chunks.jsonl")
        compact, result = verify_episode(summary, rows, chunks, cfg)
        if first_prediction(chunks[0]) != reference["first_prediction_sha256"]:
            raise ValueError("AutoHorizon first prediction differs from retained controls")
        if digest(chunks[0]["diagnostics"]["clean_estimates"]) != reference["first_clean_estimates_sha256"]:
            raise ValueError("AutoHorizon first FM trajectory differs from retained controls")
        for chunk in chunks:
            if chunk["diagnostics"]["attention_capture"] != capture_protocol[(episode.suite, episode.task)]:
                raise ValueError("Attention capture metadata changed after the invariance check")
        first_diagnostics[episode.pair_id].add(digest({
            "clean_estimates": chunks[0]["diagnostics"]["clean_estimates"],
            "attention": chunks[0]["diagnostics"]["attention"],
            "attention_layers": chunks[0]["diagnostics"]["attention_layers"],
        }))
        item = {"episode_id": identity, "summary_sha256": sha256(directory / "summary.json"),
                "files": hashes, **result}
        if decode_videos and cfg.base.record_video:
            item["video"] = verify_video(directory / "video.mp4", summary["steps"], cfg.base.control_hz)
        for key in ("steps", "model_calls", "clean_estimate_values", "executed_action_values", "attention_values"):
            totals[key] += result[key]
        max_variance_error = max(max_variance_error, result["max_variance_error"])
        checked.append(item)
        records.append({"summary": summary, "chunks": compact,
                        "origin": "autohorizon", "raw_source_root": str(root)})
    if any(len(values) != 1 for values in starts.values()) or any(len(values) != 1 for values in first_diagnostics.values()):
        raise ValueError("AutoHorizon caps disagree on initial conditions or first diagnostic capture")
    report = {
        "complete": True, "episodes": len(records), "autohorizon_episodes": len(records),
        "planned_autohorizon_episodes": len(expected),
        "matched_task_state_seed_blocks": len(first_diagnostics),
        "physical_task_state_settings": len(starts) // 2,
        "manifest_sha256": manifest_sha, "config_fingerprint": cfg.fingerprint,
        "invariance_sha256": sha256(invariance_path), "verifier_source_sha256": sha256(__file__),
        "verification_dependency_sha256": {name: sha256(PROJECT / name) for name in (
            "scripts/verify_dvac.py", "scripts/verify_capped.py")},
        "collection_source_checks": check_sources, "videos_decoded": decode_videos,
        "first_prediction_equality_all_methods": True,
        "first_fm_trajectory_equality_to_retained_dvac": True,
        "max_variance_error": max_variance_error, "totals": dict(totals),
        "selector_variant": SELECTOR_VARIANT, "upstream_commit": UPSTREAM_COMMIT,
        "historical_controls": reference_report, "evidence": checked,
        "timing_note": "AutoHorizon captures attention in addition to the retained FM diagnostics. Model-call counts are comparable, but one call has different diagnostic overhead. Its measured inference time includes attention capture; cross-family wall times are not an equal-instrumentation comparison.",
    }
    return cfg, records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference-roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-videos", action="store_true")
    args = parser.parse_args()
    _, _, result = load_and_verify(args.input, decode_videos=args.decode_videos,
                                  reference_roots=args.reference_roots)
    if args.output:
        protected = [args.input.resolve()] + [Path(root).resolve() for root in (
            args.reference_roots or [PROJECT / "outputs/capped"])]
        if any(args.output.resolve().is_relative_to(root) for root in protected):
            raise ValueError("Verification output must be outside immutable raw evidence")
        atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("evidence", "historical_controls")}, indent=2))


if __name__ == "__main__":
    main()
