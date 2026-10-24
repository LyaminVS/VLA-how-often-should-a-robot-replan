"""Validate and analyze a complete moving-target stress test without altering raw data.

The initial state is the bootstrap cluster: all policy seeds, K values and motion
conditions stay together. Contact is a simulator contact event, never a grasp label.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moving_target.config import CONDITIONS, MovingExperiment
from vla_replan.analysis import write_csv
from vla_replan.config import digest
from vla_replan.storage import atomic_json

DATA_KIND = "moving_target_stress_test"
METRICS = ("success", "model_calls", "compute_s", "inference_s", "episode_wall_s", "sim_time_s")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, label, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid finite metric: {label}")
    return float(value)


def vector(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected a three-dimensional vector: {label}")
    return [finite(item, label, -math.inf) for item in value]


def mean(values):
    return sum(values) / len(values) if values else None


def read_jsonl(path):
    records = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL record: {path}:{number}")
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected a JSON object: {path}:{number}")
            records.append(item)
    return records


def motion_metrics(summary, rows, control_hz, force_offset_z_m):
    """Measure observed trajectories; never substitute commanded motion for motion achieved."""
    trajectory = []
    active = []
    contacts = []
    for row in rows:
        motion = row.get("motion")
        if not isinstance(motion, dict):
            raise ValueError("Every step needs motion telemetry")
        if "condition" in summary and motion.get("condition") != summary["condition"]:
            raise ValueError("Motion telemetry uses a different condition")
        position = vector(motion.get("object_pos"), "motion.object_pos")
        vector(motion.get("target_pos"), "motion.target_pos")
        finite(motion.get("elapsed_sim_time_s"), "motion.elapsed_sim_time_s")
        if type(motion.get("motion_active")) is not bool:
            raise ValueError("motion_active must be boolean")
        finite(motion.get("tracking_error_m"), "motion.tracking_error_m")
        finite(motion.get("horizontal_force_n"), "motion.horizontal_force_n", -math.inf)
        torque = finite(motion.get("applied_torque_y_nm"), "motion.applied_torque_y_nm", -math.inf)
        if not math.isclose(torque, force_offset_z_m * motion["horizontal_force_n"], abs_tol=1e-9):
            raise ValueError("Applied torque does not match the configured force application point")
        if type(motion.get("force_saturated")) is not bool:
            raise ValueError("force_saturated must be boolean")
        contact = motion.get("first_contact_step")
        contact_time = motion.get("first_contact_sim_time")
        if (contact is None) != (contact_time is None):
            raise ValueError("Contact step and time must both be known or both be null")
        if contact is not None:
            if type(contact) is not int or not 0 <= contact <= summary["steps"]:
                raise ValueError("Invalid first contact step")
            finite(contact_time, "motion.first_contact_sim_time")
            contacts.append((contact, float(contact_time)))
            if motion["motion_active"] or abs(motion["horizontal_force_n"]) > 1e-9:
                raise ValueError("Motion force must stay disabled after first contact")
        trajectory.append((row["step"], position, motion))
        if motion["motion_active"]:
            active.append(motion)
    final = summary.get("motion_final")
    if not isinstance(final, dict):
        raise ValueError("Summary needs final motion telemetry")
    if "condition" in summary and final.get("condition") != summary["condition"]:
        raise ValueError("Final telemetry uses a different condition")
    vector(final.get("object_pos"), "motion_final.object_pos")
    contact, contact_time = final.get("first_contact_step"), final.get("first_contact_sim_time")
    if (contact is None) != (contact_time is None):
        raise ValueError("Final contact step and time must both be known or null")
    if contact is not None:
        if type(contact) is not int or not 0 <= contact <= summary["steps"]:
            raise ValueError("Invalid final contact step")
        finite(contact_time, "motion_final.first_contact_sim_time")
        if contact_time > summary["sim_time_s"] + 1 / control_hz + 1e-6:
            raise ValueError("Contact time lies after episode termination")
        contacts.append((contact, float(contact_time)))
    if contacts and len(set(contacts)) != 1:
        raise ValueError("The latched first contact changed during an episode")
    if contacts and contact is None:
        raise ValueError("Final telemetry lost a recorded contact")
    if type(final.get("motion_active")) is not bool:
        raise ValueError("Final motion_active must be boolean")
    if contact is not None and final["motion_active"]:
        raise ValueError("Motion cannot remain active after final recorded contact")
    final_force = finite(final.get("horizontal_force_n"), "motion_final.horizontal_force_n", -math.inf)
    final_torque = finite(final.get("applied_torque_y_nm"), "motion_final.applied_torque_y_nm", -math.inf)
    if not math.isclose(final_torque, force_offset_z_m * final_force, abs_tol=1e-9):
        raise ValueError("Final torque does not match the configured force application point")
    if contact is not None and (abs(final_force) > 1e-9 or abs(final_torque) > 1e-9):
        raise ValueError("Final force and moment must be zero after contact")
    for field in ("active_sim_time_s", "peak_force_n", "max_force_after_contact_n",
                  "precontact_x_range_m", "precontact_y_range_m", "precontact_z_range_m",
                  "max_precontact_tilt_deg"):
        finite(final.get(field), f"motion_final.{field}")
    if final["max_precontact_tilt_deg"] > 180 + 1e-6:
        raise ValueError("Invalid pre-contact tilt angle")
    if final["max_force_after_contact_n"] > 1e-9:
        raise ValueError("Nonzero intervention force was applied after contact")
    physics_samples = final.get("active_physics_samples")
    if type(physics_samples) is not int or physics_samples < 0:
        raise ValueError("Invalid active physics sample count")
    for field in ("tracking_mae_m", "tracking_rmse_m", "force_saturation_fraction"):
        if physics_samples:
            finite(final.get(field), f"motion_final.{field}")
        elif final.get(field) is not None:
            raise ValueError("Inactive episodes must not invent active force diagnostics")
    if physics_samples and final["force_saturation_fraction"] > 1:
        raise ValueError("Invalid saturation fraction")
    # Pre-contact trajectory includes the non-actuated delay. This common window
    # is meaningful for static as well as moving conditions, unlike active-only time.
    precontact = [item for item in trajectory if contact_time is None
                  or item[2]["elapsed_sim_time_s"] < contact_time]
    positions = [position for _, position, _ in precontact]
    if contact is None:
        positions.append(final["object_pos"])
    speeds, path = [], 0.0
    for first, second in zip(positions, positions[1:]):
        distance = math.hypot(second[0] - first[0], second[1] - first[1])
        path += distance
        speeds.append(distance * control_hz)
    return {
        "contact": contact is not None,
        "first_contact_step": contact,
        "first_contact_sim_time_s": contact_time,
        "precontact_samples": len(positions),
        "precontact_x_range_m": final["precontact_x_range_m"],
        "precontact_y_range_m": final["precontact_y_range_m"],
        "precontact_z_range_m": final["precontact_z_range_m"],
        "max_precontact_tilt_deg": final["max_precontact_tilt_deg"],
        "precontact_horizontal_path_m": path if len(positions) >= 2 else None,
        "precontact_mean_horizontal_speed_m_s": mean(speeds),
        "active_samples": len(active),
        "active_physics_samples": physics_samples,
        "active_duration_s": final["active_sim_time_s"],
        "active_mean_tracking_error_m": final["tracking_mae_m"],
        "active_tracking_rmse_m": final["tracking_rmse_m"],
        "active_max_tracking_error_m": max((item["tracking_error_m"] for item in active), default=None),
        "active_mean_horizontal_force_n": mean([abs(item["horizontal_force_n"]) for item in active]),
        "active_saturation_fraction": final["force_saturation_fraction"],
        "peak_force_n": final["peak_force_n"],
    }


def load_run(root):
    """Reject missing/extra episodes, changed evidence, unfair starts and invalid logs."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("data_kind") != DATA_KIND:
        raise ValueError("Only moving_target_stress_test data are accepted")
    cfg = MovingExperiment.from_dict(manifest["config"])
    if manifest.get("config_fingerprint") != cfg.fingerprint:
        raise ValueError("Invalid manifest configuration fingerprint")
    expected = {episode.id: episode for episode in cfg.episodes()}
    planned = manifest.get("planned_episode_ids")
    if not isinstance(planned, list) or planned != list(expected):
        raise ValueError("Manifest planned episode IDs disagree with the configuration")
    for field in ("runtime", "hardware", "sources"):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise ValueError(f"Missing run metadata: {field}")
    source_files = manifest["sources"].get("files")
    if not isinstance(source_files, dict) or not source_files or digest(source_files) != manifest["sources"].get("sha256"):
        raise ValueError("Invalid source-code evidence digest")
    summaries = list(root.glob("episodes/*/summary.json"))
    if (root / "episodes").exists() and any(
        not path.is_dir() or path.name not in expected for path in (root / "episodes").iterdir()
    ):
        raise ValueError("Unexpected published episode outside the experiment plan")
    if len(summaries) != len(expected):
        raise ValueError(f"Incomplete experiment: {len(summaries)}/{len(expected)} episodes")
    rows, evidence, seen, starts = [], [], set(), defaultdict(set)
    for path in sorted(summaries):
        summary = json.loads(path.read_text())
        identity = summary.get("episode_id")
        if identity in seen or identity not in expected or path.parent.name != identity:
            raise ValueError(f"Duplicate, unexpected or misplaced episode: {identity}")
        seen.add(identity)
        episode = expected[identity]
        if any(summary.get(key) != value for key, value in episode.asdict().items()):
            raise ValueError(f"Episode specification mismatch: {identity}")
        if (summary.get("pair_id") != episode.pair_id or summary.get("data_kind") != DATA_KIND
                or summary.get("config_fingerprint") != cfg.fingerprint):
            raise ValueError(f"Episode identity or configuration mismatch: {identity}")
        if type(summary.get("success")) is not bool or type(summary.get("steps")) is not int:
            raise ValueError("Outcome must be boolean and steps must be integer")
        if not 1 <= summary["steps"] <= episode.base_episode.max_steps:
            raise ValueError("Invalid episode length")
        calls = (summary["steps"] + summary["k"] - 1) // summary["k"]
        if summary.get("model_calls") != calls:
            raise ValueError("Model calls disagree with the execution interval")
        for metric in METRICS[1:]:
            finite(summary.get(metric), metric)
        finite(summary.get("simulation_s"), "simulation_s")
        if not math.isclose(summary["compute_s"], summary["inference_s"] + summary["simulation_s"], abs_tol=1e-6):
            raise ValueError("Compute time disagrees with inference plus simulation")
        if not math.isclose(summary["sim_time_s"], summary["steps"] / cfg.base_config.control_hz, abs_tol=1e-6):
            raise ValueError("Simulated duration disagrees with the step count")
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            value = summary.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing physical start evidence: {field}")
            starts[(summary["suite"], summary["task"], summary["state"], field)].add(value)
        files = ["summary.json", "steps.jsonl", "chunks.jsonl"]
        if cfg.base_config.record_video:
            files.append("video.mp4")
        evidence_path = path.parent / "evidence.json"
        stored = json.loads(evidence_path.read_text())
        recorded = stored.get("files")
        if not isinstance(recorded, dict):
            raise ValueError("Episode evidence must include a files mapping")
        if (stored.get("schema_version") != 1 or stored.get("episode_id") != identity
                or stored.get("config_fingerprint") != cfg.fingerprint
                or stored.get("sources_sha256") != manifest["sources"]["sha256"]
                or set(recorded) != set(files)):
            raise ValueError("Episode evidence identity or file inventory mismatch")
        hashes = {}
        for filename in files:
            source = path.parent / filename
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"Missing or empty episode file: {source}")
            hashes[filename] = sha256(source)
            value = recorded.get(filename)
            recorded_hash = value.get("sha256") if isinstance(value, dict) else value
            if recorded_hash != hashes[filename]:
                raise ValueError(f"Episode evidence checksum mismatch: {source}")
        steps = read_jsonl(path.parent / "steps.jsonl")
        chunks = read_jsonl(path.parent / "chunks.jsonl")
        if [row.get("step") for row in steps] != list(range(summary["steps"])):
            raise ValueError("Step log is incomplete or out of order")
        if [row.get("step") for row in chunks] != list(range(0, summary["steps"], summary["k"])):
            raise ValueError("Chunk log disagrees with the configured replan schedule")
        for row in steps:
            if row.get("replanned") is not (row["step"] % summary["k"] == 0):
                raise ValueError("Replanning was triggered outside the configured interval")
            if row.get("model_calls") != row["step"] // summary["k"] + 1:
                raise ValueError("Step model call count is inconsistent")
        metrics = motion_metrics(summary, steps, cfg.base_config.control_hz, cfg.motion.force_application_offset_z_m)
        if summary["condition"] == "static" and (metrics["active_physics_samples"] or metrics["peak_force_n"]):
            raise ValueError("Static control contains active motion forces")
        if metrics["peak_force_n"] > cfg.motion.max_force_n + 1e-9:
            raise ValueError("Applied intervention force exceeded the configured limit")
        rows.append({
            **{field: summary[field] for field in (
                "episode_id", "pair_id", "suite", "task", "state", "policy_seed", "condition", "k",
                "success", "steps", "model_calls", "compute_s", "inference_s", "simulation_s",
                "episode_wall_s", "sim_time_s",
            )}, **metrics,
        })
        evidence.append({
            "episode_id": identity, "files": hashes, "evidence_sha256": sha256(evidence_path),
        })
    if seen != set(expected):
        raise ValueError("Completed episodes disagree with the exact experiment plan")
    if any(len(values) != 1 for values in starts.values()):
        raise ValueError("Physical starts differ across K, conditions or policy seeds")
    return cfg, rows, {
        "root": str(root), "manifest_sha256": sha256(manifest_path),
        "config_fingerprint": cfg.fingerprint, "runtime": manifest["runtime"],
        "hardware": manifest["hardware"], "sources": manifest["sources"], "episodes": evidence,
    }


def cluster_statistics(rows, intervals, *, draws=5000, seed=20261024):
    """Return condition means and paired condition-minus-static differences by K."""
    import numpy as np

    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    states = sorted({row["state"] for row in rows})
    seeds = sorted({row["policy_seed"] for row in rows})
    lookup = {(row["state"], row["policy_seed"], row["condition"], row["k"]): row for row in rows}
    if len(lookup) != len(rows) or len(lookup) != len(states) * len(seeds) * len(CONDITIONS) * len(intervals):
        raise ValueError("Statistics require complete, unique paired state/seed/treatment blocks")
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(states), size=(draws, len(states)))
    estimates, comparisons = [], []
    for metric in METRICS:
        matrix = np.array([
            [[mean([float(lookup[(state, policy_seed, condition, k)][metric]) for policy_seed in seeds])
              for k in intervals] for condition in CONDITIONS] for state in states
        ])
        average = matrix.mean(axis=0)
        boot = matrix[draw_indices].mean(axis=1)
        for c, condition in enumerate(CONDITIONS):
            for i, k in enumerate(intervals):
                low, high = np.quantile(boot[:, c, i], [0.025, 0.975])
                estimates.append({
                    "metric": metric, "condition": condition, "k": k, "mean": float(average[c, i]),
                    "ci_low": float(low) if len(states) >= 2 else None,
                    "ci_high": float(high) if len(states) >= 2 else None,
                    "episodes": len(states) * len(seeds), "initial_states": len(states),
                    "policy_seeds": len(seeds),
                })
                if condition == "static":
                    continue
                difference = boot[:, c, i] - boot[:, 0, i]
                dlo, dhi = np.quantile(difference, [0.025, 0.975])
                comparisons.append({
                    "metric": metric, "condition": condition, "k": k, "reference_condition": "static",
                    "paired_difference": float(average[c, i] - average[0, i]),
                    "ci_low": float(dlo) if len(states) >= 2 else None,
                    "ci_high": float(dhi) if len(states) >= 2 else None,
                    "paired_episodes": len(states) * len(seeds), "initial_state_clusters": len(states),
                })
    return estimates, comparisons


def contact_and_motion_summary(rows, intervals):
    result = []
    for condition in CONDITIONS:
        for k in intervals:
            group = [row for row in rows if row["condition"] == condition and row["k"] == k]
            contacts = [row for row in group if row["contact"]]
            active = [row for row in group if row["active_physics_samples"] > 0]
            result.append({
                "condition": condition, "k": k, "episodes": len(group),
                "successes": sum(row["success"] for row in group), "contacts": len(contacts),
                "no_contact_episodes": len(group) - len(contacts),
                "contact_fraction": len(contacts) / len(group),
                "mean_contact_time_s_among_contacts": mean([row["first_contact_sim_time_s"] for row in contacts]),
                "motion_exposed_episodes": len(active),
                "mean_active_duration_s_all_episodes": mean([row["active_duration_s"] for row in group]),
                "mean_precontact_x_range_m": mean([row["precontact_x_range_m"] for row in group
                                                    if row["precontact_x_range_m"] is not None]),
                "mean_precontact_horizontal_speed_m_s": mean([row["precontact_mean_horizontal_speed_m_s"]
                    for row in group if row["precontact_mean_horizontal_speed_m_s"] is not None]),
                "mean_active_tracking_error_m_among_exposed": mean([row["active_mean_tracking_error_m"] for row in active]),
                "mean_active_saturation_fraction_among_exposed": mean([row["active_saturation_fraction"] for row in active]),
                "mean_max_precontact_tilt_deg": mean([row["max_precontact_tilt_deg"] for row in group]),
                "max_precontact_tilt_deg": max(row["max_precontact_tilt_deg"] for row in group),
            })
    return result


def interaction_statistics(rows, intervals, *, draws=5000, seed=20261024, reference_k=10):
    """Paired difference-in-differences of success, with the same state-cluster draws."""
    import numpy as np

    if reference_k not in intervals:
        raise ValueError(f"Interaction analysis requires reference K={reference_k}")
    states = sorted({row["state"] for row in rows})
    seeds = sorted({row["policy_seed"] for row in rows})
    lookup = {(row["state"], row["policy_seed"], row["condition"], row["k"]): row for row in rows}
    if len(lookup) != len(rows) or len(lookup) != len(states) * len(seeds) * len(CONDITIONS) * len(intervals):
        raise ValueError("Interactions require complete paired state/seed/treatment blocks")
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    matrix = np.array([
        [[mean([float(lookup[(state, policy_seed, condition, k)]["success"]) for policy_seed in seeds])
          for k in intervals] for condition in CONDITIONS] for state in states
    ])
    rng = np.random.default_rng(seed)
    sampled = matrix[rng.integers(0, len(states), size=(draws, len(states)))].mean(axis=1)
    average = matrix.mean(axis=0)
    ref = list(intervals).index(reference_k)
    records = []
    for c, condition in enumerate(CONDITIONS[1:], 1):
        for i, k in enumerate(intervals):
            if k == reference_k:
                continue
            observed = (average[c, i] - average[0, i]) - (average[c, ref] - average[0, ref])
            bootstrap = (sampled[:, c, i] - sampled[:, 0, i]) - (sampled[:, c, ref] - sampled[:, 0, ref])
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            count = lambda treatment, interval: sum(
                lookup[(state, policy_seed, treatment, interval)]["success"] for state in states for policy_seed in seeds
            )
            records.append({
                "metric": "success", "condition": condition, "k": k, "reference_k": reference_k,
                "interaction": float(observed), "ci_low": float(low) if len(states) >= 2 else None,
                "ci_high": float(high) if len(states) >= 2 else None,
                "initial_state_clusters": len(states), "policy_seeds": len(seeds),
                "episodes_per_cell": len(states) * len(seeds),
                "moving_successes_at_k": count(condition, k), "static_successes_at_k": count("static", k),
                "moving_successes_at_reference_k": count(condition, reference_k),
                "static_successes_at_reference_k": count("static", reference_k),
            })
    return records


def make_plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"static": "#5a6677", "slow": "#207f88", "fast": "#bd4f39"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, metric, title, scale in zip(axes, ("success", "model_calls", "compute_s"),
            ("Task success (%)", "Mean model calls", "Mean compute time (s)"), (100, 1, 1), strict=True):
        for condition in CONDITIONS:
            values = [row for row in result["estimates"] if row["metric"] == metric and row["condition"] == condition]
            x, y = [row["k"] for row in values], [row["mean"] * scale for row in values]
            ax.plot(x, y, "o-", label=condition.capitalize(), color=colors[condition])
            if all(row["ci_low"] is not None for row in values):
                ax.vlines(x, [row["ci_low"] * scale for row in values],
                          [row["ci_high"] * scale for row in values], color=colors[condition], alpha=.55)
        ax.set(title=title, xlabel="Actions executed before replanning (K)", xticks=result["intervals"])
        ax.grid(axis="y", alpha=.2)
        if metric == "success":
            ax.set_ylim(-3, 103)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Moving-target stress test: complete paired episodes", fontsize=12)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"success-and-cost.{suffix}"
        figure.savefig(path, dpi=160)
        paths.append(path)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, field, title, scale in zip(axes.flat,
            ("mean_precontact_x_range_m", "mean_precontact_horizontal_speed_m_s", "contact_fraction", "mean_max_precontact_tilt_deg"),
            ("Observed pre-contact x range (cm)", "Observed pre-contact mean speed (cm/s)",
             "Episodes with finger contact (%)", "Mean maximum pre-contact tilt (degrees)"),
            (100, 100, 100, 1), strict=True):
        for condition in CONDITIONS:
            values = [row for row in result["contact_and_motion"] if row["condition"] == condition]
            available = [row for row in values if row[field] is not None]
            ax.plot([row["k"] for row in available], [row[field] * scale for row in available],
                    "o-", label=condition.capitalize(), color=colors[condition])
        ax.set(title=title, xlabel="K", xticks=result["intervals"])
        ax.grid(axis="y", alpha=.2)
        if field == "contact_fraction":
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
    axes.flat[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Motion and contact diagnostics; contact does not establish a grasp", fontsize=12)
    for suffix in ("png", "pdf"):
        path = output / f"motion-and-contact.{suffix}"
        figure.savefig(path, dpi=160)
        paths.append(path)
    plt.close(figure)
    return paths


def format_value(value, digits=2):
    return "not observed" if value is None else f"{value:.{digits}f}"


def write_report(result, output):
    cfg = result["config"]
    base, motion = cfg["base_config"], cfg["motion"]
    first_after_onset = "; ".join(
        f"K={k}: {(math.floor(motion['start_delay_s'] * base['control_hz'] / k + 1e-10) + 1) * k / base['control_hz']:.2f} s"
        for k in sorted(base["intervals"])
    )
    lines = [
        "# Moving-target replanning experiment", "",
        "This is an altered-environment stress test of SmolVLA on the LIBERO alphabet-soup task. "
        "Its results are reported separately from standard LIBERO performance.", "",
        "## Question and method", "",
        "Does changing the interval between observations and action-chunk regeneration change "
        "robustness when the target can moves during the robot's approach?", "",
        f"The complete design contains **{result['episodes']} episodes**: "
        f"{len(base['initial_states'])} initial states × {len(base['policy_seeds'])} policy seeds × "
        f"{len(base['intervals'])} execution intervals × three conditions. "
        f"Initial state IDs: {base['initial_states']}; policy seeds: {base['policy_seeds']}; K: {base['intervals']}.", "",
        f"The policy predicts {base['chunk_size']} actions per call. The robot executes the first K "
        f"at {base['control_hz']} Hz and then receives a fresh observation. "
        "Simulation pauses during model inference. Motion follows simulation time, not wall-clock time. "
        "The force intervention never triggers an extra policy call.", "",
        f"The commanded path oscillates along world {motion['force_axis']} about the initial can position "
        f"with amplitude {motion['amplitude_m'] * 100:.1f} cm. The slow and fast periods are "
        f"{motion['periods_s']['slow']:g} s and {motion['periods_s']['fast']:g} s. "
        f"It starts after {motion['start_delay_s']:g} s with a {motion['ramp_s']:g} s ramp. "
        f"A bounded physical force (maximum {motion['max_force_n']:g} N) tracks this path; "
        f"its application point is {abs(motion['force_application_offset_z_m']) * 100:g} cm "
        "below the object center of mass. The corresponding y-axis moment is offset_z × force_x, "
        "equivalent to applying that force at the offset point; there is no separate orientation servo. "
        "The object is not teleported. The static control applies no motion force or moment. "
        "Motion force stops permanently at first finger contact in moving conditions. "
        "The object can subsequently move through normal physics and robot contact.", "",
        "The actual trajectory depends on friction and contact. Slow/fast label the commanded "
        "trajectory, so measured pre-contact motion and tracking error are reported below. "
        "Physical initial-state and initial-observation hashes match across treatments and policy seeds.", "",
        "## Completion and cost", "",
        "| Condition | K | Success | Mean calls | Mean compute (s) | Mean episode wall time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lookup = {(row["condition"], row["k"], row["metric"]): row for row in result["estimates"]}
    for row in result["contact_and_motion"]:
        get = lambda metric: lookup[(row["condition"], row["k"], metric)]["mean"]
        lines.append(f"| {row['condition']} | {row['k']} | {row['successes']}/{row['episodes']} "
                     f"({get('success') * 100:.1f}%) | {get('model_calls'):.1f} | {get('compute_s'):.2f} "
                     f"| {get('episode_wall_s'):.2f} |")
    lines.extend(["", "Compute time is inference plus simulator stepping; episode wall time also "
                  "includes reset, in-memory telemetry and video creation, recording and finalization. "
                  "It excludes JSON log writing, file checksums, model loading and warmup. "
                  "Lower cost can reflect an early success "
                  "or failure as well as a larger K.", "", "![Success and cost](success-and-cost.png)", "",
                  "## Paired change from the static control", "",
                  "Differences are moving-condition success minus static success at the same K. "
                  "Negative values indicate reduced completion under target motion.", "",
                  "| Condition | K | Difference (percentage points) | Exploratory 95% interval |",
                  "|---|---:|---:|---:|"])
    for row in result["paired_vs_static"]:
        if row["metric"] == "success":
            interval = "unavailable" if row["ci_low"] is None else f"[{row['ci_low'] * 100:.1f}, {row['ci_high'] * 100:.1f}]"
            lines.append(f"| {row['condition']} | {row['k']} | {row['paired_difference'] * 100:+.1f} | {interval} |")
    lines.extend(["", f"Percentile bootstrap with {result['bootstrap']['draws']} draws resamples initial states, "
        "keeping all seeds, K values and conditions together within each sampled state. "
        "Seeds are repeated policy samples, not additional independent scene clusters. "
        f"Only {len(base['initial_states'])} initial-state clusters are available; uncertainty estimates "
        "are weak and identical outcomes can produce zero-width intervals. The bootstrap is conditional "
        "on the selected seed set and task. The intervals are exploratory and are not adjusted for multiple comparisons. "
        "A difference whose interval includes zero does not establish equivalence.", "",
        "## Exploratory interaction between motion and K", "",
        "The interaction is (moving − static at K) − (moving − static at K=10). "
        "A positive estimate means that the motion-related change in success is more favorable at "
        "that K than at K=10. The same paired initial-state bootstrap is used, retaining all seeds. "
        "This accounts for baseline success differences but cannot remove floor/ceiling constraints "
        "or contact-dependent exposure. Raw success counts for all cells are given above.", "",
        "| Condition | K vs. 10 | Interaction (percentage points) | Exploratory 95% interval | State clusters / seeds |",
        "|---|---:|---:|---:|---:|"])
    for row in result["condition_by_k_interactions"]:
        interval = "unavailable" if row["ci_low"] is None else f"[{row['ci_low'] * 100:.1f}, {row['ci_high'] * 100:.1f}]"
        lines.append(f"| {row['condition']} | {row['k']} | {row['interaction'] * 100:+.1f} | {interval} "
                     f"| {row['initial_state_clusters']} / {row['policy_seeds']} |")
    lines.extend(["", "## Observed motion and contact", "",
        "| Condition | K | Contact / episodes | No contact | Contact time among contacts (s) | Mean pre-contact x range (cm) | Mean pre-contact speed (cm/s) |",
        "|---|---:|---:|---:|---:|---:|---:|"])
    for row in result["contact_and_motion"]:
        scaled = lambda key: None if row[key] is None else row[key] * 100
        lines.append(f"| {row['condition']} | {row['k']} | {row['contacts']}/{row['episodes']} | "
                     f"{row['no_contact_episodes']} | {format_value(row['mean_contact_time_s_among_contacts'])} | "
                     f"{format_value(scaled('mean_precontact_x_range_m'))} | "
                     f"{format_value(scaled('mean_precontact_horizontal_speed_m_s'))} |")
    tilt = "; ".join(f"{condition}: {max(row['max_precontact_tilt_deg'] for row in result['contact_and_motion'] if row['condition'] == condition):.1f}°"
                     for condition in CONDITIONS)
    lines.extend(["", f"Largest observed pre-contact tilt across K: {tilt}. The tilt diagnostic "
                  "checks whether an intended sliding target also tips or rotates substantially."])
    lines.extend(["", "Contact time is conditional on an observed finger-contact event. "
        "Episodes without contact are counted explicitly and are excluded from that time average; "
        "they are not assigned zero time. Contact is neither a verified grasp nor successful recovery. "
        "Pre-contact motion includes the initial delay, ends before first recorded contact, "
        "and therefore has a different observation duration in different episodes. "
        "Ranges, force exposure, tracking error and saturation are recorded at physics-substep "
        "resolution; mean speed is a control-rate finite-difference diagnostic. "
        "Tracking error and force-saturation diagnostics are available in contact-coverage.csv.", "",
        "![Observed motion and contact](motion-and-contact.png)", "", "## Interpretation limits", "",
        "- This single-task modification evaluates robustness to a specified artificial force, "
        "not real-world moving-object grasping or the unmodified LIBERO benchmark.",
        "- Force exposure stops at contact. Successful approaches can receive a shorter intervention; "
        "total displacement is an outcome-dependent diagnostic, not an independently randomized dose.",
        "- Reference phase is fixed. The first scheduled replan strictly after the configured onset is "
        f"{first_after_onset}, if the episode continues. An oscillating target can return toward an "
        "older planned grasp position, so a shorter reference period need not be harder. Actual motion "
        "can lag the reference. This is a possible timing mechanism, not a verified explanation of "
        "any individual outcome.",
        "- Smaller K changes both the arrival of fresh observations and stochastic regeneration of "
        "the action sequence. These mechanisms are not isolated by this experiment.",
        "- The simulator pauses during inference. The study does not measure a robot tracking an "
        "object that continues moving during real-world inference latency.",
        "- There are few independent initial states and only one checkpoint. Numeric rankings alone "
        "do not establish a general optimal K or a reliable difference between close results.",
        "- Native simulator success is used as recorded; post-release stability and missed-grasp "
        "event counts require separate evidence and are not inferred from finger contact.", "",
        "## Reproducibility", "",
        f"Model: `{base['model_id']}` at `{base['model_revision']}`. "
        f"Configuration fingerprint: `{result['provenance']['config_fingerprint']}`.", "",
        "statistics.json contains the complete configuration, runtime metadata, bootstrap settings "
        "and SHA-256 evidence for every raw episode file. artifacts.json binds this report and its "
        "plots to that statistics record and to the analysis source. No raw episode is modified "
        "during analysis. All planned episodes must be present before this report is generated.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def analyze(input_path, output_path, *, draws=5000, seed=20261024, plots=True):
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Analysis output must be separate from the raw run directory")
    cfg, rows, provenance = load_run(source)
    intervals = sorted(cfg.base_config.intervals)
    estimates, comparisons = cluster_statistics(rows, intervals, draws=draws, seed=seed)
    interactions = interaction_statistics(rows, intervals, draws=draws, seed=seed)
    result = {
        "schema_version": 1, "data_kind": DATA_KIND, "complete": True, "episodes": len(rows),
        "config": cfg.asdict(), "intervals": intervals,
        "bootstrap": {"draws": draws, "seed": seed, "unit": "initial_state", "policy_seeds_retained_in_cluster": True},
        "estimates": estimates, "paired_vs_static": comparisons,
        "condition_by_k_interactions": interactions,
        "contact_and_motion": contact_and_motion_summary(rows, intervals), "provenance": provenance,
        "analysis_source_sha256": sha256(__file__),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "statistics.json", result)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "summary.csv", estimates)
    write_csv(output / "paired-vs-static.csv", comparisons)
    write_csv(output / "condition-by-k-interactions.csv", interactions)
    write_csv(output / "contact-coverage.csv", result["contact_and_motion"])
    generated = make_plots(result, output) if plots else []
    if plots:
        write_report(result, output)
        generated.append(output / "report.md")
    generated.extend(output / name for name in ("episodes.csv", "summary.csv", "paired-vs-static.csv",
                                                "condition-by-k-interactions.csv", "contact-coverage.csv"))
    atomic_json(output / "artifacts.json", {
        "schema_version": 1, "data_kind": DATA_KIND, "statistics_sha256": sha256(output / "statistics.json"),
        "analysis_source_sha256": sha256(__file__), "files": {path.name: sha256(path) for path in generated},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = analyze(args.input, args.output, draws=args.draws, seed=args.seed, plots=not args.no_plots)
    print(f"Validated and analyzed {result['episodes']} complete moving-target episodes: {args.output}")


if __name__ == "__main__":
    main()
