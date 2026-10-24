"""Scientific aggregation/integrity safeguards; fixtures are not robot results."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moving_target.config import CONDITIONS, MotionSpec, MovingExperiment
from scripts.analyze_moving_target import (
    analyze, cluster_statistics, contact_and_motion_summary, interaction_statistics, load_run, motion_metrics, sha256,
)
from vla_replan.config import Experiment, digest
from vla_replan.storage import atomic_json, write_jsonl


def telemetry(condition, elapsed=0.0, contact=None):
    active = condition != "static" and contact is None
    return {
        "condition": condition, "object_pos": [elapsed * .01, 0., 1.], "target_pos": [0., 0., 1.],
        "elapsed_sim_time_s": elapsed, "motion_active": active,
        "first_contact_step": 1 if contact is not None else None, "first_contact_sim_time": contact,
        "tracking_error_m": .002, "horizontal_force_n": -.05 if active else 0.,
        "applied_torque_y_nm": .0015 if active else 0.,
        "force_saturated": False, "active_physics_samples": 10 if condition != "static" else 0,
        "active_sim_time_s": .02 if condition != "static" else 0.,
        "tracking_mae_m": .002 if condition != "static" else None,
        "tracking_rmse_m": .002 if condition != "static" else None,
        "force_saturation_fraction": 0. if condition != "static" else None,
        "peak_force_n": .05 if condition != "static" else 0., "max_force_after_contact_n": 0.,
        "precontact_x_range_m": .003, "precontact_y_range_m": .001, "precontact_z_range_m": .0001,
        "max_precontact_tilt_deg": 1.5,
    }


@pytest.fixture
def raw_run(tmp_path):
    base = Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    cfg = MovingExperiment(replace(base, initial_states=(10, 11), policy_seeds=(11, 22),
                                  intervals=(5, 10), record_video=False), MotionSpec())
    root = tmp_path / "raw"
    sources = {"files": {"fixture-only.py": "a" * 64}}
    sources["sha256"] = digest(sources["files"])
    manifest = {
        "data_kind": "moving_target_stress_test", "config": cfg.asdict(),
        "config_fingerprint": cfg.fingerprint, "planned_episode_ids": [ep.id for ep in cfg.episodes()],
        "runtime": {"fixture": True}, "hardware": {"fixture": True}, "sources": sources,
    }
    atomic_json(root / "manifest.json", manifest)
    for ep in cfg.episodes():
        directory = root / "episodes" / ep.id
        directory.mkdir(parents=True)
        summary = {
            **ep.asdict(), "episode_id": ep.id, "pair_id": ep.pair_id, "env_seed": ep.env_seed,
            "data_kind": "moving_target_stress_test", "config_fingerprint": cfg.fingerprint,
            "success": ep.base_episode.state == 10, "steps": 2, "model_calls": 1,
            "compute_s": 1.1, "inference_s": 1., "simulation_s": .1, "episode_wall_s": 1.2,
            "sim_time_s": .1, "initial_state_sha256": f"state{ep.base_episode.state}",
            "initial_observation_sha256": f"observation{ep.base_episode.state}",
            "motion_final": telemetry(ep.condition, .1, .06 if ep.base_episode.state == 10 else None),
        }
        atomic_json(directory / "summary.json", summary)
        write_jsonl(directory / "steps.jsonl", [
            {"step": step, "replanned": step == 0, "model_calls": 1, "motion": telemetry(ep.condition, step / 20)}
            for step in range(2)
        ])
        write_jsonl(directory / "chunks.jsonl", [{"step": 0}])
        atomic_json(directory / "evidence.json", {
            "schema_version": 1, "episode_id": ep.id, "config_fingerprint": cfg.fingerprint,
            "sources_sha256": sources["sha256"],
            "files": {name: sha256(directory / name) for name in ("summary.json", "steps.jsonl", "chunks.jsonl")},
        })
    return root, cfg


def update_summary(root, mutation):
    path = next((root / "episodes").glob("*/summary.json"))
    data = json.loads(path.read_text())
    mutation(data)
    atomic_json(path, data)
    evidence_path = path.parent / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["files"][path.name] = sha256(path)
    atomic_json(evidence_path, evidence)


def test_complete_run_is_paired_and_raw_data_remain_unchanged(raw_run, tmp_path):
    root, cfg = raw_run
    before = {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}
    result = analyze(root, tmp_path / "analysis", draws=100, plots=False)
    assert result["episodes"] == 24
    assert result["complete"] is True
    assert all(row["paired_difference"] == 0 for row in result["paired_vs_static"] if row["metric"] == "success")
    assert all(row["contacts"] == row["no_contact_episodes"] == 2 for row in result["contact_and_motion"])
    assert all(row["mean_contact_time_s_among_contacts"] == .06 for row in result["contact_and_motion"])
    assert before == {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}


def test_incomplete_run_never_creates_report(raw_run, tmp_path):
    root, _ = raw_run
    next((root / "episodes").glob("*/summary.json")).unlink()
    with pytest.raises(ValueError, match="Incomplete experiment"):
        analyze(root, tmp_path / "analysis", draws=100, plots=False)
    assert not (tmp_path / "analysis").exists()


def test_mutated_payload_is_rejected(raw_run):
    root, _ = raw_run
    path = next((root / "episodes").glob("*/steps.jsonl"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_run(root)


def test_plan_cannot_silently_omit_treatment(raw_run):
    root, _ = raw_run
    path = root / "manifest.json"
    data = json.loads(path.read_text())
    data["planned_episode_ids"].pop()
    atomic_json(path, data)
    with pytest.raises(ValueError, match="planned episode IDs"):
        load_run(root)


def test_physical_start_mismatch_is_rejected_even_across_seeds(raw_run):
    root, _ = raw_run
    update_summary(root, lambda item: item.update(initial_observation_sha256="different"))
    with pytest.raises(ValueError, match="Physical starts differ"):
        load_run(root)


def test_source_identity_cannot_be_changed_with_intact_payloads(raw_run):
    root, _ = raw_run
    path = next((root / "episodes").glob("*/evidence.json"))
    data = json.loads(path.read_text())
    data["sources_sha256"] = "0" * 64
    atomic_json(path, data)
    with pytest.raises(ValueError, match="evidence identity"):
        load_run(root)


def test_output_cannot_overlap_raw_run(raw_run):
    root, _ = raw_run
    with pytest.raises(ValueError, match="separate from the raw"):
        analyze(root, root / "report", draws=100, plots=False)


def test_physics_diagnostics_replace_downsampled_motion_estimates():
    final = telemetry("fast", .1)
    final["precontact_x_range_m"] = .037
    final["active_sim_time_s"] = .082
    final["tracking_mae_m"] = .004
    metrics = motion_metrics({"steps": 2, "sim_time_s": .1, "motion_final": final}, [
        {"step": step, "motion": telemetry("fast", step / 20)} for step in range(2)
    ], 20, -.03)
    assert metrics["precontact_x_range_m"] == .037
    assert metrics["active_duration_s"] == .082
    assert metrics["active_mean_tracking_error_m"] == .004
    assert metrics["active_mean_horizontal_force_n"] == .05


def test_force_after_contact_is_rejected():
    final = telemetry("fast", .1, .06)
    final["max_force_after_contact_n"] = .001
    with pytest.raises(ValueError, match="after contact"):
        motion_metrics({"steps": 2, "sim_time_s": .1, "motion_final": final}, [
            {"step": step, "motion": telemetry("fast", step / 20)} for step in range(2)
        ], 20, -.03)


def test_torque_must_be_force_application_moment():
    rows = [{"step": step, "motion": telemetry("fast", step / 20)} for step in range(2)]
    rows[1]["motion"]["applied_torque_y_nm"] = .2
    with pytest.raises(ValueError, match="force application point"):
        motion_metrics({"steps": 2, "sim_time_s": .1, "motion_final": telemetry("fast", .1)}, rows, 20, -.03)


def test_duplicated_seed_samples_do_not_narrow_cluster_intervals():
    def data(seeds):
        return [{
            "state": state, "policy_seed": seed, "condition": condition, "k": 5,
            **{metric: float(state % 2) for metric in ("success", "model_calls", "compute_s", "inference_s", "episode_wall_s", "sim_time_s")},
        } for state in range(5) for seed in seeds for condition in CONDITIONS]
    first, first_delta = cluster_statistics(data([11]), [5], draws=1000)
    repeated, repeated_delta = cluster_statistics(data([11, 22, 33]), [5], draws=1000)
    for one, three in zip(first, repeated, strict=True):
        assert (one["mean"], one["ci_low"], one["ci_high"]) == (three["mean"], three["ci_low"], three["ci_high"])
        assert one["initial_states"] == three["initial_states"] == 5
    assert all(row["ci_low"] == row["ci_high"] == 0 for row in first_delta + repeated_delta)


def test_no_contact_times_are_missing_not_zero(raw_run):
    _, rows, _ = load_run(raw_run[0])
    for row in rows:
        row["contact"] = False
        row["first_contact_sim_time_s"] = None
    summary = contact_and_motion_summary(rows, [5, 10])
    assert all(row["mean_contact_time_s_among_contacts"] is None for row in summary)
    assert all(row["no_contact_episodes"] == row["episodes"] == 4 for row in summary)


def test_interaction_uses_paired_motion_changes_and_preserves_seed_clusters():
    def data(seeds):
        # Motion has no effect at K=5 and harms every state equally at K=10.
        # The interaction is exactly +1 despite heterogeneous static K=5 success.
        return [{
            "state": state, "policy_seed": seed, "condition": condition, "k": k,
            "success": bool(state % 2) if k == 5 else condition == "static",
        } for state in range(5) for seed in seeds for condition in CONDITIONS for k in (5, 10)]
    first = interaction_statistics(data([11]), [5, 10], draws=1000)
    repeated = interaction_statistics(data([11, 22, 33]), [5, 10], draws=1000)
    assert all(row["interaction"] == row["ci_low"] == row["ci_high"] == 1 for row in first + repeated)
    assert all(row["initial_state_clusters"] == 5 and row["policy_seeds"] == 3 for row in repeated)
    assert all(row["moving_successes_at_k"] == row["static_successes_at_k"] == 6 for row in repeated)
    assert all(row["moving_successes_at_reference_k"] == 0 and row["static_successes_at_reference_k"] == 15 for row in repeated)
