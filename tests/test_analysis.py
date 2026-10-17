import json
from dataclasses import replace
from pathlib import Path

import pytest

from vla_replan.analysis import (
    analyze,
    annotation_template,
    cluster_estimates,
    estimate_budget,
    load_results,
)
from vla_replan.config import Experiment
from vla_replan.storage import atomic_json, exclusive_run


@pytest.fixture
def dataset(tmp_path):
    """Temporary schema fixtures only. These are not real VLA results."""
    cfg = Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    cfg = replace(cfg, initial_states=(0, 1, 2), policy_seeds=(11, 29))
    root = tmp_path / "schema_fixture"
    atomic_json(
        root / "manifest.json",
        {"data_kind": "real_libero", "config": cfg.asdict(), "config_fingerprint": cfg.fingerprint},
    )
    for e in cfg.episodes():
        row = {
            **e.asdict(),
            "episode_id": e.id,
            "pair_id": e.pair_id,
            "config_fingerprint": cfg.fingerprint,
            "data_kind": "real_libero",
            "success": e.k < 25 or e.state == 0,
            "steps": 100,
            "model_calls": (100 + e.k - 1) // e.k,
            "initial_state_sha256": f"state{e.state}",
            "initial_observation_sha256": f"obs{e.state}",
            "compute_s": 10 + 100 / e.k,
            "inference_s": 100 / e.k,
            "simulation_s": 10,
            "episode_wall_s": 15 + 100 / e.k,
            "sim_time_s": 5,
            "eef_path_m": 0.4,
            "gripper_close_commands": 2,
        }
        atomic_json(root / "episodes" / e.id / "summary.json", row)
    return root, cfg


def test_paired_bootstrap_retains_seed_clusters(dataset):
    root, cfg = dataset
    _, rows, _ = load_results(root)
    estimates, tasks = cluster_estimates(rows, cfg.intervals, "success", draws=1000)
    assert estimates[0]["mean"] == 1
    assert estimates[-1]["mean"] == pytest.approx(1 / 3)
    assert estimates[-1]["paired_difference"] == pytest.approx(-2 / 3)
    assert estimates[2]["paired_difference"] == 0
    assert estimates[2]["difference_ci_low"] == 0
    assert all(r["initial_states"] == 3 for r in tasks)  # six episodes are three clusters


def test_incomplete_runs_do_not_silently_change_cohort(dataset):
    root, cfg = dataset
    (root / "episodes" / cfg.episodes()[0].id / "summary.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        load_results(root)
    _, rows, coverage = load_results(root, allow_incomplete=True)
    assert len(rows) == 20
    assert coverage["completed_episodes"] == 29
    assert coverage["omitted_incomplete_state_blocks"] == 1
    assert coverage["provisional"]


def test_start_hash_mismatch_is_rejected(dataset):
    root, cfg = dataset
    path = root / "episodes" / cfg.episodes()[0].id / "summary.json"
    row = json.loads(path.read_text())
    row["initial_observation_sha256"] = "different"
    atomic_json(path, row)
    with pytest.raises(ValueError, match="Initial states differ"):
        load_results(root)


def test_synthetic_kind_cannot_be_used_as_research(dataset):
    root, _ = dataset
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["data_kind"] = "synthetic"
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="synthetic"):
        load_results(root)


def test_model_call_count_is_checked(dataset):
    root, cfg = dataset
    path = root / "episodes" / cfg.episodes()[0].id / "summary.json"
    row = json.loads(path.read_text())
    row["model_calls"] += 1
    atomic_json(path, row)
    with pytest.raises(ValueError, match="call count"):
        load_results(root)


def test_report_figures_and_tables(dataset, tmp_path):
    root, _ = dataset
    output = tmp_path / "report"
    coverage = analyze(root, output, draws=200)
    assert not coverage["provisional"]
    assert (output / "comparison.png").stat().st_size > 1000
    assert (output / "comparison.pdf").stat().st_size > 1000
    assert (output / "demo_candidates.csv").exists()
    assert len(json.loads((output / "statistics.json").read_text())["estimates"]) == 35


def test_runtime_estimate_uses_measured_timings(dataset):
    root, cfg = dataset
    budget = estimate_budget(cfg, root)
    assert budget["measured_seconds_per_call"] == 1
    assert budget["measured_seconds_per_control_step"] == 0.1
    with pytest.raises(ValueError, match="precision"):
        estimate_budget(replace(cfg, precision="fp16"), root)


def test_annotation_template_preserves_manual_work(dataset, tmp_path):
    root, _ = dataset
    output = tmp_path / "labels.csv"
    annotation_template(root, output)
    assert output.read_text().count(",no,") == 30
    with pytest.raises(FileExistsError):
        annotation_template(root, output)


def test_writer_lock_rejects_overlapping_runs(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises(RuntimeError, match="Another experiment"):
            with exclusive_run(tmp_path):
                pass
    with exclusive_run(tmp_path):
        pass
