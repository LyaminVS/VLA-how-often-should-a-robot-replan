"""Temporary schema fixtures test aggregation safeguards, not robot performance."""

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_extension import analyze_extension, local_source_hashes
from vla_replan.analysis import ANNOTATION_FIELDS
from vla_replan.config import Experiment
from vla_replan.storage import atomic_json


@pytest.fixture
def sources(tmp_path):
    base = Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")

    def make(name, states, **changes):
        cfg = replace(base, name=name, initial_states=tuple(states), **changes)
        root = tmp_path / name
        manifest = {
            "schema_version": 1, "data_kind": "real_libero", "config": cfg.asdict(),
            "config_fingerprint": cfg.fingerprint,
            "planned_episode_ids": [episode.id for episode in cfg.episodes()],
            "runtime": {
                "python": "3.12.3", "system": "Linux", "machine": "x86_64",
                "packages": {"torch": "2.7.1", "mujoco": "3.3.2"},
                **local_source_hashes(),
                "simulator_assets": {"repo_id": "test-schema-assets", "revision": "a" * 40},
                "render_backend": "egl", "cpu_count": 12,
            },
            "hardware": {
                "device": "NVIDIA GeForce RTX 2060", "vram_bytes": 6014369792,
                "cuda": "12.6", "precision": "fp32", "deterministic_algorithms": True,
                "torch_cpu_threads": 6,
            },
        }
        atomic_json(root / "manifest.json", manifest)
        for episode in cfg.episodes():
            row = {
                **episode.asdict(), "episode_id": episode.id, "pair_id": episode.pair_id,
                "config_fingerprint": cfg.fingerprint, "data_kind": "real_libero",
                "success": episode.state < 2, "steps": 100,
                "model_calls": (100 + episode.k - 1) // episode.k,
                "initial_state_sha256": f"state{episode.state}",
                "initial_observation_sha256": f"observation{episode.state}",
                "compute_s": 10 + 100 / episode.k, "inference_s": 100 / episode.k,
                "simulation_s": 10, "episode_wall_s": 15 + 100 / episode.k,
                "sim_time_s": 5, "eef_path_m": .4, "gripper_close_commands": 2,
            }
            directory = root / "episodes" / episode.id
            atomic_json(directory / "summary.json", row)
            for filename in ("steps.jsonl", "chunks.jsonl", "video.mp4"):
                (directory / filename).write_bytes(b"schema fixture only\n")
        return root, cfg

    pilot, pilot_cfg = make("pilot", (0, 1))
    validation, validation_cfg = make("validation", (2, 3, 4, 5))
    return pilot, validation, pilot_cfg, validation_cfg, make


def run(sources, tmp_path, **kwargs):
    return analyze_extension(sources[0], sources[1], tmp_path / "derived", draws=100, plots=False, **kwargs)


def edit_manifest(root, edit):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    atomic_json(path, manifest)


def test_disjoint_aggregation_keeps_phase_estimates_and_original_provenance(sources, tmp_path):
    pilot, validation, *_ = sources
    before = {str(path): path.read_bytes() for root in (pilot, validation) for path in root.rglob("*")
              if path.is_file()}
    result = run(sources, tmp_path)
    assert result["primary_phase"] == "combined"
    assert result["combined_analysis_role"] == "main_descriptive"
    assert result["coverage"]["analyzed_episodes"] == 30
    assert result["provenance"]["unique_episode_count"] == 30
    for phase, expected in (("pilot", 1), ("validation", 0), ("combined", 1 / 3)):
        estimates = result["phases"][phase]["estimates"]
        assert all(row["mean"] == pytest.approx(expected) for row in estimates if row["metric"] == "success")
    # Six states contribute per K: pooling episodes must not average the two
    # source means with equal weight when their sample sizes differ (2 vs. 4).
    assert result["estimates"] == result["phases"]["combined"]["estimates"]
    assert all(row["mean"] == pytest.approx(1 / 3) for row in result["estimates"]
               if row["metric"] == "success")
    assert all(row["initial_states"] == 6 for row in result["per_task"])
    assert all(row["total_episodes"] == 6 for row in result["annotation_coverage"])
    assert all(path.read_bytes() == before[str(path)] for root in (pilot, validation)
               for path in root.rglob("*") if path.is_file())
    assert not (tmp_path / "derived" / "manifest.json").exists()
    with (tmp_path / "derived" / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["phase"] for row in rows} == {"pilot", "validation"}
    assert len({row["episode_id"] for row in rows}) == len(rows)
    for source in result["provenance"]["sources"]:
        assert len(source["manifest_sha256"]) == 64
        assert all("video.mp4" in row["files"] for row in source["episodes"])


def test_overlapping_initial_states_are_rejected(sources, tmp_path):
    pilot, _, _, _, make = sources
    overlap, _ = make("overlap", (1, 2))
    with pytest.raises(ValueError, match="states overlap"):
        analyze_extension(pilot, overlap, tmp_path / "derived", draws=100, plots=False)


def test_incomplete_validation_is_rejected_before_output(sources, tmp_path):
    next((sources[1] / "episodes").glob("*/summary.json")).unlink()
    with pytest.raises(ValueError, match="Incomplete experiment"):
        run(sources, tmp_path)
    assert not (tmp_path / "derived").exists()


def test_mismatched_experimental_setting_is_rejected(sources, tmp_path):
    pilot, _, _, _, make = sources
    changed, _ = make("changed", (2, 3), num_denoising_steps=5)
    with pytest.raises(ValueError, match="settings differ: num_denoising_steps"):
        analyze_extension(pilot, changed, tmp_path / "derived", draws=100, plots=False)


@pytest.mark.parametrize("section,field,value", [
    ("runtime", "packages", {"torch": "99", "mujoco": "3.3.2"}),
    ("hardware", "torch_cpu_threads", 4),
    ("hardware", "device", "different GPU"),
])
def test_mismatched_runtime_or_hardware_is_rejected(sources, tmp_path, section, field, value):
    edit_manifest(sources[1], lambda manifest: manifest[section].update({field: value}))
    with pytest.raises(ValueError, match=f"Run {section} differs"):
        run(sources, tmp_path)


@pytest.mark.parametrize("field", ["code_sha256", "dependency_lock_sha256"])
def test_equal_but_changed_frozen_code_or_lock_is_rejected(sources, tmp_path, field):
    for source in sources[:2]:
        edit_manifest(source, lambda manifest: manifest["runtime"].update({field: "0" * 64}))
    with pytest.raises(ValueError, match=field):
        run(sources, tmp_path)


def test_missing_runtime_evidence_is_rejected(sources, tmp_path):
    edit_manifest(sources[1], lambda manifest: manifest["runtime"].pop("simulator_assets"))
    with pytest.raises(ValueError, match="missing required runtime metadata"):
        run(sources, tmp_path)


def test_missing_raw_recording_is_rejected(sources, tmp_path):
    next((sources[1] / "episodes").glob("*/video.mp4")).unlink()
    with pytest.raises(ValueError, match="missing or empty episode artifact"):
        run(sources, tmp_path)


def test_manifest_plan_cannot_hide_duplicate_ids(sources, tmp_path):
    edit_manifest(sources[1], lambda manifest: manifest["planned_episode_ids"].append(
        manifest["planned_episode_ids"][0]
    ))
    with pytest.raises(ValueError, match="planned episode IDs"):
        run(sources, tmp_path)


def write_labels(path, labels):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(labels)


def test_partial_event_annotation_and_missing_labels_keep_honest_denominators(sources, tmp_path):
    episode = next(e for e in sources[3].episodes() if e.state == 2 and e.k == 25)
    write_labels(sources[1] / "annotations.csv", [{
        "episode_id": episode.id, "reviewed": "no", "missed_grasps": "", "dropped_objects": 1,
        "correction_after_replan": "unclear", "failure_category": "unclear", "notes": "One known drop.",
    }])
    result = run(sources, tmp_path)
    validation = result["phases"]["validation"]
    item = next(row for row in validation["annotation_coverage"] if row["k"] == 25)
    assert item["total_episodes"] == 4
    assert item["selected_for_visual_review"] == 2
    assert item["fully_reviewed_episodes"] == 0
    assert item["inspected_episodes"] is None
    assert item["dropped_objects_known_episodes"] == item["dropped_objects_total_known"] == 1
    assert item["missed_grasps_known_episodes"] == 0
    assert item["missed_grasps_total_known"] is None
    assert all(row["drops_per_reviewed_episode"] is None for row in validation["annotations"])


def test_outside_subset_annotation_is_rejected(sources, tmp_path):
    episode = next(e for e in sources[3].episodes() if e.state == 4)
    write_labels(sources[1] / "annotations.csv", [{
        "episode_id": episode.id, "reviewed": "yes", "missed_grasps": 0, "dropped_objects": 0,
        "correction_after_replan": "no", "failure_category": "timeout", "notes": "Outside subset.",
    }])
    with pytest.raises(ValueError, match="outside the prespecified"):
        run(sources, tmp_path)


def test_aggregate_output_cannot_write_inside_source_run(sources):
    with pytest.raises(ValueError, match="outside both source run"):
        analyze_extension(sources[0], sources[1], sources[0] / "report", draws=100, plots=False)


def inspection_record(sources):
    return {
        "selected_episode_ids": [episode.id for episode in sources[3].episodes() if episode.state in (2, 3)],
        "inspected_episode_ids": [],
        "method": "Both cameras and logs, with ambiguous event counts retained as unknown.",
    }


def test_inspected_ambiguous_episode_is_distinct_from_selected_and_resolved(sources, tmp_path):
    evidence = inspection_record(sources)
    episode = next(e for e in sources[3].episodes() if e.state == 2 and e.k == 25)
    evidence["inspected_episode_ids"] = [episode.id]
    atomic_json(sources[1] / "inspection.json", evidence)
    result = run(sources, tmp_path)
    coverage = next(row for row in result["phases"]["validation"]["annotation_coverage"] if row["k"] == 25)
    assert coverage["selected_for_visual_review"] == 2
    assert coverage["inspected_episodes"] == 1
    assert coverage["fully_reviewed_episodes"] == 0
    assert coverage["dropped_objects_known_episodes"] == 0
    source = next(source for source in result["provenance"]["sources"] if source["phase"] == "validation")
    assert source["inspection"]["method"] == evidence["method"]
    assert len(source["inspection"]["sha256"]) == 64
    # The pilot has no inspection sidecar, so total inspected coverage cannot be asserted.
    assert all(row["inspected_episodes"] is None for row in result["annotation_coverage"])
    with (tmp_path / "derived" / "annotations.csv").open() as handle:
        labels = list(csv.DictReader(handle))
    assert next(row for row in labels if row["episode_id"] == episode.id)["inspected"] == "True"
    assert all(row["inspected"] == "" for row in labels if row["phase"] == "pilot")


@pytest.mark.parametrize("problem,match", [
    ("missing_selected", "does not equal"),
    ("duplicate_selected", "Duplicate inspection IDs"),
    ("outside_inspected", "outside the selected"),
    ("duplicate_inspected", "Duplicate inspection IDs"),
    ("empty_method", "nonempty method"),
])
def test_invalid_inspection_evidence_is_rejected(sources, tmp_path, problem, match):
    evidence = inspection_record(sources)
    if problem == "missing_selected":
        evidence["selected_episode_ids"].pop()
    elif problem == "duplicate_selected":
        evidence["selected_episode_ids"].append(evidence["selected_episode_ids"][0])
    elif problem == "outside_inspected":
        evidence["inspected_episode_ids"] = [next(e.id for e in sources[3].episodes() if e.state == 4)]
    elif problem == "duplicate_inspected":
        evidence["inspected_episode_ids"] = [evidence["selected_episode_ids"][0]] * 2
    else:
        evidence["method"] = " "
    atomic_json(sources[1] / "inspection.json", evidence)
    with pytest.raises(ValueError, match=match):
        run(sources, tmp_path)


def test_resolved_labels_cannot_contradict_known_inspection_coverage(sources, tmp_path):
    evidence = inspection_record(sources)
    episode = evidence["selected_episode_ids"][0]
    atomic_json(sources[1] / "inspection.json", evidence)
    write_labels(sources[1] / "annotations.csv", [{
        "episode_id": episode, "reviewed": "yes", "missed_grasps": 0, "dropped_objects": 0,
        "correction_after_replan": "unclear", "failure_category": "timeout", "notes": "Resolved.",
    }])
    with pytest.raises(ValueError, match="require an inspected episode"):
        run(sources, tmp_path)
