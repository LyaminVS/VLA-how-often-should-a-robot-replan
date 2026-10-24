"""Guards against retrospective or unbound protocol records using test-only data."""

import json
from types import SimpleNamespace

import pytest

from moving_target.config import MotionSpec
from scripts.freeze_moving_target_protocol import ensure_fresh_destination, register_file, validate_physics


def test_existing_frozen_record_is_never_overwritten(tmp_path):
    record = tmp_path / "frozen.json"
    record.write_text("original evidence")
    with pytest.raises(FileExistsError, match="will not be replaced"):
        ensure_fresh_destination(record, tmp_path / "not-started")
    assert record.read_text() == "original evidence"


def test_even_empty_main_directory_blocks_retrospective_freeze(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    with pytest.raises(ValueError, match="before creating"):
        ensure_fresh_destination(tmp_path / "frozen.json", main)


def test_fresh_paths_pass_without_creating_any_files(tmp_path):
    ensure_fresh_destination(tmp_path / "frozen.json", tmp_path / "main")
    assert list(tmp_path.iterdir()) == []


def test_calibration_cannot_escape_evidence_root(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="inside"):
        register_file(outside, root, {})


def test_register_file_binds_content_with_relative_path(tmp_path):
    path = tmp_path / "physics" / "calibration.json"
    path.parent.mkdir()
    path.write_text("test-only data")
    hashes = {}
    assert register_file(path, tmp_path, hashes) == "physics/calibration.json"
    assert len(hashes["physics/calibration.json"]) == 64


@pytest.fixture
def synthetic_physics(tmp_path):
    """Minimal validation fixture; dummy MP4 bytes are never decoded or published."""
    root = tmp_path / "physics"
    root.mkdir()
    motion = MotionSpec()
    results = {}
    for condition in ("static", "slow", "fast"):
        trajectory = []
        for step in range(241):
            active = condition != "static" and step / 20 >= 0.5
            force = 0.1 if active else 0.0
            trajectory.append({
                "step": step, "condition": condition, "elapsed_sim_time_s": step / 20,
                "horizontal_force_n": force, "applied_torque_y_nm": motion.force_application_offset_z_m * force,
                "motion_active": active, "first_contact_step": None, "first_contact_sim_time": None,
                "max_force_after_contact_n": 0.0, "peak_force_n": force,
                "precontact_x_range_m": 0.02 if active else 0.0,
                "precontact_y_range_m": 0.0, "precontact_z_range_m": 0.0,
                "max_precontact_tilt_deg": 0.0, "active_physics_samples": 100 if active else 0,
            })
        results[condition] = {
            "reset": {"initial_state_sha256": "test-only-state", "initial_observation_sha256": "test-only-observation"},
            "final": {key: value for key, value in trajectory[-1].items() if key != "step"},
        }
        (root / f"{condition}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trajectory))
        (root / f"{condition}.mp4").write_bytes(b"TEST ONLY: a nonempty checksum fixture, not a video")
    record = {"motion": motion.asdict(), "state": 20, "steps": 240, "results": results}
    (root / "calibration.json").write_text(json.dumps(record))
    cfg = SimpleNamespace(motion=motion, base_config=SimpleNamespace(control_hz=20))
    return cfg, root, record


def test_valid_physics_record_binds_all_seven_inputs(synthetic_physics, tmp_path):
    cfg, root, _ = synthetic_physics
    hashes = {}
    assert validate_physics(cfg, root, tmp_path, hashes) == "physics/calibration.json"
    assert len(hashes) == 7


def test_calibration_final_summary_must_match_last_trajectory(synthetic_physics, tmp_path):
    cfg, root, record = synthetic_physics
    record["results"]["slow"]["final"]["precontact_x_range_m"] += 0.001
    (root / "calibration.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="final diagnostics differ"):
        validate_physics(cfg, root, tmp_path, {})


def test_short_calibration_trajectory_is_rejected(synthetic_physics, tmp_path):
    cfg, root, _ = synthetic_physics
    path = root / "fast.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="trajectory is incomplete"):
        validate_physics(cfg, root, tmp_path, {})


def test_calibration_with_robot_contact_cannot_be_frozen(synthetic_physics, tmp_path):
    cfg, root, record = synthetic_physics
    record["results"]["slow"]["final"]["first_contact_step"] = 10
    record["results"]["slow"]["final"]["first_contact_sim_time"] = 0.5
    (root / "calibration.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="no robot finger contact"):
        validate_physics(cfg, root, tmp_path, {})
