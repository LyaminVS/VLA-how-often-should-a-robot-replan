"""Publication, telemetry and resume integrity tests with lightweight fakes."""

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from moving_target import runner
from moving_target.config import MotionSpec, MovingExperiment
from vla_replan import backend
from vla_replan.config import Experiment
from vla_replan.core import Observation


@pytest.fixture
def config():
    base = Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    base = replace(base, intervals=(5, 10), initial_states=(10,), policy_seeds=(11, 22), record_video=False, warmup_calls=1)
    return MovingExperiment(base, MotionSpec())


@pytest.fixture
def fake_runtime(monkeypatch):
    class Environment:
        resets = []
        conditions = []

        def __init__(self, base, motion):
            self.motion = motion

        def set_condition(self, condition):
            self.condition = condition
            self.conditions.append(condition)

        def reset(self, episode):
            self.step_index = 0
            self.episode = episode
            self.resets.append((episode.state, episode.policy_seed, self.condition, episode.k))
            return Observation(None, [0, 0, 0], [0, 0]), {
                "initial_state_sha256": f"state-{episode.state}",
                "initial_observation_sha256": f"observation-{episode.state}",
                "instruction": "TEST ONLY",
            }

        def step(self, action):
            self.step_index += 1
            return Observation(None, [0, 0, 0], [0, 0]), self.step_index == 3, False

        def telemetry(self):
            contact = self.step_index >= 3
            return {
                "condition": self.condition,
                "step": self.step_index,
                "object_pos": [self.step_index * 0.001, 0, 0],
                "target_pos": [self.step_index * 0.001 + 0.01, 0, 0],
                "motion_active": self.condition != "static" and not contact,
                "tracking_error_m": 0.01,
                "horizontal_force_n": [0.2, 0.0],
                "force_saturated": False,
                "first_contact_step": 3 if contact else None,
                "first_contact_sim_time": 0.15 if contact else None,
            }

        def close(self):
            pass

    class Policy:
        def __init__(self, config, device):
            self.runtime = {"device": "TEST_ONLY"}

        def reset(self):
            pass

        def predict(self, observation, seed):
            return [[0, 0, 0, 0, 0, 0, -1] for _ in range(50)]

        def peak_memory(self):
            return 0

        def reset_peak_memory(self):
            pass

    module = types.ModuleType("moving_target.environment")
    module.MovingTargetEnvironment = Environment
    monkeypatch.setitem(sys.modules, "moving_target.environment", module)
    monkeypatch.setattr(backend, "SmolVLAPolicy", Policy)
    monkeypatch.setattr(runner, "runtime_metadata", lambda: {"test_only": True})
    monkeypatch.setattr(runner, "git_metadata", lambda: {"test_only": True})
    return Environment, Policy


def test_partial_resume_preserves_completed_episode_bytes(fake_runtime, config, tmp_path):
    first = runner.run(config, tmp_path, limit=2)
    assert first == {"completed": 2, "new_episodes": 2, "planned": 12}
    paths = list((tmp_path / "episodes").glob("*/*"))
    originals = {path: path.read_bytes() for path in paths}
    last = runner.run(config, tmp_path)
    assert last == {"completed": 12, "new_episodes": 10, "planned": 12}
    assert all(path.read_bytes() == content for path, content in originals.items())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["data_kind"] == "moving_target_stress_test"
    assert len(manifest["planned_episode_ids"]) == 12


def test_recorder_adds_pre_action_and_final_motion_without_changing_core(fake_runtime, config, tmp_path):
    runner.run(config, tmp_path, limit=1)
    directory = next((tmp_path / "episodes").iterdir())
    rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    summary = json.loads((directory / "summary.json").read_text())
    assert [row["motion"]["step"] for row in rows] == [0, 1, 2]
    assert summary["motion_final"]["step"] == 3
    assert summary["motion_summary"]["contact_observed"]
    assert summary["motion_summary"]["first_contact_sim_time"] == 0.15
    assert summary["episode_id"].endswith("__motion_" + summary["condition"])
    assert "motion" not in summary["pair_id"]
    assert all(row["motion"]["condition"] == summary["condition"] for row in rows)


def test_corrupted_raw_evidence_is_never_silently_skipped(fake_runtime, config, tmp_path):
    runner.run(config, tmp_path, limit=1)
    path = next((tmp_path / "episodes").glob("*/steps.jsonl"))
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.run(config, tmp_path)


def test_missing_integrity_record_blocks_resume(fake_runtime, config, tmp_path):
    runner.run(config, tmp_path, limit=1)
    next((tmp_path / "episodes").glob("*/evidence.json")).unlink()
    with pytest.raises(ValueError, match="Missing episode integrity"):
        runner.run(config, tmp_path)


def test_motion_change_cannot_mix_into_existing_collection(fake_runtime, config, tmp_path):
    runner.run(config, tmp_path, limit=1)
    changed = replace(config, motion=replace(config.motion, amplitude_m=0.03))
    with pytest.raises(ValueError, match="changed"):
        runner.run(changed, tmp_path)


def test_completed_collection_checks_integrity_without_loading_model(fake_runtime, config, tmp_path, monkeypatch):
    runner.run(config, tmp_path)

    def unexpected_load(*args):
        raise AssertionError("Completed runs should not load a model")

    monkeypatch.setattr(backend, "SmolVLAPolicy", unexpected_load)
    assert runner.run(config, tmp_path)["new_episodes"] == 0


def test_simulator_error_remains_failed_attempt_not_task_failure(fake_runtime, config, tmp_path):
    environment, _ = fake_runtime

    def fail(self, action):
        raise RuntimeError("test simulator fault")

    environment.step = fail
    with pytest.raises(RuntimeError, match="simulator fault"):
        runner.run(config, tmp_path)
    assert not list(tmp_path.glob("episodes/*/summary.json"))
    errors = list(tmp_path.glob("attempts/*/error.json"))
    assert len(errors) == 1
    assert "test simulator fault" in json.loads(errors[0].read_text())["traceback"]


def test_physical_start_must_match_across_sampling_seeds(fake_runtime, config, tmp_path):
    environment, _ = fake_runtime
    original = environment.reset

    def inconsistent(self, episode):
        obs, info = original(self, episode)
        info["initial_observation_sha256"] = f"seed-{episode.policy_seed}"
        return obs, info

    environment.reset = inconsistent
    with pytest.raises(RuntimeError, match="physical initial observations differ"):
        runner.run(config, tmp_path)
    assert len(list(tmp_path.glob("episodes/*/summary.json"))) == 6


def test_static_condition_has_no_active_tracking_measurements():
    summary = runner.summarize_motion([
        {"motion": {"motion_active": False, "tracking_error_m": 0.0, "object_pos": [0, 0, 0]}}
    ], {"object_pos": [0, 0, 0], "first_contact_step": None, "first_contact_sim_time": None}, 20)
    assert summary["active_samples"] == 0
    assert summary["tracking_error_mean_m"] is None
    assert summary["contact_observed"] is False
