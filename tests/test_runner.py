import json
from dataclasses import replace
from pathlib import Path

import pytest

from vla_replan import backend
from vla_replan.config import Experiment
from vla_replan.core import Observation
from vla_replan.runner import run


@pytest.fixture
def fake_backend(monkeypatch):
    """Test publication/resume semantics only; does not run a learned policy."""

    class Environment:
        def __init__(self, config):
            pass

        def reset(self, episode):
            self.step_index = 0
            return Observation(None, [0, 0, 0], [0, 0]), {
                "initial_state_sha256": "fixture",
                "initial_observation_sha256": "fixture",
                "instruction": "TEST ONLY",
            }

        def step(self, action):
            self.step_index += 1
            return Observation(None, [0, 0, 0], [0, 0]), self.step_index == 3, False

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

    monkeypatch.setattr(backend, "LiberoEnvironment", Environment)
    monkeypatch.setattr(backend, "SmolVLAPolicy", Policy)
    return Environment, Policy


def test_partial_run_resumes_without_replacing_completed_episodes(fake_backend, tmp_path):
    config = Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    config = replace(config, record_video=False)
    run(config, tmp_path, limit=2)
    paths = list(tmp_path.glob("episodes/*/summary.json"))
    assert len(paths) == 2
    originals = {p: p.read_bytes() for p in paths}
    run(config, tmp_path)
    assert len(list(tmp_path.glob("episodes/*/summary.json"))) == 5
    assert all(p.read_bytes() == data for p, data in originals.items())
    with pytest.raises(ValueError, match="changed"):
        run(replace(config, precision="fp16"), tmp_path)


def test_program_errors_remain_attempts_not_robot_failures(fake_backend, tmp_path):
    config = Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    config = replace(config, record_video=False)
    env_class, _ = fake_backend

    def fail(self, action):
        raise RuntimeError("renderer failed")

    env_class.step = fail
    with pytest.raises(RuntimeError, match="renderer failed"):
        run(config, tmp_path)
    assert not list(tmp_path.glob("episodes/*/summary.json"))
    errors = list(tmp_path.glob("attempts/*/error.json"))
    assert len(errors) == 1
    assert "renderer failed" in json.loads(errors[0].read_text())["traceback"]
