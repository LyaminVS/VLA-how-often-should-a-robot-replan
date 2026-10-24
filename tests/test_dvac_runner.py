"""Integrity gates and paired collection behavior using lightweight simulator fakes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_replan import policy as policy_module
from adaptive_replan import runner
from adaptive_replan.config import AdaptiveExperiment
from vla_replan import backend
from vla_replan.core import Observation


@pytest.fixture
def config():
    original = AdaptiveExperiment.load(Path(__file__).resolve().parents[1] / "configs/dvac.json")
    base = replace(
        original.base, tasks=(replace(original.base.tasks[0], max_steps=3),),
        intervals=(5,), initial_states=(15,), policy_seeds=(11, 22),
        record_video=False, warmup_calls=1,
    )
    return replace(original, base=base, methods=("fixed_5", "dvac"))


@pytest.fixture
def fake_runtime(monkeypatch):
    class Environment:
        instances = 0
        closed = 0

        def __init__(self, base):
            type(self).instances += 1

        def reset(self, episode):
            self.steps = 0
            return Observation({"instruction": "TEST ONLY"}, [0, 0, 0], [0, 0]), {
                "initial_state_sha256": f"state-{episode.state}",
                "initial_observation_sha256": f"observation-{episode.state}",
            }

        def step(self, action):
            self.steps += 1
            return Observation(None, [0, 0, 0], [0, 0]), self.steps == 3, False

        def close(self):
            type(self).closed += 1

    class Policy:
        def __init__(self, config, device="cuda", tail_length=5):
            self.runtime = {"device": "TEST_ONLY"}

        def sampler_manifest(self):
            return {"test_sampler.py": "test_sampler_hash"}

        def reset(self):
            pass

        def predict(self, observation, seed):
            self.last_diagnostics = {
                "clean_estimates": [[[0.0] * 7 for _ in range(50)] for _ in range(5)],
                "fm_times": [0.5, 0.4, 0.3, 0.2, 0.1],
                "variance": [0.0] * 50,
            }
            return [[0.0] * 7 for _ in range(50)]

        def peak_memory(self):
            return 0

        def reset_peak_memory(self):
            pass

    monkeypatch.setattr(backend, "LiberoEnvironment", Environment)
    monkeypatch.setattr(policy_module, "InstrumentedSmolVLAPolicy", Policy)
    monkeypatch.setattr(runner, "runtime_metadata", lambda: {"test_only": True})
    monkeypatch.setattr(runner, "git_metadata", lambda: {"test_only": True})
    monkeypatch.setattr(runner, "source_manifest", lambda: {"test_source.py": "test_source_hash"})
    return Environment, Policy


@pytest.fixture
def check_path(tmp_path, config, fake_runtime):
    path = tmp_path / "check.json"
    path.write_text(json.dumps({
        "passed": True,
        "checks": [
            {"suite": task.suite, "task": task.name,
             "exact_action_equality": True, "max_absolute_action_error": 0.0}
            for task in config.base.tasks
        ],
        "config_fingerprint": config.fingerprint,
        "source_manifest": {"test_source.py": "test_source_hash"},
        "sampler_manifest": {"test_sampler.py": "test_sampler_hash"},
        "runtime": {"test_only": True},
        "hardware": {"device": "TEST_ONLY"},
    }))
    return path


def test_partial_resume_preserves_all_completed_evidence(config, check_path, tmp_path):
    output = tmp_path / "collection"
    runner.run(config, output, limit=1, integrity_check=check_path)
    before = {path: path.read_bytes() for path in output.glob("episodes/*/*")}
    runner.run(config, output, integrity_check=check_path)
    assert len(list(output.glob("episodes/*/summary.json"))) == 4
    assert all(path.read_bytes() == data for path, data in before.items())
    assert (output / "invariance.json").read_bytes() == check_path.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["invariance_check_sha256"] == runner.file_sha256(check_path)
    for path in output.glob("episodes/*/summary.json"):
        summary = json.loads(path.read_text())
        assert summary["manifest_sha256"] == runner.file_sha256(output / "manifest.json")
        assert set(summary["evidence_sha256"]) == {"steps.jsonl", "chunks.jsonl"}


@pytest.mark.parametrize("field,value", [
    ("passed", False),
    ("config_fingerprint", "different"),
    ("source_manifest", {"test_source.py": "different"}),
    ("sampler_manifest", {"test_sampler.py": "different"}),
    ("runtime", {"test_only": False}),
    ("hardware", {"device": "different"}),
    ("checks", []),
])
def test_invalid_precollection_evidence_stops_before_environment_creation(
    config, check_path, tmp_path, fake_runtime, field, value,
):
    check = json.loads(check_path.read_text())
    check[field] = value
    check_path.write_text(json.dumps(check))
    with pytest.raises(ValueError, match="passing invariance check"):
        runner.run(config, tmp_path / "collection", integrity_check=check_path)
    assert fake_runtime[0].instances == 0


@pytest.mark.parametrize("change", ["nonidentical", "nonzero_error", "wrong_task", "duplicate_task"])
def test_each_expected_task_must_have_exactly_equal_actions(config, check_path, tmp_path, change):
    check = json.loads(check_path.read_text())
    if change == "nonidentical":
        check["checks"][0]["exact_action_equality"] = False
    elif change == "nonzero_error":
        check["checks"][0]["max_absolute_action_error"] = 1e-12
    elif change == "wrong_task":
        check["checks"][0]["task"] = "another_task"
    else:
        check["checks"].append(dict(check["checks"][0]))
    check_path.write_text(json.dumps(check))
    with pytest.raises(ValueError, match="passing invariance check"):
        runner.run(config, tmp_path / "collection", integrity_check=check_path)


def test_replacing_passing_check_with_different_bytes_blocks_resume(config, check_path, tmp_path):
    output = tmp_path / "collection"
    runner.run(config, output, limit=1, integrity_check=check_path)
    check = json.loads(check_path.read_text())
    check["additional_note"] = "This is a different check artifact, even though the required checks pass."
    check_path.write_text(json.dumps(check))
    with pytest.raises(ValueError, match="invariance-check evidence changed"):
        runner.run(config, output, integrity_check=check_path)


def test_altered_copied_check_blocks_resume(config, check_path, tmp_path):
    output = tmp_path / "collection"
    runner.run(config, output, limit=1, integrity_check=check_path)
    copied = output / "invariance.json"
    copied.write_text(copied.read_text() + "\n")
    with pytest.raises(ValueError, match="Stored invariance-check evidence changed"):
        runner.run(config, output, integrity_check=check_path)


@pytest.mark.parametrize("filename", ["steps.jsonl", "chunks.jsonl"])
def test_modified_raw_evidence_is_never_skipped(config, check_path, tmp_path, filename):
    output = tmp_path / "collection"
    runner.run(config, output, limit=1, integrity_check=check_path)
    path = next(output.glob(f"episodes/*/{filename}"))
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="Stored evidence changed"):
        runner.run(config, output, integrity_check=check_path)


def test_missing_trace_hash_blocks_resume(config, check_path, tmp_path):
    output = tmp_path / "collection"
    runner.run(config, output, limit=1, integrity_check=check_path)
    path = next(output.glob("episodes/*/summary.json"))
    summary = json.loads(path.read_text())
    del summary["evidence_sha256"]["chunks.jsonl"]
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Missing evidence hashes"):
        runner.run(config, output, integrity_check=check_path)


def test_changed_episode_identity_blocks_resume(config, check_path, tmp_path):
    output = tmp_path / "collection"
    runner.run(config, output, limit=1, integrity_check=check_path)
    path = next(output.glob("episodes/*/summary.json"))
    summary = json.loads(path.read_text())
    summary["policy_seed"] += 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Stored episode differs"):
        runner.run(config, output, integrity_check=check_path)


def test_physical_start_must_match_across_policy_seeds(config, check_path, tmp_path, fake_runtime, monkeypatch):
    environment = fake_runtime[0]
    original = environment.reset

    def changed_reset(self, episode):
        observation, info = original(self, episode)
        info["initial_observation_sha256"] = f"observation-seed-{episode.policy_seed}"
        return observation, info

    monkeypatch.setattr(environment, "reset", changed_reset)
    output = tmp_path / "collection"
    with pytest.raises(RuntimeError, match="Paired initial observations differ"):
        runner.run(config, output, integrity_check=check_path)
    # Each randomized block contains both methods for one policy seed.
    assert len(list(output.glob("episodes/*/summary.json"))) == 2
    assert len(list(output.glob("attempts/*/error.json"))) == 1
    assert not list(output.glob("attempts/*/summary.json"))
    assert environment.closed == 1


def test_resume_rejects_conflicting_stored_reset_fingerprints(config, check_path, tmp_path, fake_runtime):
    output = tmp_path / "collection"
    runner.run(config, output, limit=2, integrity_check=check_path)
    path = next(output.glob("episodes/*/summary.json"))
    summary = json.loads(path.read_text())
    summary["initial_observation_sha256"] = "changed"
    path.write_text(json.dumps(summary))
    before_instances = fake_runtime[0].instances
    with pytest.raises(ValueError, match="Stored paired initial observations differ"):
        runner.run(config, output, integrity_check=check_path)
    assert fake_runtime[0].instances == before_instances


def test_source_change_blocks_resume_before_new_inference(config, check_path, tmp_path, monkeypatch):
    output = tmp_path / "collection"
    runner.run(config, output, limit=1, integrity_check=check_path)
    monkeypatch.setattr(runner, "source_manifest", lambda: {"test_source.py": "changed"})
    with pytest.raises(ValueError, match="Configuration, code or software changed"):
        runner.run(config, output, integrity_check=check_path)


def test_runtime_error_remains_an_attempt_not_an_unsuccessful_episode(
    config, check_path, tmp_path, fake_runtime, monkeypatch,
):
    def fail(self, action):
        raise RuntimeError("test simulator failure")

    monkeypatch.setattr(fake_runtime[0], "step", fail)
    output = tmp_path / "collection"
    with pytest.raises(RuntimeError, match="test simulator failure"):
        runner.run(config, output, integrity_check=check_path)
    assert not list(output.glob("episodes/*/summary.json"))
    path = next(output.glob("attempts/*/error.json"))
    assert "test simulator failure" in json.loads(path.read_text())["traceback"]
    assert fake_runtime[0].closed == 1
