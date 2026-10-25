"""JSON manifest regressions without model loading or simulated robot actions."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autohorizon_replan import policy as policy_module
from autohorizon_replan import runner
from autohorizon_replan.config import AutoHorizonExperiment
from capped_replan import integrity as capped_integrity
from capped_replan.config import CappedEpisode, CappedExperiment
from vla_replan import backend

ROOT = Path(__file__).resolve().parents[1]
TEST_SOURCES = {"test_only_source.py": "test_only_source_hash"}
TEST_RUNTIME = {"test_only": True}
TEST_HARDWARE = {"device": "TEST_ONLY_NO_GPU"}
TEST_SAMPLER = {"test_only_sampler.py": "test_only_sampler_hash"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def config():
    original = AutoHorizonExperiment.load(ROOT / "configs/autohorizon.json")
    base = replace(original.base, tasks=(replace(original.base.tasks[0], max_steps=2),),
                   initial_states=(0,), policy_seeds=(11,))
    result = replace(original, base=base, methods=("autohorizon_cap_50",))
    result.validate()
    return result


@pytest.fixture
def historical_root(tmp_path, monkeypatch, config):
    monkeypatch.setattr(capped_integrity, "source_manifest", lambda: dict(TEST_SOURCES))
    root = tmp_path / "historical"
    old_config = CappedExperiment(config.base, ("dvac_cap_50",), config.dvac)
    old_episode = CappedEpisode(
        config.base.tasks[0].suite, config.base.tasks[0].name, 0, 11, "dvac_cap_50", 2)
    manifest = {
        "data_kind": "real_libero", "capped_source": TEST_SOURCES,
        "config": old_config.asdict(), "config_fingerprint": old_config.fingerprint,
        "runtime": TEST_RUNTIME, "hardware": TEST_HARDWARE, "upstream_sampler": TEST_SAMPLER,
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "invariance.json", {"passed": True, "test_only": True})
    directory = root / "episodes" / old_episode.id
    directory.mkdir(parents=True)
    first = {"step": 0, "sampling_seed": 11 * 1_000_003, "actions": [[0.0] * 7 for _ in range(50)]}
    (directory / "chunks.jsonl").write_text(json.dumps(first) + "\n")
    (directory / "steps.jsonl").write_text("{}\n")
    (directory / "video.mp4").write_bytes(b"TEST_ONLY_NOT_A_VIDEO")
    summary = {
        **old_episode.asdict(), "episode_id": old_episode.id, "pair_id": old_episode.pair_id,
        "env_seed": old_episode.env_seed, "data_kind": "real_libero",
        "manifest_sha256": runner.file_sha256(root / "manifest.json"),
        "config_fingerprint": old_config.fingerprint,
        "initial_state_sha256": "a" * 64, "initial_observation_sha256": "b" * 64,
        "evidence_sha256": {name: runner.file_sha256(directory / name)
                            for name in ("chunks.jsonl", "steps.jsonl", "video.mp4")},
    }
    write_json(directory / "summary.json", summary)
    return root


def test_json_loaded_historical_tasks_match_dataclass_tuple(config, historical_root):
    # JSON lists and dataclass tuples are semantically the same configuration.
    assert isinstance(config.base.asdict()["tasks"], tuple)
    stored = json.loads((historical_root / "manifest.json").read_text())
    assert isinstance(stored["config"]["base"]["tasks"], list)
    references, audit, manifest = runner.load_references(historical_root, config)
    episode = config.episodes()[0]
    assert set(references) == {episode.pair_id}
    assert references[episode.pair_id]["actions"] == [[0.0] * 7 for _ in range(50)]
    assert manifest == stored
    assert audit["files"]["manifest.json"] == runner.file_sha256(historical_root / "manifest.json")
    assert len(audit["files"]) == 6  # manifest, check, summary, and three episode artifacts


def test_a_real_historical_task_limit_change_still_fails(config, historical_root):
    manifest_path = historical_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["base"]["tasks"][0]["max_steps"] += 1
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="tasks or episode limits"):
        runner.load_references(historical_root, config)


@pytest.mark.parametrize("missing", ["chunks.jsonl", "steps.jsonl", "video.mp4"])
def test_missing_historical_evidence_hash_fails(config, historical_root, missing):
    summary_path = next(historical_root.glob("episodes/*/summary.json"))
    summary = json.loads(summary_path.read_text())
    del summary["evidence_sha256"][missing]
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="all three evidence"):
        runner.load_references(historical_root, config)


@pytest.fixture
def completed_collection(tmp_path, monkeypatch, config, historical_root):
    class NoInferencePolicy:
        instances = 0

        def __init__(self, *args, **kwargs):
            type(self).instances += 1
            self.runtime = dict(TEST_HARDWARE)

        def sampler_manifest(self):
            return dict(TEST_SAMPLER)

        def predict(self, *args, **kwargs):
            raise AssertionError("A complete collection must not run inference")

    class NoRolloutEnvironment:
        instances = 0
        closes = 0

        def __init__(self, *args, **kwargs):
            type(self).instances += 1

        def reset(self, *args, **kwargs):
            raise AssertionError("A complete collection must not reset the simulator")

        def close(self):
            type(self).closes += 1

    monkeypatch.setattr(policy_module, "AutoHorizonSmolVLAPolicy", NoInferencePolicy)
    monkeypatch.setattr(backend, "LiberoEnvironment", NoRolloutEnvironment)
    monkeypatch.setattr(runner, "source_manifest", lambda: dict(TEST_SOURCES))
    monkeypatch.setattr(runner, "runtime_metadata", lambda: dict(TEST_RUNTIME))
    _, audit, _ = runner.load_references(historical_root, config)
    check_path = tmp_path / "matching-check.json"
    evidence = {
        "passed": True, "source_manifest": TEST_SOURCES, "config_fingerprint": config.fingerprint,
        "runtime": TEST_RUNTIME, "hardware": TEST_HARDWARE, "sampler_manifest": TEST_SAMPLER,
        "reference_sources": audit,
        "checks": [{"suite": task.suite, "task": task.name,
                    "exact_action_equality": True, "max_absolute_action_error": 0}
                   for task in config.base.tasks],
    }
    write_json(check_path, evidence)
    output = tmp_path / "complete"
    output.mkdir()
    manifest = {
        "config_fingerprint": config.fingerprint, "config": config.asdict(),
        "source_manifest": TEST_SOURCES, "runtime": TEST_RUNTIME, "hardware": TEST_HARDWARE,
        "upstream_sampler": TEST_SAMPLER, "reference_sources": audit,
        "invariance_check_sha256": runner.file_sha256(check_path),
        "planned_episode_ids": [episode.id for episode in config.episodes()],
        "data_kind": "real_libero", "experiment_kind": "autohorizon_comparison", "schema_version": 1,
    }
    write_json(output / "manifest.json", manifest)
    (output / "invariance.json").write_bytes(check_path.read_bytes())
    for episode in config.episodes():
        directory = output / "episodes" / episode.id
        directory.mkdir(parents=True)
        for name in ("steps.jsonl", "chunks.jsonl", "video.mp4"):
            (directory / name).write_bytes(b"TEST_ONLY_RESUME_ARTIFACT")
        summary = {
            **episode.asdict(), "episode_id": episode.id, "pair_id": episode.pair_id,
            "config_fingerprint": config.fingerprint,
            "manifest_sha256": runner.file_sha256(output / "manifest.json"),
            "evidence_sha256": {name: runner.file_sha256(directory / name)
                                for name in ("steps.jsonl", "chunks.jsonl", "video.mp4")},
        }
        write_json(directory / "summary.json", summary)
    return output, check_path, NoInferencePolicy, NoRolloutEnvironment


def test_resume_compares_canonical_json_config_and_preserves_complete_evidence(
    config, historical_root, completed_collection,
):
    output, check_path, policy, environment = completed_collection
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    runner.run(config, output, check_path, historical_root)
    runner.run(config, output, check_path, historical_root)
    assert policy.instances == environment.instances == environment.closes == 2
    assert all((output / name).read_bytes() == content for name, content in before.items())
    assert not list(output.glob("attempts/*"))


def test_resume_still_rejects_a_real_stored_configuration_difference(
    config, historical_root, completed_collection,
):
    output, check_path, _, environment = completed_collection
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["base"]["settle_steps"] += 1
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Resume configuration/code/runtime changed"):
        runner.run(config, output, check_path, historical_root)
    assert environment.instances == 0
