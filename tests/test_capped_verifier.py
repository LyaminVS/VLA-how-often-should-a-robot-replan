"""Independent cap replay must reject plausible but incorrectly capped evidence."""

import copy
import json
import math
from types import SimpleNamespace

import pytest

from capped_replan.config import CappedEpisode
from capped_replan.core import run_episode
from capped_replan.selector import CappedDVACSelector
from scripts.verify_capped import verify_episode, verify_reference_paths
from scripts.verify_dvac import sha256
from vla_replan.core import Observation


class FakeEnvironment:
    def reset(self, episode):
        self.steps = 0
        return self.observation(), {
            "initial_state_sha256": "a" * 64,
            "initial_observation_sha256": "b" * 64,
        }

    def observation(self):
        return Observation({"instruction": "test"}, [self.steps / 100, 0, 0], [0, 0])

    def step(self, action):
        self.steps += 1
        return self.observation(), False, False


class FakePolicy:
    def reset(self):
        self.call = 0

    def predict(self, observation, seed):
        # Prior threshold=1. Call2 crosses at j=3 beyond cap2; call3 never crosses.
        variance = [[1, 1, 1, 1], [0, 0, 0, 4], [0, 0, 0, 0]][min(self.call, 2)]
        self.call += 1
        clean = [[[sign * math.sqrt(value)] + [0.0] * 6 for value in variance]
                 for sign in (-1, 1)]
        self.last_diagnostics = {
            "clean_estimates": clean, "fm_times": [1.0, .5], "variance": variance,
            "velocity_evaluations": 2, "action_dimensions": 7,
            "statistics_dtype": "float64", "diagnostics_finalize_s": 0.0,
        }
        return [[j * .6, 0, 0, 0, 0, 0, .5] for j in range(4)]


@pytest.fixture
def capped_evidence():
    cfg = SimpleNamespace(
        base=SimpleNamespace(chunk_size=4, num_denoising_steps=2, control_hz=20),
        dvac=SimpleNamespace(tail_length=2, window_size=2, alpha=0, min_steps=1),
    )
    episode = CappedEpisode("libero_object", "test", 0, 11, "dvac_cap_2", 5)
    summary, rows, chunks = run_episode(
        FakeEnvironment(), FakePolicy(), episode, chunk_size=4,
        selector=CappedDVACSelector(2, horizon=4, tail_length=2, window_size=2, alpha=0),
    )
    summary["episode_wall_s"] = summary["rollout_wall_s"]
    return summary, rows, chunks, cfg


def test_cap_replay_covers_bootstrap_crossing_beyond_cap_and_no_crossing(capped_evidence):
    summary, rows, chunks, cfg = capped_evidence
    assert [chunk["decision"]["raw_chosen_k"] for chunk in chunks] == [1, 3, 4]
    assert [chunk["decision"]["chosen_k"] for chunk in chunks] == [1, 2, 2]
    assert [chunk["decision"]["cap_applied"] for chunk in chunks] == [False, True, True]
    assert [chunk["step"] for chunk in chunks] == [0, 1, 3]
    compact, result = verify_episode(summary, rows, chunks, cfg)
    assert result["steps"] == 5
    assert result["max_variance_error"] == 0
    assert compact[1]["decision"]["first_crossing"] == 3


@pytest.mark.parametrize(
    ("call", "field", "wrong"),
    [(1, "raw_chosen_k", 2), (1, "first_crossing", 2), (1, "chosen_k", 3),
     (1, "execution_cap", 3), (1, "cap_applied", False),
     (2, "chosen_k", 4), (2, "cap_applied", False), (0, "raw_chosen_k", 2),
     (0, "chosen_k", 2), (0, "cap_applied", True)],
)
def test_wrong_cap_branch_is_rejected_even_if_logged_actions_are_plausible(capped_evidence, call, field, wrong):
    summary, rows, chunks, cfg = copy.deepcopy(capped_evidence)
    chunks[call]["decision"][field] = wrong
    with pytest.raises(ValueError):
        verify_episode(summary, rows, chunks, cfg)


def test_full_horizon_variance_is_required_even_when_cap_is_small(capped_evidence):
    summary, rows, chunks, cfg = copy.deepcopy(capped_evidence)
    chunks[1]["diagnostics"]["variance"] = [0, 0]
    with pytest.raises(ValueError, match="vector length"):
        verify_episode(summary, rows, chunks, cfg)


def test_row_cap_metadata_must_match_the_decision(capped_evidence):
    summary, rows, chunks, cfg = copy.deepcopy(capped_evidence)
    rows[2]["cap_applied"] = False
    with pytest.raises(ValueError, match="capped step metadata"):
        verify_episode(summary, rows, chunks, cfg)


def test_method_cannot_silently_use_a_different_cap(capped_evidence):
    summary, rows, chunks, cfg = copy.deepcopy(capped_evidence)
    summary["execution_cap"] = 3
    with pytest.raises(ValueError, match="identity"):
        verify_episode(summary, rows, chunks, cfg)


def test_cap_count_in_summary_is_recomputed(capped_evidence):
    summary, rows, chunks, cfg = copy.deepcopy(capped_evidence)
    summary["cap_applied_calls"] = 1
    with pytest.raises(ValueError, match="cap count"):
        verify_episode(summary, rows, chunks, cfg)


def reference_fixture(tmp_path):
    actual = tmp_path / "relocated"
    actual.mkdir()
    (actual / "manifest.json").write_text(json.dumps({"source": "fixed"}))
    (actual / "chunks.jsonl").write_text('{}\n')
    recorded = "/home/example/project/outputs/compact"
    audit = {"roots": [recorded], "files": {
        recorded + "/manifest.json": sha256(actual / "manifest.json"),
        recorded + "/chunks.jsonl": sha256(actual / "chunks.jsonl"),
    }}
    return actual, recorded, audit


def test_reference_relocation_matches_manifest_and_all_declared_hashes(tmp_path):
    actual, recorded, audit = reference_fixture(tmp_path)
    mapping, files = verify_reference_paths(audit, [actual])
    assert list(mapping.values()) == [actual]
    assert set(files) == {actual / "manifest.json", actual / "chunks.jsonl"}


def test_reference_relocation_rejects_changed_trace(tmp_path):
    actual, recorded, audit = reference_fixture(tmp_path)
    (actual / "chunks.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="checksum"):
        verify_reference_paths(audit, [actual])


def test_reference_relocation_cannot_escape_declared_root(tmp_path):
    actual, recorded, audit = reference_fixture(tmp_path)
    secret = tmp_path / "outside.json"
    secret.write_text('{}')
    audit["files"][recorded + "/../outside.json"] = sha256(secret)
    with pytest.raises(ValueError, match="checksum"):
        verify_reference_paths(audit, [actual])


def test_reference_relocation_rejects_ambiguous_manifest_matches(tmp_path):
    actual, recorded, audit = reference_fixture(tmp_path)
    second = tmp_path / "second"
    second.mkdir()
    (second / "manifest.json").write_bytes((actual / "manifest.json").read_bytes())
    audit["roots"].append("/another/source")
    audit["files"]["/another/source/manifest.json"] = sha256(second / "manifest.json")
    with pytest.raises(ValueError, match="uniquely"):
        verify_reference_paths(audit, [actual, second])
