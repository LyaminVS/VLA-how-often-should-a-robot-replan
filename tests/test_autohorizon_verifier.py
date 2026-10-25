"""Recorded evidence must prove attention selection, not merely plausible motion."""

import copy
import json
from types import SimpleNamespace

import pytest

from autohorizon_replan.config import AutoEpisode
from autohorizon_replan.selector import AutoHorizonSelector
from capped_replan.core import run_episode
from scripts.verify_autohorizon import (compare_replay, relocate_reference,
                                        replay_attention, verify_episode)
from scripts.verify_dvac import sha256
from vla_replan.core import Observation

torch = pytest.importorskip("torch")


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
        self.calls = 0

    def predict(self, observation, seed):
        self.calls += 1
        matrix = torch.zeros(10, 10)
        matrix[:, 3 if self.calls == 1 else 4 if self.calls == 2 else 0] = 1
        layers = torch.stack([matrix * 0.5, matrix])
        self.last_diagnostics = {
            "clean_estimates": [[[0.0] * 7 for _ in range(10)] for _ in range(2)],
            "variance": [0.0] * 10, "fm_times": [0.2, 0.1],
            "velocity_evaluations": 10, "action_dimensions": 7,
            "statistics_dtype": "float64", "diagnostics_finalize_s": 0.0,
            "attention": layers.mean(0).tolist(), "attention_layers": layers.tolist(),
            "attention_capture": {
                "selected_fm_step": 3, "layer_indices": [0, 2], "head_count": 8,
                "horizon": 10, "fm_time": 0.8, "statistics_dtype": "float32",
                "extraction": "self_attention_layers_only; mean_heads_then_layers_before_row_normalization",
            },
        }
        return [[index * 0.6, 0, 0, 0, 0, 0, 0.5] for index in range(10)]


class FakeAdapter:
    def __init__(self, policy):
        self.selector = AutoHorizonSelector(horizon=10, execution_cap=3)
        self.policy = policy
        self.execution_cap = 3

    def reset(self):
        self.selector.reset()

    @property
    def last_decision(self):
        return self.selector.last_decision

    def select(self, variance):
        return self.selector.select(self.policy.last_diagnostics["attention"])


@pytest.fixture
def evidence():
    cfg = SimpleNamespace(
        base=SimpleNamespace(chunk_size=10, num_denoising_steps=10, control_hz=20),
        dvac=SimpleNamespace(tail_length=2),
        autohorizon=SimpleNamespace(selected_fm_step=3, hold_thr=0.3, max_entropy_q=0.9),
    )
    episode = AutoEpisode("libero_object", "test", 0, 11, "autohorizon_cap_3", 7)
    policy = FakePolicy()
    summary, rows, chunks = run_episode(
        FakeEnvironment(), policy, episode, chunk_size=10, selector=FakeAdapter(policy))
    summary["episode_wall_s"] = summary["rollout_wall_s"]
    return summary, rows, chunks, cfg


def test_replay_checks_uncapped_rule_cap_and_no_forced_first_action(evidence):
    summary, rows, chunks, cfg = evidence
    assert [chunk["decision"]["raw_chosen_k"] for chunk in chunks] == [4, 10, 1]
    assert [chunk["decision"]["chosen_k"] for chunk in chunks] == [3, 3, 1]
    assert [chunk["step"] for chunk in chunks] == [0, 3, 6]
    compact, result = verify_episode(summary, rows, chunks, cfg)
    assert result["steps"] == 7 and result["attention_values"] == 300
    assert summary["bootstrap_calls"] == 0
    assert compact[0]["decision"]["diagnostics"]["N_backward"] == 4


@pytest.mark.parametrize("field,wrong", [
    ("chosen_k", 1), ("raw_chosen_k", 10), ("bootstrap", True),
    ("execution_cap", 5), ("cap_applied", False), ("selected_fm_step", 2),
    ("upstream_commit", "different"), ("selector_device", "cuda"),
    ("selector_dtype", "float64"), ("run_len", 2), ("window_calls", 1),
])
def test_forged_decision_is_rejected(evidence, field, wrong):
    summary, rows, chunks, cfg = copy.deepcopy(evidence)
    chunks[0]["decision"][field] = wrong
    with pytest.raises(ValueError, match="[Rr]eplay"):
        verify_episode(summary, rows, chunks, cfg)


@pytest.mark.parametrize("field,wrong", [
    ("N_backward", 7), ("N_forward", 3), ("stop_forward", 0),
    ("stop_backward", 1), ("join_row", None), ("full_coverage", True),
])
def test_wrong_official_intermediate_is_rejected_even_if_cap_hides_it(evidence, field, wrong):
    summary, rows, chunks, cfg = copy.deepcopy(evidence)
    chunks[0]["decision"]["diagnostics"][field] = wrong
    with pytest.raises(ValueError, match="[Rr]eplay"):
        verify_episode(summary, rows, chunks, cfg)


@pytest.mark.parametrize("field,wrong", [
    ("selected_fm_step", 2), ("fm_time", 0.7), ("horizon", 9),
    ("layer_indices", [0, 0]), ("head_count", 0),
    ("extraction", "mean_across_fm_steps"), ("statistics_dtype", "float64"),
])
def test_wrong_attention_capture_protocol_is_rejected(evidence, field, wrong):
    summary, rows, chunks, cfg = copy.deepcopy(evidence)
    chunks[0]["diagnostics"]["attention_capture"][field] = wrong
    with pytest.raises(ValueError):
        verify_episode(summary, rows, chunks, cfg)


def test_recorded_attention_must_equal_mean_of_retained_layer_matrices(evidence):
    summary, rows, chunks, cfg = copy.deepcopy(evidence)
    chunks[0]["diagnostics"]["attention_layers"][0][0][3] = 0.2
    with pytest.raises(ValueError, match="layer mean"):
        verify_episode(summary, rows, chunks, cfg)


def test_complete_future_matrix_is_required_under_a_small_cap(evidence):
    summary, rows, chunks, cfg = copy.deepcopy(evidence)
    chunks[0]["diagnostics"]["attention"] = [row[:3] for row in chunks[0]["diagnostics"]["attention"][:3]]
    with pytest.raises(ValueError, match="attention horizon"):
        verify_episode(summary, rows, chunks, cfg)


def test_action_and_noise_schedule_are_checked(evidence):
    for kind in ("action", "sampling_seed"):
        summary, rows, chunks, cfg = copy.deepcopy(evidence)
        if kind == "action":
            rows[1]["action"][0] += 0.01
        else:
            chunks[1]["sampling_seed"] += 1
        with pytest.raises(ValueError):
            verify_episode(summary, rows, chunks, cfg)


def test_no_hidden_dvac_bootstrap_can_be_reported(evidence):
    summary, rows, chunks, cfg = copy.deepcopy(evidence)
    summary["bootstrap_calls"] = 1
    with pytest.raises(ValueError, match="bootstrap"):
        verify_episode(summary, rows, chunks, cfg)


def test_scalar_and_bool_types_are_not_interchangeable():
    with pytest.raises(ValueError, match="Replay"):
        compare_replay({"index": True}, {"index": 1})
    with pytest.raises(ValueError, match="inventory"):
        compare_replay({"index": 1, "extra": 1}, {"index": 1})


def test_backward_mapping_replay_is_independent_of_production_selector(monkeypatch):
    def unusable(*args, **kwargs):
        raise AssertionError("The verifier imported the production selector")
    monkeypatch.setattr(AutoHorizonSelector, "select", unusable)
    attention = [[float(column == 3) for column in range(10)] for _ in range(10)]
    result = replay_attention(attention, horizon=10, execution_cap=10)
    assert result["raw_chosen_k"] == 4
    assert result["diagnostics"]["N_backward"] == 4
    assert result["diagnostics"]["full_coverage"] is False


def test_attention_rows_must_have_positive_finite_probability_mass():
    for attention in ([[0.0] * 2] * 2, [[1.0, -1.0]] * 2,
                      [[float("nan"), 1.0]] * 2, [[1e100, 1e100]] * 2):
        with pytest.raises(ValueError):
            replay_attention(attention, horizon=2, execution_cap=2)


def reference_fixture(tmp_path):
    root = tmp_path / "relocated"
    root.mkdir()
    for name in ("manifest.json", "invariance.json"):
        (root / name).write_text(json.dumps({"source": name}))
    audit = {"root": "/home/example/outputs/capped", "files": {
        name: sha256(root / name) for name in ("manifest.json", "invariance.json")}}
    return root, audit


def test_reference_can_be_relocated_with_unchanged_hashes(tmp_path):
    root, audit = reference_fixture(tmp_path)
    actual, files = relocate_reference(audit, root)
    assert actual == root and files == {"manifest.json", "invariance.json"}


def test_changed_reference_evidence_is_rejected(tmp_path):
    root, audit = reference_fixture(tmp_path)
    (root / "invariance.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        relocate_reference(audit, root)


def test_reference_cannot_escape_root(tmp_path):
    root, audit = reference_fixture(tmp_path)
    other = tmp_path / "outside.json"
    other.write_text("{}")
    audit["files"]["../outside.json"] = sha256(other)
    with pytest.raises(ValueError, match="path"):
        relocate_reference(audit, root)
