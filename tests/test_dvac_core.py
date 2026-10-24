"""Behavioral checks for adaptive deadlines, seeds, termination and action logging."""

from dataclasses import replace

import pytest

from adaptive_replan.config import AdaptiveEpisode
from adaptive_replan.core import run_episode
from vla_replan.core import Observation, sampling_seed


class FakeEnvironment:
    def __init__(self, success_step=None, terminal_step=None):
        self.success_step = success_step
        self.terminal_step = terminal_step
        self.actions = []

    def observation(self):
        return Observation({"instruction": "test instruction"}, [len(self.actions) / 100, 0, 0], [0, 0])

    def reset(self, episode):
        self.actions = []
        return self.observation(), {"initial_state_sha256": "state", "initial_observation_sha256": "observation"}

    def step(self, action):
        self.actions.append(action)
        n = len(self.actions)
        return self.observation(), n == self.success_step, n == self.terminal_step


class FakePolicy:
    def __init__(self, horizon=6, invalid_variance=False):
        self.horizon = horizon
        self.invalid_variance = invalid_variance
        self.seeds = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.seeds = []

    def predict(self, observation, seed):
        self.seeds.append(seed)
        self.last_diagnostics = {
            "clean_estimates": [[[0.0] * 7 for _ in range(self.horizon)] for _ in range(5)],
            "fm_times": [0.5, 0.4, 0.3, 0.2, 0.1],
            "variance": [float("nan") if self.invalid_variance else 0.0] * self.horizon,
        }
        return [[0.1 * (j + 1), 0, 0, 0, 0, 0, 2.0] for j in range(self.horizon)]


class ScheduledSelector:
    def __init__(self, horizons):
        self.horizons = horizons
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.calls = 0

    def select(self, variance):
        k = self.horizons[self.calls]
        self.calls += 1
        return {
            "chosen_k": k, "threshold": None if self.calls == 1 else 0.0,
            "first_crossing": None, "window_calls": self.calls - 1,
            "window_mean": None, "window_std": None, "bootstrap": self.calls == 1,
        }


def episode(method="dvac", max_steps=6):
    return AdaptiveEpisode("libero_object", "test_task", 15, 11, method, max_steps)


def test_adaptive_deadlines_execute_only_selected_final_prefix_and_stop_on_success():
    env = FakeEnvironment(success_step=5)
    policy = FakePolicy()
    selector = ScheduledSelector([1, 3, 2])
    summary, rows, chunks = run_episode(env, policy, episode(), selector=selector, chunk_size=6)
    assert [p["step"] for p in chunks] == [0, 1, 4]
    assert [p["executed_steps"] for p in chunks] == [1, 3, 1]
    assert [p["decision"]["chosen_k"] for p in chunks] == [1, 3, 2]
    assert [r["chunk_index"] for r in rows] == [0, 0, 1, 2, 0]
    assert [r["next_replan_step"] for r in rows] == [1, 4, 4, 4, 6]
    assert [a[0] for a in env.actions] == pytest.approx([0.1, 0.1, 0.2, 0.3, 0.1])
    assert all(a[6] == 1.0 for a in env.actions)
    assert chunks[0]["actions"][0][6] == 2.0  # Preserve final raw predictions before clipping.
    assert policy.seeds == [sampling_seed(11, s) for s in [0, 1, 4]]
    assert summary["success"] is True
    assert summary["steps"] == 5
    assert summary["model_calls"] == 3
    assert summary["clipped_action_values"] == 5
    assert summary["mean_chosen_k"] == 2
    assert selector.reset_count == policy.reset_count == 1


def test_fixed_control_still_collects_identical_diagnostic_fields():
    policy = FakePolicy()
    summary, rows, chunks = run_episode(FakeEnvironment(), policy, episode("fixed_2", 5), chunk_size=6)
    assert [p["step"] for p in chunks] == [0, 2, 4]
    assert [p["executed_steps"] for p in chunks] == [2, 2, 1]
    assert all("diagnostics" in p for p in chunks)
    assert all(p["selector_s"] == 0 for p in chunks)
    assert all(p["decision"]["chosen_k"] == 2 for p in chunks)
    assert all(p["decision"]["threshold"] is None for p in chunks)
    assert summary["bootstrap_calls"] == 0
    assert summary["termination"] == "time_limit"
    assert len(rows) == 5


def test_terminal_at_horizon_boundary_prevents_extra_model_call():
    policy = FakePolicy()
    summary, rows, chunks = run_episode(
        FakeEnvironment(terminal_step=2), policy, episode("fixed_2"), chunk_size=6,
    )
    assert len(chunks) == 1
    assert len(rows) == 2
    assert summary["termination"] == "terminated"
    assert summary["success"] is False


def test_selector_history_resets_between_episodes():
    selector = ScheduledSelector([1, 5])
    policy = FakePolicy()
    for state in [15, 16]:
        _, _, chunks = run_episode(
            FakeEnvironment(success_step=2), policy, replace(episode(), state=state),
            selector=selector, chunk_size=6,
        )
        assert chunks[0]["decision"]["bootstrap"] is True
        assert chunks[0]["decision"]["chosen_k"] == 1
    assert selector.reset_count == 2


def test_video_observations_align_with_action_then_final_outcome():
    recorded = []
    summary, rows, _ = run_episode(
        FakeEnvironment(success_step=2), FakePolicy(), episode("fixed_2"), chunk_size=6,
        record=lambda frame, row, outcome: recorded.append((row.copy(), outcome)),
    )
    assert [r[0]["step"] for r in recorded] == [0, 1, 2]
    assert [r[1] for r in recorded] == [None, None, True]
    assert recorded[0][0]["replanned"] is True
    assert recorded[0][0]["instruction"] == "test instruction"
    assert recorded[-1][0]["chosen_k"] == 2


@pytest.mark.parametrize("bad_k", [0, 7, 1.5, True])
def test_invalid_selected_horizon_is_rejected_before_motion(bad_k):
    env = FakeEnvironment()
    with pytest.raises(ValueError, match="invalid execution horizon"):
        run_episode(env, FakePolicy(), episode(), selector=ScheduledSelector([bad_k]), chunk_size=6)
    assert env.actions == []


def test_nonfinite_variance_rejected_before_motion():
    env = FakeEnvironment()
    with pytest.raises(ValueError, match="denoising variance"):
        run_episode(env, FakePolicy(invalid_variance=True), episode("fixed_2"), chunk_size=6)
    assert env.actions == []


def test_dvac_requires_selector():
    with pytest.raises(ValueError, match="requires a selector"):
        run_episode(FakeEnvironment(), FakePolicy(), episode(), chunk_size=6)
