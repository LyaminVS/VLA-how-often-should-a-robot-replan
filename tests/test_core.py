from dataclasses import replace
from pathlib import Path

import pytest

from vla_replan.config import Episode, Experiment
from vla_replan.core import Observation, run_episode, sampling_seed

ROOT = Path(__file__).resolve().parents[1]


class FakeEnv:
    """Scripted test double; never a benchmark result."""

    def __init__(self, success_at=None, terminate_at=None):
        self.success_at, self.terminate_at = success_at, terminate_at
        self.actions = []

    def observation(self):
        return Observation(self.step_count, [self.step_count * 0.01, 0, 0], [0, 0])

    def reset(self, episode):
        self.step_count = 0
        self.actions = []
        return self.observation(), {"initial_state_sha256": "test", "initial_observation_sha256": "test"}

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        return self.observation(), self.step_count == self.success_at, self.step_count == self.terminate_at


class FakePolicy:
    def reset(self):
        self.calls = []

    def predict(self, obs, seed):
        self.calls.append((obs.payload, seed))
        return [[obs.payload / 1000 + i / 10000, 0, 0, 0, 0, 0, -1] for i in range(50)]


def episode(k=10, max_steps=103):
    return Episode("libero_object", "test_task", 0, 11, k, max_steps)


@pytest.mark.parametrize("k", [1, 5, 10, 25, 50])
def test_exact_action_execution_and_call_count(k):
    env, policy = FakeEnv(), FakePolicy()
    summary, rows, chunks = run_episode(env, policy, episode(k))
    assert summary["model_calls"] == (103 + k - 1) // k
    assert [step for step, _ in policy.calls] == list(range(0, 103, k))
    assert [r["step"] for r in rows if r["replanned"]] == list(range(0, 103, k))
    assert len(chunks) == summary["model_calls"]
    for step, action in enumerate(env.actions):
        # A new chunk replaces the old chunk, even though the old chunk has unused actions.
        assert action[0] == pytest.approx((step // k * k) / 1000 + (step % k) / 10000)
    assert summary["eef_path_m"] == pytest.approx(1.03)
    assert not summary["success"]
    assert summary["termination"] == "time_limit"


@pytest.mark.parametrize("k", [1, 10, 50])
def test_success_stops_inside_chunk(k):
    env, policy = FakeEnv(success_at=7), FakePolicy()
    summary, rows, _ = run_episode(env, policy, episode(k))
    assert summary["success"] and len(rows) == 7 and len(env.actions) == 7
    assert summary["model_calls"] == (7 + k - 1) // k


def test_failure_termination_and_reset_do_not_leak():
    env, policy = FakeEnv(terminate_at=3), FakePolicy()
    first, _, _ = run_episode(env, policy, episode())
    second, _, _ = run_episode(env, policy, episode())
    assert not first["success"] and first["termination"] == "terminated"
    assert second["steps"] == 3 and second["model_calls"] == 1
    assert policy.calls[0][0] == 0


def test_video_replan_frame_precedes_action_and_has_final_outcome():
    frames = []

    def record(frame, row, outcome):
        frames.append((row["step"], row["replanned"], outcome))

    run_episode(FakeEnv(success_at=6), FakePolicy(), episode(5), record=record)
    assert frames == [
        (0, True, None),
        (1, False, None),
        (2, False, None),
        (3, False, None),
        (4, False, None),
        (5, True, None),
        (6, False, True),
    ]


@pytest.mark.parametrize("bad_chunk", [[], [[0] * 7] * 49, [[0] * 6] * 50, [[float("nan")] * 7] * 50])
def test_invalid_predictions_fail_before_any_action(bad_chunk):
    policy = FakePolicy()
    policy.predict = lambda obs, seed: bad_chunk
    env = FakeEnv()
    with pytest.raises(ValueError):
        run_episode(env, policy, episode())
    assert env.actions == []


def test_clipping_is_counted_not_hidden():
    policy = FakePolicy()
    policy.predict = lambda obs, seed: [[2, 0, 0, 0, 0, 0, 1] for _ in range(50)]
    env = FakeEnv(success_at=2)
    summary, _, _ = run_episode(env, policy, episode())
    assert summary["clipped_action_values"] == 2
    assert summary["gripper_close_commands"] == 1
    assert env.actions[0][0] == 1


def test_plan_has_balanced_treatments_and_reproducible_order():
    cfg = Experiment.load(ROOT / "configs/full.json")
    assert len(cfg.episodes()) == 900
    assert cfg.episodes() == cfg.episodes()
    assert len({e.id for e in cfg.episodes()}) == 900
    assert cfg.budget()["model_calls_upper_bound"] == 70320
    for start in range(0, 900, 5):
        block = cfg.episodes()[start : start + 5]
        assert {e.k for e in block} == {1, 5, 10, 25, 50}
        assert len({e.pair_id for e in block}) == 1
        assert len({e.env_seed for e in block}) == 1


def test_policy_seed_does_not_change_environment_start():
    a = episode()
    b = replace(a, k=50, policy_seed=29)
    assert a.env_seed == b.env_seed
    assert sampling_seed(a.policy_seed, 50) == sampling_seed(a.policy_seed, 50)
    assert sampling_seed(a.policy_seed, 50) != sampling_seed(b.policy_seed, 50)


@pytest.mark.parametrize(
    "changes",
    [
        {"intervals": (0, 1)},
        {"intervals": (1, 51)},
        {"intervals": (1, 1)},
        {"initial_states": (-1,)},
        {"model_revision": "main"},
        {"precision": "bf16"},
    ],
)
def test_invalid_experiment_is_rejected(changes):
    cfg = Experiment.load(ROOT / "configs/smoke.json")
    with pytest.raises(ValueError):
        replace(cfg, **changes).validate()
