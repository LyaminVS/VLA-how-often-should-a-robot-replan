from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vla_replan.config import Episode, Experiment
from vla_replan.core import Observation, run_episode, sampling_seed
from vla_replan.research import choose_interval, research_jobs, run_trial


class Environment:
    """Deterministic test double; never included in experimental results."""

    def observation(self):
        return Observation({"raw": {"step": np.array([self.step_count])}, "instruction": "fixture"},
                           [self.step_count * .01, 0, 0], [0, 0])

    def reset(self, episode):
        self.step_count, self.actions = 0, []
        return self.observation(), {"initial_state_sha256": "fixture", "initial_observation_sha256": "fixture", "instruction": "fixture"}

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        return self.observation(), False, False


class Policy:
    def reset(self):
        self.calls = []

    def predict(self, obs, seed):
        step = int(obs.payload["raw"]["step"][0])
        self.calls.append((step, seed))
        return [[step / 1000 + i / 10000, 0, 0, 0, 0, 0, 1 if step >= 25 else -1] for i in range(50)]


@pytest.fixture
def config():
    return Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")


def episode(k=1, steps=42):
    return Episode("libero_object", "fixture", 4, 11, k, steps)


def test_frozen_adaptive_rule():
    chunk = [[1, 0, 0, 0, 0, 0, -1] for _ in range(50)]
    assert choose_interval(chunk, "adaptive_v1") == (25, "open_gripper_motion")
    chunk[8][6] = 1
    assert choose_interval(chunk, "adaptive_v1")[0] == 5
    for a in chunk:
        a[6] = 1
    assert choose_interval(chunk, "adaptive_v1")[0] == 10
    chunk[9][0] = -1
    assert choose_interval(chunk, "adaptive_v1") == (5, "predicted_translation_turn")


@pytest.mark.parametrize("k", [1, 10, 25])
def test_research_fixed_execution_matches_original_core(config, k):
    env1, env2 = Environment(), Environment()
    a, rows_a, chunks_a = run_episode(env1, Policy(), episode(k))
    b, rows_b, chunks_b = run_trial(env2, Policy(), episode(k), f"fixed{k}", config)
    assert env1.actions == env2.actions
    assert a["steps"] == b["steps"] and a["model_calls"] == b["model_calls"]
    assert [c["sampling_seed"] for c in chunks_a] == [c["sampling_seed"] for c in chunks_b]
    assert [r["chunk_index"] for r in rows_a] == [r["chunk_index"] for r in rows_b]


def test_variable_intervals_follow_selected_lengths(config):
    _, rows, chunks = run_trial(Environment(), Policy(), episode(), "adaptive_v1", config)
    assert [c["step"] for c in chunks] == [0, 25, 35]
    assert [c["selected_k"] for c in chunks] == [25, 10, 10]
    for r in rows:
        c = next(c for c in chunks if c["step"] == r["plan_step"])
        assert r["action"] == c["actions"][r["chunk_index"]]


def test_replay_fork_retains_feedback_and_isolates_noise(config):
    original, rows, chunks = run_trial(Environment(), Policy(), episode(), "fixed1", config)
    ref = {"summary": original, "steps": rows, "chunks": chunks}
    env, policy = Environment(), Policy()
    summary, continued, predicted = run_trial(env, policy, episode(), "constant_noise", config, reference=ref, fork=20)
    assert env.actions[:20] == [r["action"] for r in rows[:20]]
    assert summary["first_chunk_max_abs_error"] == 0
    assert continued[0]["step"] == 20 and summary["executed_steps"] == 22
    assert {c["sampling_seed"] for c in predicted} == {sampling_seed(11, 20)}
    assert [step for step, _ in policy.calls] == list(range(20, 42))
    _, _, committed = run_trial(Environment(), Policy(), episode(), "commit10", config, reference=ref, fork=20)
    assert [c["step"] for c in committed] == [20, 30, 40]
    ref["steps"][5]["observation_sha256"] = "corrupt"
    with pytest.raises(RuntimeError, match="Replay observation differs"):
        run_trial(Environment(), Policy(), episode(), "commit10", config, reference=ref, fork=20)


def test_job_design_is_disjoint_and_balanced(config):
    import json

    plan = json.loads((Path(__file__).resolve().parents[1] / "configs/research_v1.json").read_text())
    full = Experiment.load(Path(__file__).resolve().parents[1] / "configs/compact.json")
    jobs = research_jobs(plan, full)
    assert len(jobs) == 78
    assert sum(phase == "heldout" for phase, _, _ in jobs) == 48
    assert sum(phase == "causal" for phase, _, _ in jobs) == 18
    assert research_jobs(plan, replace(full)) == jobs
    assert set(ep.state for phase, ep, _ in jobs if phase == "development") == {2, 3}


def test_identical_observations_do_not_hide_a_physics_mismatch(config, monkeypatch):
    from vla_replan import research

    monkeypatch.setattr(research, "simulation_hash", lambda env: str(env.step_count))
    summary, steps, chunks = run_trial(Environment(), Policy(), episode(), "fixed1", config)
    reference = {"summary": summary, "steps": steps, "chunks": chunks}
    steps[20]["physics_state_sha256"] = "different_hidden_velocity"
    with pytest.raises(RuntimeError, match="Fork physics states differ"):
        run_trial(Environment(), Policy(), episode(), "commit10", config, reference=reference, fork=20)


def test_seed_replication_does_not_change_physical_reset_seeds():
    import json

    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/research_v1_seed_replication.json").read_text())
    jobs = research_jobs(plan, Experiment.load(root / "configs/compact.json"))
    assert len(jobs) == 24
    assert {phase for phase, _, _ in jobs} == {"heldout"}
    assert {ep.policy_seed for _, ep, _ in jobs} == {29,47}
    physical = {}
    for _, ep, _ in jobs:
        physical.setdefault((ep.suite,ep.task,ep.state),set()).add(ep.env_seed)
    assert len(physical) == 6 and all(len(seeds)==1 for seeds in physical.values())
