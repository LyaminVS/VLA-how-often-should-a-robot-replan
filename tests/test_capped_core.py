import copy

import pytest

from adaptive_replan.config import AdaptiveEpisode
from adaptive_replan.core import run_episode as original_run
from adaptive_replan.selector import DVACSelector
from capped_replan.config import CappedEpisode
from capped_replan.core import SchedulingView, run_episode
from capped_replan.selector import CappedDVACSelector
from vla_replan.core import Observation, sampling_seed


class Environment:
    def __init__(self, success_step=9):
        self.success_step = success_step
        self.actions = []

    def reset(self, episode):
        self.actions = []
        return Observation({"instruction": "TEST ONLY"}, [0, 0, 0], [0, 0]), {
            "initial_state_sha256": "a" * 64, "initial_observation_sha256": "b" * 64,
        }

    def step(self, action):
        self.actions.append(action)
        return Observation(None, [0, 0, 0], [0, 0]), len(self.actions) == self.success_step, False


class Policy:
    def reset(self):
        pass

    def predict(self, obs, seed):
        self.last_diagnostics = {
            "clean_estimates": [[[0.0] * 7 for _ in range(50)] for _ in range(5)],
            "variance": [0.0] * 50, "fm_times": [.5, .4, .3, .2, .1],
        }
        return [[(seed % 100) / 100, 0, 0, 0, 0, 0, -1.0] for _ in range(50)]


def episode(cap=5):
    return CappedEpisode("libero_object", "test_task", 0, 11, f"dvac_cap_{cap}", 20)


def reference(ep):
    return {"pair_id": ep.pair_id, "initial_state_sha256": "a" * 64,
            "initial_observation_sha256": "b" * 64,
            "sampling_seed": sampling_seed(11, 0), "actions": Policy().predict(None, sampling_seed(11, 0))}


def test_scheduling_view_preserves_recorded_identity():
    ep = episode()
    view = SchedulingView(ep)
    assert view.method == "dvac"
    assert view.id == ep.id and view.pair_id == ep.pair_id
    assert view.asdict()["method"] == "dvac_cap_5"
    assert view.asdict()["execution_cap"] == 5


def test_cap_deadlines_and_logged_raw_decisions():
    frames = []
    summary, rows, chunks = run_episode(
        Environment(), Policy(), episode(), selector=CappedDVACSelector(5),
        record=lambda frame, row, result: frames.append((dict(row), result)),
    )
    assert summary["method"] == "dvac_cap_5"
    assert [chunk["step"] for chunk in chunks] == [0, 1, 6]
    assert [chunk["executed_steps"] for chunk in chunks] == [1, 5, 3]
    assert [chunk["decision"]["raw_chosen_k"] for chunk in chunks] == [1, 50, 50]
    assert [chunk["decision"]["chosen_k"] for chunk in chunks] == [1, 5, 5]
    assert summary["cap_applied_calls"] == 2
    assert rows[1]["raw_chosen_k"] == frames[1][0]["raw_chosen_k"] == 50
    assert frames[-1][0]["execution_cap"] == 5


def test_cap_fifty_executes_exact_original_actions_and_replan_boundaries():
    ep = episode(50)
    old = AdaptiveEpisode(ep.suite, ep.task, ep.state, ep.policy_seed, "dvac", ep.max_steps)
    old_env, new_env = Environment(), Environment()
    old_summary, old_rows, old_chunks = original_run(old_env, Policy(), old, selector=DVACSelector())
    new_summary, new_rows, new_chunks = run_episode(new_env, Policy(), ep, selector=CappedDVACSelector(50))
    assert old_env.actions == new_env.actions
    assert [row["replanned"] for row in old_rows] == [row["replanned"] for row in new_rows]
    assert [chunk["sampling_seed"] for chunk in old_chunks] == [chunk["sampling_seed"] for chunk in new_chunks]
    assert old_summary["steps"] == new_summary["steps"]
    assert new_summary["cap_applied_calls"] == 0


@pytest.mark.parametrize("field", ["pair_id", "initial_state_sha256", "initial_observation_sha256", "sampling_seed", "actions"])
def test_reference_mismatch_stops_before_first_motion(field):
    ep, env = episode(), Environment()
    expected = reference(ep)
    if field == "actions":
        expected[field][0][0] += .01
    elif field == "sampling_seed":
        expected[field] += 1
    else:
        expected[field] = "wrong"
    with pytest.raises(ValueError, match="[Rr]eference"):
        run_episode(env, Policy(), ep, selector=CappedDVACSelector(5), reference=expected)
    assert env.actions == []


def test_matching_reference_is_logged():
    ep = episode()
    summary, _, _ = run_episode(Environment(), Policy(), ep, selector=CappedDVACSelector(5), reference=reference(ep))
    assert summary["reference_initial_state_match"] is True
    assert summary["reference_first_prediction_equal"] is True
    assert summary["reference_max_absolute_action_error"] == 0


def test_mismatched_cap_selector_fails():
    with pytest.raises(ValueError, match="declared execution cap"):
        run_episode(Environment(), Policy(), episode(5), selector=CappedDVACSelector(10))


def test_cap_one_equals_fixed_one_even_when_internal_variances_change():
    class ChangingPolicy(Policy):
        def reset(self):
            self.calls = 0

        def predict(self, obs, seed):
            actions = super().predict(obs, seed)
            self.calls += 1
            self.last_diagnostics["variance"] = [float((self.calls * 7 + index) % 13) for index in range(50)]
            return actions

    capped = episode(1)
    fixed = CappedEpisode(capped.suite, capped.task, capped.state, capped.policy_seed, "fixed_1", capped.max_steps)
    cap_env, fixed_env = Environment(), Environment()
    cap_summary, cap_rows, cap_chunks = run_episode(cap_env, ChangingPolicy(), capped, selector=CappedDVACSelector(1))
    fixed_summary, fixed_rows, fixed_chunks = run_episode(fixed_env, ChangingPolicy(), fixed)
    assert cap_env.actions == fixed_env.actions
    assert [row["replanned"] for row in cap_rows] == [row["replanned"] for row in fixed_rows]
    assert [chunk["step"] for chunk in cap_chunks] == [chunk["step"] for chunk in fixed_chunks]
    assert [chunk["sampling_seed"] for chunk in cap_chunks] == [chunk["sampling_seed"] for chunk in fixed_chunks]
    assert cap_summary["success"] == fixed_summary["success"]
    assert cap_summary["model_calls"] == fixed_summary["model_calls"] == 9
