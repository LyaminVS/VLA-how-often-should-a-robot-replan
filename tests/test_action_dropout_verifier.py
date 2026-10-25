"""Independent applied-action replay, immutable commands, and partial coverage."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from action_dropout.config import DropoutEpisode, DropoutExperiment
from action_dropout.core import run_episode
from capped_replan.config import CappedExperiment
from scripts.verify_action_dropout import (
    inventory_report,
    registered_plan,
    reference_mask_sha256,
    reference_uniform,
    verify_applied_actions,
    verify_commanded_episode,
)
from vla_replan.core import Observation


def configuration():
    original = CappedExperiment.load(Path(__file__).resolve().parents[1] / "configs/capped.json")
    base = replace(original.base, tasks=(replace(original.base.tasks[0], max_steps=8),),
                   initial_states=(0,), policy_seeds=(11,), intervals=(5,), record_video=False)
    return DropoutExperiment(base, ("fixed_5",), original.dvac, (0.1, 0.3), 20261025)


class Environment:
    def __init__(self):
        self.applied = []

    def reset(self, episode):
        self.applied = []
        return Observation(None, [0.0, 0.0, 0.0], [0.0, 0.0]), {}

    def step(self, action):
        self.applied.append(list(action))
        return Observation(None, [len(self.applied) * 0.01, 0.0, 0.0], [0.0, 0.0]), len(self.applied) == 8, False


class Policy:
    def reset(self):
        pass

    def predict(self, observation, seed):
        self.last_diagnostics = {
            "clean_estimates": [[[0.0] * 7 for _ in range(50)] for _ in range(5)],
            "variance": [0.0] * 50, "fm_times": [0.5, 0.4, 0.3, 0.2, 0.1],
            "velocity_evaluations": 10, "action_dimensions": 7,
            "statistics_dtype": "float64", "diagnostics_finalize_s": 0.0,
        }
        return [[0.2, -0.3, 0.4, 0.1, -0.2, 0.3, 1.0] for _ in range(50)]


def evidence(probability=0.2):
    config = configuration()
    task = config.base.tasks[0]
    episode = DropoutEpisode(task.suite, task.name, 0, 11, "fixed_5", 8,
                             probability, config.disturbance_seed)
    environment = Environment()
    summary, rows, chunks = run_episode(environment, Policy(), episode)
    summary["episode_wall_s"] = 0.0
    return config, environment, summary, rows, chunks


@pytest.mark.parametrize("probability", [0.0, 0.1, 0.2, 0.3, 1.0])
def test_command_replay_and_independent_applied_replay_both_pass(probability):
    config, environment, summary, rows, chunks = evidence(probability)
    _, command_audit = verify_commanded_episode(summary, rows, chunks, config)
    applied_audit = verify_applied_actions(summary, rows)
    assert command_audit["steps"] == 8
    assert applied_audit["verified_applied_action_values"] == 56
    assert environment.applied == [row["applied_action"] for row in rows]
    assert all(row["action"] == row["commanded_action"] for row in rows)
    assert all(row["action"][3:] == row["applied_action"][3:] for row in rows)
    assert summary["success"] is True
    if probability == 0:
        assert applied_audit["translation_drop_count"] == 0
    elif probability == 1:
        assert applied_audit["translation_drop_count"] == 8
        assert all(row["applied_action"][:3] == [0.0, 0.0, 0.0] for row in rows)


def test_all_probabilities_and_methods_share_the_same_step_uniforms():
    cfg, _, low, low_rows, _ = evidence(0.1)
    _, _, high, high_rows, _ = evidence(0.3)
    assert [row["dropout_u"] for row in low_rows] == [row["dropout_u"] for row in high_rows]
    assert all(not left["translation_dropped"] or right["translation_dropped"]
               for left, right in zip(low_rows, high_rows, strict=True))
    task = cfg.base.tasks[0]
    changed_method = DropoutEpisode(task.suite, task.name, 0, 11, "fixed_1", 8, 0.3, cfg.disturbance_seed)
    summary, rows, _ = run_episode(Environment(), Policy(), changed_method)
    assert [row["dropout_u"] for row in rows] == [row["dropout_u"] for row in high_rows]
    assert summary["dropout_mask_sha256"] == high["dropout_mask_sha256"]
    assert reference_mask_sha256(low["pair_id"], cfg.disturbance_seed, 0.1, 8) == low["dropout_mask_sha256"]


@pytest.mark.parametrize("mutation,message", [
    (lambda s, r: r[0].update(dropout_u=0.123456789), "uniform"),
    (lambda s, r: r[0].update(translation_dropped=not r[0]["translation_dropped"]), "flag"),
    (lambda s, r: r[0]["applied_action"].__setitem__(3, 0.9), "Applied action"),
    (lambda s, r: r[0]["applied_action"].__setitem__(6, -1.0), "Applied action"),
    (lambda s, r: r[0]["applied_action"].__setitem__(0, 0.9), "Applied action"),
    (lambda s, r: r[0]["commanded_action"].__setitem__(0, 0.9), "alias"),
    (lambda s, r: s.update(translation_drop_count=999), "dropout count"),
    (lambda s, r: s.update(translation_changed_count=999), "changed-translation"),
    (lambda s, r: s.update(dropout_mask_sha256="0" * 64), "mask checksum"),
    (lambda s, r: s.update(dropout_mask_length=3), "full configured"),
    (lambda s, r: s.update(dropout_model="different"), "Unknown"),
    (lambda s, r: r[0].update(translation_dropped=1), "flag"),
])
def test_mutated_applied_evidence_fails_independently(mutation, message):
    _, _, summary, rows, _ = evidence()
    summary, rows = copy.deepcopy(summary), copy.deepcopy(rows)
    mutation(summary, rows)
    with pytest.raises(ValueError, match=message):
        verify_applied_actions(summary, rows)


def test_success_is_checked_against_post_applied_action_environment_log():
    config, _, summary, rows, chunks = evidence(1.0)
    summary["success"] = False
    with pytest.raises(ValueError, match="success disagrees"):
        verify_commanded_episode(summary, rows, chunks, config)


def test_partial_inventory_is_explicit_and_never_counts_missing_as_failure():
    config = configuration()
    all_ids = {episode.id for episode in config.episodes()}
    one = {next(iter(all_ids))}
    partial = inventory_report(config, one)
    assert partial["complete"] is False
    assert partial["completed_new_episodes"] == 1
    assert partial["planned_new_episodes"] == 2
    assert set(partial["missing_episode_ids"]) == all_ids - one
    assert sum(cell["completed"] for cell in partial["coverage"]) == 1
    assert inventory_report(config, all_ids)["complete"] is True
    with pytest.raises(ValueError, match="unexpected"):
        inventory_report(config, all_ids | {"unplanned_episode"})


def test_schedule_identity_changes_with_seed_pair_or_step_but_not_global_rng():
    first = reference_uniform("suite/task/state0/seed11", 7, 3)
    assert 0 <= first < 1
    assert reference_uniform("suite/task/state0/seed11", 7, 3) == first
    assert len({first, reference_uniform("suite/task/state1/seed11", 7, 3),
                reference_uniform("suite/task/state0/seed11", 8, 3),
                reference_uniform("suite/task/state0/seed11", 7, 4)}) == 4


def test_registered_two_probability_plan_preserves_all_starts_and_methods():
    from action_dropout.config import ALL_METHODS

    config = DropoutExperiment.load(Path(__file__).resolve().parents[1] / "configs/action-dropout.json")
    config = replace(config, probabilities=(.1, .3))
    expected = registered_plan(config)
    assert len(expected) == 540
    assert len({episode.pair_id for episode in expected.values()}) == 30
    assert {episode.method for episode in expected.values()} == set(ALL_METHODS)
    assert all(episode.method != "fixed_1" for episode in list(expected.values())[:480])
    assert all(episode.method == "fixed_1" for episode in list(expected.values())[480:])
    assert len(registered_plan(replace(config, probabilities=(.1, .2, .3)))) == 810
    with pytest.raises(ValueError, match="nine methods"):
        registered_plan(replace(config, methods=tuple(m for m in config.methods if m != "fixed_1")))
    with pytest.raises(ValueError, match="positive probabilities"):
        registered_plan(replace(config, probabilities=(0.0, .1, .3)))
