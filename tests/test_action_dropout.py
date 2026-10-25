"""Actuator dropout must not alter scheduling, observation reset, or policy RNG."""

import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from action_dropout.config import ALL_METHODS, DropoutEpisode, DropoutExperiment
from action_dropout.core import (DropoutEnvironment, dropout_mask_sha256, run_episode,
                                 transform_action, uniform_for_step)
from capped_replan.config import CappedEpisode
from capped_replan.core import run_episode as frozen_run_episode
from capped_replan.selector import CappedDVACSelector
from vla_replan.core import Observation


PROJECT = Path(__file__).resolve().parents[1]


def episode(probability=0.2, method="fixed_5", *, state=0, disturbance_seed=20261025, max_steps=13):
    return DropoutEpisode("libero_object", "test-task", state, 11, method, max_steps,
                          probability, disturbance_seed)


class FakeEnvironment:
    def __init__(self, success_step=9):
        self.success_step = success_step

    def observation(self):
        return Observation({"instruction": "TEST ONLY"}, list(self.position), [0.0, 0.0])

    def reset(self, selected):
        self.steps, self.actions, self.position = 0, [], [0.0, 0.0, 0.0]
        # The real backend settles inside reset. These operations must not
        # consume any control-step disturbance counters.
        self.settled_steps = 10
        return self.observation(), {
            "initial_state_sha256": "a" * 64, "initial_observation_sha256": "b" * 64,
        }

    def step(self, action):
        self.actions.append(list(action))
        self.steps += 1
        self.position = [old + value * 0.001 for old, value in zip(self.position, action[:3], strict=True)]
        return self.observation(), self.steps == self.success_step, False


class FakePolicy:
    def reset(self):
        self.calls = 0

    def predict(self, observation, seed):
        self.calls += 1
        self.last_diagnostics = {
            "clean_estimates": [[[0.0] * 7 for _ in range(50)] for _ in range(5)],
            "variance": [0.0] * 50, "fm_times": [.5, .4, .3, .2, .1],
        }
        # Nonzero translation, rotation and gripper; first XYZ values require
        # clipping before any disturbance is applied. Feedback affects z.
        return [[1.5, -2.0, .5 + observation.eef[2], .1, -.2, .3, (-1.0) ** index]
                for index in range(50)]


def independent_uniform(pair_id, seed, step):
    text = json.dumps(["translation_dropout_v1", pair_id, seed, step], separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 9007199254740992


def test_counter_uniform_matches_declared_hash_protocol_and_is_bounded():
    selected = episode()
    for step in range(50):
        actual = uniform_for_step(selected.pair_id, selected.disturbance_seed, step)
        assert actual == independent_uniform(selected.pair_id, selected.disturbance_seed, step)
        assert 0 <= actual < 1
    assert uniform_for_step(selected.pair_id, selected.disturbance_seed, 0) != uniform_for_step(
        selected.pair_id, selected.disturbance_seed + 1, 0)


def test_masks_are_shared_across_methods_and_nested_across_probabilities():
    settings = [episode(p, method) for p in (.1, .2, .3) for method in ALL_METHODS]
    assert len({selected.pair_id for selected in settings}) == 1
    for step in range(100):
        values = [uniform_for_step(selected.pair_id, selected.disturbance_seed, step) for selected in settings]
        assert len(set(values)) == 1
        flags = [values[0] < p for p in (.1, .2, .3)]
        assert flags == sorted(flags)


def test_disturbance_does_not_consume_the_global_random_stream():
    previous = random.getstate()
    try:
        random.seed(874)
        start = random.getstate()
        for step in range(50):
            uniform_for_step(episode().pair_id, 77, step)
        assert random.getstate() == start
    finally:
        random.setstate(previous)


def test_zero_probability_reproduces_frozen_rollout_actions_and_predictions():
    dropped = episode(0.0)
    original = CappedEpisode(dropped.suite, dropped.task, dropped.state, dropped.policy_seed,
                             dropped.method, dropped.max_steps)
    old_env, new_env = FakeEnvironment(), FakeEnvironment()
    old_summary, old_rows, old_chunks = frozen_run_episode(old_env, FakePolicy(), original)
    new_summary, new_rows, new_chunks = run_episode(new_env, FakePolicy(), dropped)
    assert old_env.actions == new_env.actions
    assert [row["action"] for row in old_rows] == [row["action"] for row in new_rows]
    assert [chunk["actions"] for chunk in old_chunks] == [chunk["actions"] for chunk in new_chunks]
    assert [chunk["sampling_seed"] for chunk in old_chunks] == [chunk["sampling_seed"] for chunk in new_chunks]
    assert [chunk["step"] for chunk in old_chunks] == [chunk["step"] for chunk in new_chunks]
    assert old_summary["success"] == new_summary["success"]
    assert new_summary["translation_drop_count"] == new_summary["translation_changed_count"] == 0
    assert new_summary["initial_state_sha256"] == old_summary["initial_state_sha256"]
    assert new_summary["initial_observation_sha256"] == old_summary["initial_observation_sha256"]


def test_probability_one_zeroes_only_applied_xyz_after_clipping():
    environment = FakeEnvironment()
    summary, rows, chunks = run_episode(environment, FakePolicy(), episode(1.0))
    for row, applied in zip(rows, environment.actions, strict=True):
        assert row["action"] == row["commanded_action"]
        assert row["action"][:2] == [1.0, -1.0]
        assert row["applied_action"] == applied
        assert applied[:3] == [0.0, 0.0, 0.0]
        assert applied[3:] == row["commanded_action"][3:]
        assert row["translation_dropped"] is True
    assert chunks[0]["actions"][0][:2] == [1.5, -2.0]
    assert summary["translation_drop_count"] == summary["translation_changed_count"] == len(rows)
    assert summary["dropout_p"] == 1.0 and summary["disturbance_seed"] == 20261025


def test_full_mask_hash_covers_unexecuted_tail_after_early_success():
    selected = episode(1.0, max_steps=13)
    summary, rows, _ = run_episode(FakeEnvironment(success_step=2), FakePolicy(), selected)
    assert len(rows) == 2 and summary["dropout_mask_length"] == 13
    assert summary["dropout_mask_sha256"] == hashlib.sha256(bytes([1] * 13)).hexdigest()
    assert summary["dropout_mask_sha256"] != hashlib.sha256(bytes([1] * 2)).hexdigest()
    assert dropout_mask_sha256(selected.pair_id, selected.disturbance_seed, 0.0, 13) == hashlib.sha256(bytes(13)).hexdigest()


def test_dvac_bootstrap_and_cap_scheduling_still_use_the_frozen_executor():
    selected = episode(.2, "dvac_cap_5")
    selector = CappedDVACSelector(5)
    summary, rows, chunks = run_episode(FakeEnvironment(), FakePolicy(), selected, selector=selector)
    assert [chunk["step"] for chunk in chunks] == [0, 1, 6]
    assert [chunk["decision"]["chosen_k"] for chunk in chunks] == [1, 5, 5]
    assert summary["bootstrap_calls"] == 1
    assert summary["method"] == "dvac_cap_5"
    for row in rows:
        assert row["dropout_u"] == uniform_for_step(selected.pair_id, selected.disturbance_seed, row["step"])


def test_pre_action_video_metadata_matches_the_actual_applied_action():
    recorded = []
    summary, rows, _ = run_episode(
        FakeEnvironment(), FakePolicy(), episode(.3),
        record=lambda frame, row, result: recorded.append((copy.deepcopy(row), result)),
    )
    action_frames = [row for row, result in recorded if "action" in row]
    assert len(action_frames) == len(rows)
    for before, after in zip(action_frames, rows, strict=True):
        for key in ("step", "commanded_action", "applied_action", "translation_dropped", "dropout_u"):
            assert before[key] == after[key]
    assert recorded[-1][0]["step"] == summary["steps"]
    assert recorded[-1][0]["dropout_p"] == .3


def test_reset_settling_does_not_consume_disturbance_steps_and_preview_is_read_only():
    selected = episode(.2)
    environment = FakeEnvironment()
    wrapped = DropoutEnvironment(environment, selected)
    wrapped.reset(selected)
    assert environment.settled_steps == 10 and wrapped.next_step == 0
    action = [.1] * 7
    first, second = wrapped.preview(action), wrapped.preview(action)
    assert first == second and wrapped.next_step == 0 and wrapped.applied_trace == []
    wrapped.step(action)
    assert wrapped.next_step == 1 and wrapped.applied_trace[0] == first
    wrapped.reset(selected)
    assert wrapped.next_step == 0 and wrapped.applied_trace == [] and wrapped.preview(action) == first


def test_failed_environment_step_does_not_advance_counter_or_publish_applied_trace():
    class Failure(FakeEnvironment):
        def step(self, action):
            raise RuntimeError("simulator failed")
    selected = episode(.3)
    wrapped = DropoutEnvironment(Failure(), selected)
    wrapped.reset(selected)
    before = wrapped.preview([.1] * 7)
    with pytest.raises(RuntimeError, match="failed"):
        wrapped.step([.1] * 7)
    assert wrapped.next_step == 0 and wrapped.applied_trace == []
    assert wrapped.preview([.1] * 7) == before


def test_drop_masks_and_actual_command_changes_are_distinct_for_zero_translation():
    class ZeroTranslation(FakePolicy):
        def predict(self, observation, seed):
            return [[0.0, 0.0, 0.0, *action[3:]] for action in super().predict(observation, seed)]
    summary, rows, _ = run_episode(FakeEnvironment(), ZeroTranslation(), episode(1.0))
    assert summary["translation_drop_count"] == len(rows)
    assert summary["translation_changed_count"] == 0


def test_transform_is_pure_and_threshold_is_strict():
    command = [.1, -.2, .3, -.4, .5, -.6, .7]
    before = list(command)
    applied, dropped = transform_action(command, .1, .1)
    assert applied == command and dropped is False
    applied, dropped = transform_action(command, .099, .1)
    assert applied == [0.0, 0.0, 0.0, *command[3:]] and dropped is True
    assert command == before


@pytest.mark.parametrize("uniform", [-.1, 1.0, float("nan"), True])
def test_invalid_uniform_cannot_be_applied(uniform):
    with pytest.raises(ValueError):
        transform_action([0.0] * 7, uniform, .1)


def test_transform_requires_already_clipped_seven_coordinate_commands():
    for action in ([0.0] * 6, [1.1] + [0.0] * 6, [float("nan")] + [0.0] * 6):
        with pytest.raises(ValueError):
            transform_action(action, .05, .1)


def configuration():
    data = json.loads((PROJECT / "configs/capped.json").read_text())
    data.update(methods=list(ALL_METHODS), probabilities=[.1, .2, .3], disturbance_seed=20261025)
    return data


def test_full_plan_contains_810_unique_paired_episodes_without_changing_reset_identity():
    cfg = DropoutExperiment.fromdict(configuration())
    plan = cfg.episodes()
    assert len(plan) == len({selected.id for selected in plan}) == 810
    assert len({selected.pair_id for selected in plan}) == 30
    assert set(Counter((selected.method, selected.dropout_p) for selected in plan).values()) == {30}
    for selected in plan:
        original = CappedEpisode(selected.suite, selected.task, selected.state, selected.policy_seed,
                                 selected.method, selected.max_steps)
        assert selected.pair_id == original.pair_id and selected.env_seed == original.env_seed
        assert selected.asdict()["dropout_p"] == selected.dropout_p
        assert selected.asdict()["disturbance_seed"] == cfg.disturbance_seed
        assert selected.id.endswith({.1: "__p010", .2: "__p020", .3: "__p030"}[selected.dropout_p])


def test_configuration_roundtrip_order_and_fingerprint_are_deterministic():
    cfg = DropoutExperiment.fromdict(configuration())
    copied = DropoutExperiment.fromdict(cfg.asdict())
    assert copied.fingerprint == cfg.fingerprint
    assert [selected.id for selected in copied.episodes()] == [selected.id for selected in cfg.episodes()]
    assert cfg.budget()["episodes"] == 810


@pytest.mark.parametrize("probabilities", [[], [.1, .1], [-.1], [1.1], [True], [.105]])
def test_invalid_or_ambiguous_probability_identifiers_are_rejected(probabilities):
    data = configuration()
    data["probabilities"] = probabilities
    with pytest.raises(ValueError):
        DropoutExperiment.fromdict(data)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_disturbance_seed_must_be_nonnegative_integer(seed):
    with pytest.raises(ValueError):
        episode(disturbance_seed=seed)


def test_reference_reset_and_first_prediction_gate_still_work():
    selected = episode(.2)
    policy = FakePolicy()
    environment = FakeEnvironment()
    observation, info = environment.reset(selected)
    seed = (selected.policy_seed * 1_000_003) % (2**63 - 1)
    policy.reset()
    reference = {"pair_id": selected.pair_id, **info,
                 "sampling_seed": seed, "actions": policy.predict(observation, seed)}
    summary, _, _ = run_episode(environment, FakePolicy(), selected, reference=reference)
    assert summary["reference_first_prediction_equal"] is True
    assert summary["reference_initial_state_match"] is True
    reference["actions"][0][0] += .01
    with pytest.raises(ValueError, match="first complete action chunk"):
        run_episode(environment, FakePolicy(), selected, reference=reference)
    assert environment.actions == []
