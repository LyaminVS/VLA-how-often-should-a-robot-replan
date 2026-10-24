"""Protocol validation and pairing tests; no policy or simulator is loaded."""

from dataclasses import replace
from pathlib import Path

import pytest

from moving_target.config import CONDITIONS, MotionSpec, MovingExperiment
from vla_replan.config import Experiment


@pytest.fixture
def moving_config():
    base = Experiment.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    base = replace(base, intervals=(5, 10, 50), initial_states=tuple(range(10, 15)), policy_seeds=(11, 22, 33))
    return MovingExperiment(base, MotionSpec())


def test_plan_has_135_unique_episodes_and_nine_treatments_per_block(moving_config):
    episodes = moving_config.episodes()
    assert len(episodes) == 135
    assert len({episode.id for episode in episodes}) == 135
    blocks = {}
    for episode in episodes:
        blocks.setdefault(episode.pair_id, set()).add((episode.condition, episode.base_episode.k))
    assert len(blocks) == 15
    assert all(values == {(condition, k) for condition in CONDITIONS for k in (5, 10, 50)} for values in blocks.values())
    assert moving_config.budget()["episodes"] == 135


def test_seed_and_treatment_do_not_change_physical_start(moving_config):
    for state in moving_config.base_config.initial_states:
        episodes = [episode for episode in moving_config.episodes() if episode.base_episode.state == state]
        assert len({episode.env_seed for episode in episodes}) == 1


def test_randomization_is_repeatable_and_blocked(moving_config):
    ids = [episode.id for episode in moving_config.episodes()]
    assert ids == [episode.id for episode in moving_config.episodes()]
    altered = replace(moving_config, base_config=replace(moving_config.base_config, order_seed=17))
    assert ids != [episode.id for episode in altered.episodes()]
    assert set(ids) == {episode.id for episode in altered.episodes()}
    episodes = moving_config.episodes()
    for start in range(0, len(episodes), 9):
        assert len({episode.pair_id for episode in episodes[start:start + 9]}) == 1


def test_json_roundtrip_preserves_protocol_identity(moving_config):
    import json

    loaded = MovingExperiment.from_dict(json.loads(json.dumps(moving_config.asdict())))
    assert loaded == moving_config
    assert loaded.fingerprint == moving_config.fingerprint
    assert loaded.episodes() == moving_config.episodes()


@pytest.mark.parametrize("name,value", [
    ("amplitude_m", 0), ("amplitude_m", float("nan")), ("kp_n_per_m", float("inf")),
    ("kp_n_per_m", True), ("max_force_n", -1), ("ramp_s", 0),
    ("start_delay_s", -0.1), ("kd_ns_per_m", -1), ("force_axis", "y"),
    ("stop_on_contact", False), ("stop_on_contact", 1),
    ("force_application_offset_z_m", -0.051), ("force_application_offset_z_m", 0.001),
    ("force_application_offset_z_m", float("nan")), ("force_application_offset_z_m", True),
])
def test_invalid_motion_parameters_fail_before_loading_simulator(name, value):
    with pytest.raises(ValueError):
        replace(MotionSpec(), **{name: value}).validate()


@pytest.mark.parametrize("periods", [
    {"static": 0, "slow": 8, "fast": 4},
    {"static": None, "slow": 4, "fast": 8},
    {"static": None, "slow": 8, "fast": float("nan")},
    {"static": None, "slow": 8},
])
def test_invalid_condition_definitions_are_rejected(periods):
    with pytest.raises(ValueError):
        MotionSpec(periods_s=periods).validate()


def test_moving_protocol_keeps_original_checkpoint_validation(moving_config):
    with pytest.raises(ValueError, match="Pin model"):
        replace(moving_config, base_config=replace(moving_config.base_config, model_revision="main")).validate()


def test_unknown_config_fields_are_rejected(moving_config):
    data = moving_config.asdict()
    data["unregistered_override"] = True
    with pytest.raises(ValueError, match="exactly"):
        MovingExperiment.from_dict(data)

