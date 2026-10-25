"""The registered AutoHorizon grid reuses the original physical comparisons."""

import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from autohorizon_replan.config import AutoHorizonExperiment
from autohorizon_replan.integrity import source_manifest as autohorizon_sources
from capped_replan.config import CappedExperiment
from capped_replan.integrity import source_manifest as capped_sources
from capped_replan.reference import MATCH_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return AutoHorizonExperiment.load(ROOT / "configs/autohorizon.json")


def test_registered_grid_has_120_unique_balanced_episodes_and_original_horizon():
    config = configuration()
    assert config.base.chunk_size == 50
    assert config.base.initial_states == tuple(range(10))
    assert config.base.policy_seeds == (11,)
    assert config.base.intervals == (1, 5, 10, 25, 50)
    assert config.autohorizon.selected_fm_step == 3
    assert config.autohorizon.hold_thr == 0.3
    assert config.autohorizon.max_entropy_q == 0.9
    assert config.autohorizon.implementation == "official_c7504f1"
    episodes = config.episodes()
    assert len(episodes) == len({episode.id for episode in episodes}) == 120
    assert len({episode.pair_id for episode in episodes}) == 30
    assert Counter(episode.execution_cap for episode in episodes) == {5: 30, 10: 30, 25: 30, 50: 30}
    assert Counter(episode.task for episode in episodes) == {task.name: 40 for task in config.base.tasks}
    for offset in range(0, 120, 4):
        block = episodes[offset:offset + 4]
        assert len({episode.pair_id for episode in block}) == 1
        assert len({episode.env_seed for episode in block}) == 1
        assert {episode.execution_cap for episode in block} == {5, 10, 25, 50}
        assert all(episode.adaptive and episode.k is None for episode in block)
    assert episodes == config.episodes()


def test_exact_pair_and_environment_seed_parity_with_existing_capped_comparison():
    config = configuration()
    historical = CappedExperiment.load(ROOT / "configs/capped.json")
    new_base, old_base = config.base.asdict(), historical.base.asdict()
    for field in (*MATCH_FIELDS, "tasks", "initial_states", "policy_seeds", "intervals"):
        assert new_base[field] == old_base[field]
    old = {(episode.pair_id, episode.execution_cap): episode for episode in historical.episodes()}
    for episode in config.episodes():
        reference = old[(episode.pair_id, episode.execution_cap)]
        assert episode.env_seed == reference.env_seed
        for field in ("suite", "task", "state", "policy_seed", "max_steps"):
            assert getattr(episode, field) == getattr(reference, field)
        assert episode.id != reference.id
        assert episode.asdict()["execution_cap"] == reference.execution_cap
        assert episode.asdict()["k"] is None


def test_policy_seed_and_method_do_not_change_physical_environment_seed():
    episode = configuration().episodes()[0]
    changed = replace(episode, policy_seed=22, method="autohorizon_cap_50")
    assert changed.env_seed == episode.env_seed
    assert changed.pair_id != episode.pair_id
    assert changed.id != episode.id


def test_roundtrip_preserves_fingerprint_ids_order_and_leaves_input_unchanged(tmp_path):
    config = configuration()
    payload = json.loads(json.dumps(config.asdict()))
    original = copy.deepcopy(payload)
    restored = AutoHorizonExperiment.fromdict(payload)
    assert payload == original
    assert restored == config
    assert restored.fingerprint == config.fingerprint
    assert restored.episodes() == config.episodes()
    destination = tmp_path / "config.json"
    destination.write_text(json.dumps(payload))
    assert AutoHorizonExperiment.load(destination) == config


def test_new_configuration_does_not_change_inherited_source_hashes():
    before = capped_sources()
    config = configuration()
    config.validate()
    config.episodes()
    AutoHorizonExperiment.fromdict(config.asdict())
    assert capped_sources() == before
    complete = autohorizon_sources()
    assert all(complete[path] == sha for path, sha in before.items())
    assert "autohorizon_replan/selector.py" in complete
    assert "configs/autohorizon.json" in complete


@pytest.mark.parametrize("methods", [
    (), ("autohorizon_cap_0",), ("autohorizon_cap_51",), ("autohorizon_cap_05",),
    ("autohorizon",), ("fixed_5",), ("dvac_cap_5",),
    ("autohorizon_cap_5", "autohorizon_cap_5"), ("autohorizon_cap_-5",),
])
def test_invalid_method_lists_are_rejected(methods):
    with pytest.raises(ValueError):
        replace(configuration(), methods=methods).validate()


@pytest.mark.parametrize("changes", [
    {"selected_fm_step": 0}, {"selected_fm_step": 11}, {"selected_fm_step": True},
    {"selected_fm_step": 3.0}, {"hold_thr": 0.4}, {"hold_thr": float("nan")},
    {"max_entropy_q": 0.8}, {"max_entropy_q": float("inf")}, {"implementation": "paper"},
])
def test_invalid_autohorizon_settings_are_rejected(changes):
    config = configuration()
    with pytest.raises(ValueError):
        replace(config, autohorizon=replace(config.autohorizon, **changes)).validate()


@pytest.mark.parametrize("changes", [
    {"precision": "fp16"}, {"chunk_size": 100}, {"initial_states": (0, 0)},
    {"policy_seeds": (True,)}, {"model_revision": "main"},
])
def test_invalid_base_comparison_settings_are_rejected(changes):
    config = configuration()
    with pytest.raises(ValueError):
        replace(config, base=replace(config.base, **changes)).validate()


def test_unknown_configuration_fields_are_rejected():
    payload = configuration().asdict()
    payload["unexpected_setting"] = 1
    with pytest.raises(ValueError, match="Unexpected"):
        AutoHorizonExperiment.fromdict(payload)


def test_actual_schedule_and_settings_are_part_of_the_fingerprint():
    config = configuration()
    assert replace(config, methods=("autohorizon_cap_50",)).fingerprint != config.fingerprint
    assert replace(config, base=replace(config.base, order_seed=1)).fingerprint != config.fingerprint
    assert replace(config, autohorizon=replace(config.autohorizon, selected_fm_step=1)).fingerprint != config.fingerprint
