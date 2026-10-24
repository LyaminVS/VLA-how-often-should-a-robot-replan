"""The comparison order and identities must preserve paired physical states."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_replan.config import AdaptiveExperiment


CONFIG = Path(__file__).resolve().parents[1] / "configs/dvac.json"


def test_frozen_comparison_contains_twelve_balanced_blocks():
    config = AdaptiveExperiment.load(CONFIG)
    episodes = config.episodes()
    assert len(episodes) == 48
    assert len({e.id for e in episodes}) == 48
    assert Counter(e.method for e in episodes) == {"fixed_5": 12, "fixed_10": 12, "fixed_25": 12, "dvac": 12}
    for offset in range(0, len(episodes), 4):
        block = episodes[offset:offset + 4]
        assert len({e.pair_id for e in block}) == 1
        assert len({e.env_seed for e in block}) == 1
        assert {e.method for e in block} == set(config.methods)
    assert episodes == config.episodes()
    assert {e.state for e in episodes} == {15, 16}
    assert {e.policy_seed for e in episodes} == {11, 22}
    assert config.budget()["matched_blocks"] == 12


def test_policy_seed_does_not_change_physical_seed():
    episode = AdaptiveExperiment.load(CONFIG).episodes()[0]
    changed = replace(episode, policy_seed=999)
    assert changed.env_seed == episode.env_seed
    assert changed.pair_id != episode.pair_id
    assert changed.id != episode.id


def test_configuration_roundtrip_and_fingerprint(tmp_path):
    config = AdaptiveExperiment.load(CONFIG)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.asdict()))
    restored = AdaptiveExperiment.load(path)
    assert restored.fingerprint == config.fingerprint
    assert restored.episodes() == config.episodes()
    changed = replace(config, dvac=replace(config.dvac, alpha=3))
    assert changed.fingerprint != config.fingerprint


@pytest.mark.parametrize("field,value", [
    ("tail_length", 1), ("tail_length", 11), ("window_size", 0),
    ("alpha", float("nan")), ("alpha", True), ("min_steps", 2),
    ("max_steps", 25), ("bootstrap", "current_call_threshold"),
])
def test_invalid_adaptive_settings_are_rejected(field, value):
    config = AdaptiveExperiment.load(CONFIG)
    changed = replace(config, dvac=replace(config.dvac, **{field: value}))
    with pytest.raises(ValueError):
        changed.validate()


@pytest.mark.parametrize("methods", [
    ("fixed_5", "fixed_10", "dvac"),
    ("fixed_5", "fixed_10", "fixed_25", "dvac", "dvac"),
    ("fixed_5", "fixed_10", "fixed_25"),
    ("fixed_5", "fixed_10", "fixed_51", "dvac"),
])
def test_invalid_method_sets_are_rejected(methods):
    config = AdaptiveExperiment.load(CONFIG)
    with pytest.raises(ValueError):
        replace(config, methods=methods).validate()
