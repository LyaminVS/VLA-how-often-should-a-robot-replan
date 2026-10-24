from dataclasses import replace
from pathlib import Path

import pytest

from capped_replan.config import CappedExperiment
from adaptive_replan.config import AdaptiveExperiment


def configuration():
    old = AdaptiveExperiment.load(Path(__file__).resolve().parents[1] / "configs/dvac.json")
    base = replace(old.base, intervals=(1, 5, 10, 25, 50), initial_states=tuple(range(10)), policy_seeds=(11,))
    return CappedExperiment(base, tuple(f"dvac_cap_{k}" for k in (5, 10, 25, 50)), old.dvac)


def test_adaptive_only_configuration_retains_generated_horizon_and_reference_intervals():
    cfg = configuration()
    cfg.validate()
    assert cfg.base.chunk_size == cfg.dvac.max_steps == 50
    assert cfg.base.intervals == (1, 5, 10, 25, 50)
    assert cfg.adaptive_caps == (5, 10, 25, 50)
    episodes = cfg.episodes()
    assert len(episodes) == len({e.id for e in episodes}) == 120
    for offset in range(0, 120, 4):
        block = episodes[offset:offset + 4]
        assert len({e.pair_id for e in block}) == 1
        assert len({e.env_seed for e in block}) == 1
        assert {e.execution_cap for e in block} == {5, 10, 25, 50}
        assert all(e.k is None and e.adaptive for e in block)
    assert episodes == cfg.episodes()


def test_full_mixed_schedule_has_every_control_and_cap_in_each_block():
    cfg = configuration()
    cfg = replace(cfg, methods=tuple(f"fixed_{k}" for k in cfg.base.intervals) + cfg.methods)
    cfg.validate()
    assert len(cfg.episodes()) == 270
    assert len({e.pair_id for e in cfg.episodes()[:9]}) == 1
    assert {e.method for e in cfg.episodes()[:9]} == set(cfg.methods)
    for episode in cfg.episodes():
        assert episode.asdict()["method"] == episode.method
        if not episode.adaptive:
            assert episode.execution_cap is None
            assert episode.k in cfg.base.intervals


def test_configuration_roundtrip():
    cfg = configuration()
    restored = CappedExperiment.fromdict(cfg.asdict())
    assert restored.fingerprint == cfg.fingerprint
    assert restored.episodes() == cfg.episodes()


@pytest.mark.parametrize("methods", [(), ("dvac_cap_0",), ("dvac_cap_51",), ("dvac",), ("dvac_cap_5", "dvac_cap_5"), ("fixed_7",)])
def test_invalid_methods_fail(methods):
    with pytest.raises(ValueError):
        replace(configuration(), methods=methods).validate()
