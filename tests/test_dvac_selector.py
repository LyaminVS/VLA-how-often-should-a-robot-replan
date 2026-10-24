"""DVAC decision invariants, using hand-computable independent examples."""

import math

import pytest

from adaptive_replan.selector import DVACSelector


def test_bootstrap_and_reset_are_episode_local():
    selector = DVACSelector(horizon=3)
    first = selector.select([1, 2, 3])
    assert first == {
        "chosen_k": 1,
        "threshold": None,
        "first_crossing": None,
        "window_calls": 0,
        "window_mean": None,
        "window_std": None,
        "bootstrap": True,
    }
    assert selector.select([0, 0, 0])["bootstrap"] is False
    selector.reset()
    assert selector.select([9, 8, 7]) == first


def test_threshold_uses_previous_calls_not_current_call():
    selector = DVACSelector(horizon=4, alpha=0)
    selector.select([2, 2, 2, 2])
    decision = selector.select([0, 0, 3, 1000])
    assert decision["threshold"] == 2
    assert decision["first_crossing"] == 2
    assert decision["chosen_k"] == 2  # Do not execute crossing action index 2.
    assert decision["window_calls"] == 1


@pytest.mark.parametrize(
    ("variance", "crossing", "chosen"),
    [
        ([2, 2, 2, 2], None, 4),  # Equality is allowed by the strict > rule.
        ([3, 0, 0, 0], 0, 1),  # Minimum prefix overrides an immediate crossing.
        ([0, 3, 0, 0], 1, 1),
        ([0, 0, 0, 3], 3, 3),
    ],
)
def test_crossing_index_is_prefix_length(variance, crossing, chosen):
    selector = DVACSelector(horizon=4, alpha=0)
    selector.select([2, 2, 2, 2])
    decision = selector.select(variance)
    assert decision["first_crossing"] == crossing
    assert decision["chosen_k"] == chosen


def test_moments_use_population_std_and_preserve_duplicate_values():
    selector = DVACSelector(horizon=3, alpha=2)
    selector.select([0, 0, 3])
    decision = selector.select([0, 0, 5])
    assert decision["window_mean"] == 1
    assert decision["window_std"] == pytest.approx(math.sqrt(2))
    assert decision["threshold"] == pytest.approx(1 + 2 * math.sqrt(2))
    assert decision["chosen_k"] == 2


def test_history_is_bounded_by_whole_calls_and_does_not_alias_input():
    selector = DVACSelector(horizon=2, window_size=2, alpha=0)
    first = [1, 1]
    selector.select(first)
    first[0] = 1000
    assert selector.select([3, 3])["threshold"] == 1
    assert selector.select([5, 5])["threshold"] == 2
    fourth = selector.select([0, 0])
    assert fourth["window_calls"] == 2
    assert fourth["threshold"] == 4


@pytest.mark.parametrize("invalid", [[0], [0, -1], [0, float("nan")], [0, float("inf")]])
def test_bad_variance_does_not_contaminate_history(invalid):
    selector = DVACSelector(horizon=2, alpha=0)
    selector.select([2, 2])
    with pytest.raises(ValueError):
        selector.select(invalid)
    decision = selector.select([0, 0])
    assert decision["window_calls"] == 1
    assert decision["threshold"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"horizon": 0}, {"horizon": True}, {"tail_length": 0}, {"window_size": 0},
     {"min_steps": 0}, {"horizon": 2, "min_steps": 3}, {"alpha": -1},
     {"alpha": float("nan")}, {"alpha": float("inf")}],
)
def test_invalid_selector_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DVACSelector(**kwargs)
