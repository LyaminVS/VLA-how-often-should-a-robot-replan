import random

import pytest

from adaptive_replan.selector import DVACSelector
from capped_replan.selector import CappedDVACSelector


def test_cap_applies_after_no_crossing_and_after_crossing_beyond_cap():
    capped = CappedDVACSelector(5)
    first = capped.select([0.0] * 50)
    assert first["bootstrap"] and first["chosen_k"] == first["raw_chosen_k"] == 1
    assert first["cap_applied"] is False
    second = capped.select([0.0] * 50)
    assert second["first_crossing"] is None
    assert second["raw_chosen_k"] == 50 and second["chosen_k"] == 5
    assert second["cap_applied"] is True
    third = capped.select([0.0] * 9 + [1.0] * 41)
    assert third["first_crossing"] == third["raw_chosen_k"] == 9
    assert third["chosen_k"] == 5 and third["cap_applied"] is True


def test_full_fifty_value_history_is_identical_to_uncapped_rule():
    capped, original = CappedDVACSelector(5), DVACSelector()
    rng = random.Random(23)
    for _ in range(20):
        values = [rng.random() for _ in range(50)]
        expected = original.select(values)
        actual = capped.select(values)
        assert actual["raw_chosen_k"] == expected["chosen_k"]
        assert actual["chosen_k"] == min(5, expected["chosen_k"])
        for key in ("threshold", "first_crossing", "window_mean", "window_std", "window_calls", "bootstrap"):
            assert actual[key] == expected[key]
        assert list(capped.selector.window) == list(original.window)
        assert all(len(history) == 50 for history in capped.selector.window)


def test_cap_fifty_reproduces_original_decisions_exactly():
    capped, original = CappedDVACSelector(50), DVACSelector()
    for values in ([0.0] * 50, [1.0] * 50, [0.0] * 25 + [20.0] * 25, [0.0] * 50):
        expected, actual = original.select(values), capped.select(values)
        assert {key: actual[key] for key in expected} == expected
        assert actual["raw_chosen_k"] == actual["chosen_k"]
        assert actual["cap_applied"] is False


def test_reset_restores_first_call_initialization():
    selector = CappedDVACSelector(10)
    selector.select([0.0] * 50)
    selector.select([0.0] * 50)
    selector.reset()
    assert selector.last_decision is None
    assert selector.select([3.0] * 50)["bootstrap"] is True


@pytest.mark.parametrize("cap", [0, 51, True, 5.0])
def test_invalid_caps(cap):
    with pytest.raises(ValueError):
        CappedDVACSelector(cap)
