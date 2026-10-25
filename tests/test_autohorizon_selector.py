"""Selection parity and edge cases; policy inference is tested separately."""

import json
import math

import pytest

from autohorizon_replan.selector import AutoHorizonSelector

torch = pytest.importorskip("torch")


def _reference(attention, tau=0.3, q=0.9):
    """Independent direct tensor translation of the pinned upstream functions.

    Keep the convolution used by upstream run_len=1 rather than calling any
    production helpers. Returns decisions and intermediates for parity checks.
    """
    matrix = torch.as_tensor(attention, dtype=torch.float32, device="cpu")
    length = matrix.shape[0]
    matrix = matrix / (matrix.sum(-1, keepdim=True) + 1e-12)

    def direction(values):
        probabilities = values / (values.sum(-1, keepdim=True) + 1e-12)
        mu = (probabilities * torch.arange(length, dtype=torch.float32)).sum(-1)
        mu = torch.maximum(mu, torch.cummax(mu, dim=0).values)
        clipped = probabilities.clamp_min(1e-12)
        entropy = -(clipped * clipped.log()).sum(-1) / math.log(length)
        reliable = entropy <= torch.quantile(entropy.float(), q=min(q, 0.999))
        delta = mu - torch.cat((mu.new_tensor([0.0]), mu[:-1]))
        hits = (delta < tau) & reliable
        windows = torch.nn.functional.conv1d(hits.float()[None, None, :], torch.ones(1, 1, 1)).squeeze()
        stop = int(torch.nonzero(windows >= 1, as_tuple=False)[0].item()) if (windows >= 1).any() else length - 1
        return mu, stop, delta, reliable

    forward, stop_f, delta_f, reliable_f = direction(matrix)
    reverse, stop_r, _, _ = direction(matrix.flip((0, 1)))
    backward = torch.flip(length - 1 - reverse, (0,))
    nf = int(torch.clamp(torch.floor(forward[stop_f]) + 1, 1, length).item())
    nb = int(torch.clamp(torch.floor(backward[length - 1 - stop_r]) + 1, 1, length).item())
    gaps = backward - forward
    joins = torch.nonzero(gaps <= 1.0, as_tuple=False)
    join = int(joins[0].item()) if len(joins) else None
    return {
        "chosen": length if nf + nb >= length and join is not None else nf,
        "N_forward": nf, "N_backward": nb, "join_row": join,
        "stop_forward": stop_f, "stop_backward": length - 1 - stop_r,
        "mu_forward": forward.tolist(), "mu_backward": backward.tolist(),
        "dmu_forward": delta_f.tolist(), "reliable_forward": reliable_f.tolist(),
    }


@pytest.mark.parametrize("horizon", [2, 10, 50])
@pytest.mark.parametrize("seed", [0, 7, 23])
def test_matches_pinned_official_rule_on_dense_and_sparse_attention(horizon, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(horizon, horizon, generator=generator)
    for attention in (logits.softmax(-1), (10 * logits).softmax(-1), torch.eye(horizon)):
        expected = _reference(attention)
        decision = AutoHorizonSelector(horizon).select(attention)
        assert decision["raw_chosen_k"] == expected.pop("chosen")
        for field, value in expected.items():
            assert decision["diagnostics"][field] == value
        json.dumps(decision, allow_nan=False)


def test_backward_uses_mapped_original_index_not_reversed_prefix_length():
    attention = torch.zeros(10, 10)
    attention[:, 3] = 1
    decision = AutoHorizonSelector(10).select(attention)
    diagnostics = decision["diagnostics"]
    assert diagnostics["N_forward"] == 4
    assert diagnostics["N_backward"] == 4
    assert diagnostics["mu_backward_reversed"] == [6.0] * 10
    # A paper-style reversed-prefix length would be 7 and incorrectly make
    # this fixture select all 10 actions, instead of the official code's 4.
    assert decision["chosen_k"] == 4
    assert diagnostics["full_coverage"] is False


def test_coverage_and_join_can_select_the_full_chunk():
    attention = torch.zeros(10, 10)
    attention[:, 4] = 1
    decision = AutoHorizonSelector(10).select(attention)
    diagnostics = decision["diagnostics"]
    assert diagnostics["N_forward"] == diagnostics["N_backward"] == 5
    assert diagnostics["join_row"] == 0
    assert diagnostics["full_coverage"] is True
    assert decision["chosen_k"] == 10


def test_no_plateau_uses_last_pointer_without_forcing_full_horizon():
    attention = torch.zeros(10, 10)
    attention[:, 3] = 1
    decision = AutoHorizonSelector(10, hold_thr=0.0).select(attention)
    assert decision["diagnostics"]["found_hold_forward"] is False
    assert decision["diagnostics"]["stop_forward"] == 9
    assert decision["chosen_k"] == 4


def test_entropy_filter_can_skip_an_early_hold():
    # Uniform row1 would be a hold after row0, but has highest entropy.
    attention = torch.eye(10)
    attention[0] = 0
    attention[0, 6] = 1
    attention[1] = 0.1
    decision = AutoHorizonSelector(10).select(attention)
    diagnostics = decision["diagnostics"]
    assert diagnostics["dmu_forward"][1] == 0
    assert diagnostics["reliable_forward"][1] is False
    assert diagnostics["holds_forward"][1] is False
    assert diagnostics["stop_forward"] == 2


def test_fractional_pointer_is_floored_then_converted_to_action_count():
    attention = torch.zeros(10, 10)
    attention[:, 2] = 0.25
    attention[:, 3] = 0.75
    decision = AutoHorizonSelector(10).select(attention)
    assert decision["diagnostics"]["mu_forward"] == [2.75] * 10
    assert decision["raw_chosen_k"] == 3


def test_all_caps_share_original_decision_and_have_no_history_bootstrap():
    attention = torch.eye(50)
    decisions = [AutoHorizonSelector(execution_cap=cap).select(attention) for cap in (5, 10, 25, 50)]
    for decision, cap in zip(decisions, (5, 10, 25, 50), strict=True):
        assert decision["raw_chosen_k"] == 50
        assert decision["chosen_k"] == cap
        assert decision["execution_cap"] == cap
        assert decision["cap_applied"] == (cap < 50)
        assert decision["method_id"] == f"autohorizon_cap_{cap}"
        assert decision["bootstrap"] is False
        assert decision["window_calls"] == 0
        for field in ("threshold", "first_crossing", "window_mean", "window_std"):
            assert decision[field] is None
        assert decision["diagnostics"] == decisions[0]["diagnostics"]


def test_short_raw_choice_is_not_increased_by_cap():
    attention = torch.zeros(50, 50)
    attention[:, 2] = 1
    decision = AutoHorizonSelector(execution_cap=25).select(attention)
    assert decision["chosen_k"] == decision["raw_chosen_k"] == 3
    assert decision["cap_applied"] is False


def test_stateless_reset_input_unchanged_and_independent_last_decision():
    selector = AutoHorizonSelector(10)
    attention = torch.eye(10, requires_grad=True)
    original = attention.detach().clone()
    expected = selector.select(attention)
    assert torch.equal(attention, original)
    expected["diagnostics"]["mu_forward"][0] = -999
    assert selector.last_decision["diagnostics"]["mu_forward"][0] != -999
    expected = selector.select(attention)
    selector.select(torch.ones(10, 10))
    assert selector.select(attention) == expected
    selector.reset()
    assert selector.last_decision is None
    assert selector.select(attention) == expected


def test_single_action_extension_and_json_serialization():
    decision = AutoHorizonSelector(1).select([[2.0]])
    assert decision["chosen_k"] == decision["raw_chosen_k"] == 1
    assert decision["diagnostics"]["single_action_extension"] is True
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize("attention", [
    [[1.0, 0.0]], [1.0, 1.0], [[[1.0, 0.0], [0.0, 1.0]]],
    [[1.0, 0.0], [0.0]], [[0.0, 0.0], [1.0, 0.0]],
    [[-0.01, 1.0], [1.0, 0.0]], [[float("nan"), 1.0], [1.0, 0.0]],
    [[float("inf"), 1.0], [1.0, 0.0]], [[1e100, 1.0], [1.0, 0.0]],
    [[-1e-100, 1.0], [1.0, 0.0]],
    [[True, False], [False, True]], [[1.0 + 1j, 0], [0, 1]],
    [["a", "b"], ["c", "d"]],
])
def test_invalid_attention_fails_before_selection(attention):
    with pytest.raises(ValueError):
        AutoHorizonSelector(2).select(attention)


@pytest.mark.parametrize("kwargs", [
    {"horizon": 0}, {"horizon": True}, {"horizon": 5.0},
    {"execution_cap": 0}, {"execution_cap": 51}, {"execution_cap": True},
    {"hold_thr": -0.1}, {"hold_thr": float("nan")}, {"hold_thr": True},
    {"max_entropy_q": 0}, {"max_entropy_q": 1.1}, {"max_entropy_q": False},
    {"selected_fm_step": 0}, {"selected_fm_step": 3.0},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        AutoHorizonSelector(**kwargs)
