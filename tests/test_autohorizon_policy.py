"""CPU mock tests for attention observation; no checkpoint or simulator needed."""

from types import SimpleNamespace

import pytest

from adaptive_replan.policy import InstrumentedSmolVLAPolicy
from autohorizon_replan.policy import AutoHorizonSmolVLAPolicy, SelectorAdapter, capture_attention
from vla_replan.backend import SmolVLAPolicy

torch = pytest.importorskip("torch")


class Expert:
    num_attention_heads = 4
    num_key_value_heads = 2

    def __init__(self):
        self.original_outputs = []
        self.inputs = []

    def eager_attention_forward(self, attention_mask, batch_size, head_dim,
                                query_states, key_states, value_states):
        repeat = self.num_attention_heads // self.num_key_value_heads
        keys = key_states.repeat_interleave(repeat, dim=2).transpose(1, 2)
        values = value_states.repeat_interleave(repeat, dim=2).transpose(1, 2)
        queries = query_states.transpose(1, 2)
        scores = queries @ keys.transpose(-1, -2) / head_dim ** 0.5
        probabilities = scores.masked_fill(~attention_mask[:, None], torch.finfo(scores.dtype).min).softmax(-1)
        output = (probabilities @ values).transpose(1, 2)
        self.original_outputs.append(output)
        self.inputs.append((query_states.clone(), key_states.clone(), attention_mask.clone()))
        return output

    def forward_attn_layer(self, layer_idx, query_states, key_states, value_states, attention_mask):
        result = self.eager_attention_forward(
            attention_mask, 1, query_states.shape[-1], query_states, key_states, value_states)
        # Proves the capture wrapper returns exactly the original tensor object.
        assert result is self.original_outputs[-1]
        return result

    def forward_cross_attn_layer(self, query_states, key_states, value_states, attention_mask):
        result = self.eager_attention_forward(
            attention_mask, 1, query_states.shape[-1], query_states, key_states, value_states)
        assert result is self.original_outputs[-1]
        return result


class Model:
    def __init__(self, horizon=3, self_layers=(0, 2)):
        self.vlm_with_expert = Expert()
        self.horizon = horizon
        self.self_layers = self_layers

    def tensors(self, layer_idx, timestep, *, cross=False):
        length = 2 if cross else self.horizon + 2
        queries = torch.arange(self.horizon * 8, dtype=torch.float32).reshape(1, self.horizon, 4, 2)
        queries = queries / 31 + timestep.item() + layer_idx * 0.07
        keys = torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 2, 2) / 19
        values = keys * 0.23
        mask = torch.ones(1, self.horizon, length, dtype=torch.bool)
        mask[:, 0, 1] = False
        if not cross:
            mask[:, 1, 3] = False
        return queries, keys, values, mask

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        result = x_t * 0.2 + timestep[:, None, None]
        for layer_idx in self.self_layers:
            q, k, v, mask = self.tensors(layer_idx, timestep)
            layer = self.vlm_with_expert.forward_attn_layer(layer_idx, q, k, v, mask)
            result = result + 0.01 * layer.sum(dim=(-1, -2))[..., None]
            # Cross attention has fewer than H keys, so erroneous capture
            # would either fail its dimension check or contaminate the mean.
            q, k, v, mask = self.tensors(layer_idx, timestep, cross=True)
            self.vlm_with_expert.forward_cross_attn_layer(q, k, v, mask)
        self.last_velocity = result
        return result


def integrate(model):
    state = torch.zeros(1, model.horizon, 7)
    for i in range(10):
        velocity = model.denoise_step(None, None, state, torch.tensor([1 - i / 10]))
        assert velocity is model.last_velocity
        state = state - 0.1 * velocity
    return state


def expected_attention(model, layer_idx, step=3):
    q, k, _, mask = model.tensors(layer_idx, torch.tensor([1 - (step - 1) / 10]))
    k = k.repeat_interleave(2, dim=2)
    logits = torch.einsum("bihd,bjhd->bhij", q, k) / 2 ** 0.5
    probabilities = logits.masked_fill(~mask[:, None], torch.finfo(logits.dtype).min).softmax(-1)
    return probabilities[0, :, :, -model.horizon:].mean(0)


def assert_hooks_removed(model):
    assert "denoise_step" not in model.__dict__
    assert "forward_attn_layer" not in model.vlm_with_expert.__dict__
    assert "eager_attention_forward" not in model.vlm_with_expert.__dict__


def test_capture_preserves_outputs_and_collects_only_third_step_self_attention():
    model = Model()
    baseline = integrate(model)
    with capture_attention(model, selected_fm_step=3, horizon=3) as capture:
        observed = integrate(model)
    assert torch.equal(observed, baseline)
    assert capture["calls"] == 10
    assert capture["layer_indices"] == [0, 2]
    assert capture["head_count"] == 4
    assert capture["fm_time"].item() == pytest.approx(0.8)
    assert capture["active_layer"] is None
    assert len(capture["layers"]) == 2
    for actual, layer_idx in zip(capture["layers"], (0, 2), strict=True):
        torch.testing.assert_close(actual, expected_attention(model, layer_idx), rtol=1e-6, atol=1e-7)
        assert actual.shape == (3, 3)
        assert not actual.requires_grad
        # Prefix tokens retain attention mass until the selector normalizes.
        assert bool((actual.sum(-1) < 1).all())
        assert not torch.allclose(actual, expected_attention(model, layer_idx, step=1))
    assert_hooks_removed(model)


def test_capture_restores_existing_instance_overrides_after_exception():
    model = Model()
    expert = model.vlm_with_expert
    saved_denoise, saved_layer, saved_attention = (
        model.denoise_step, expert.forward_attn_layer, expert.eager_attention_forward)
    model.denoise_step = saved_denoise
    expert.forward_attn_layer = saved_layer
    expert.eager_attention_forward = saved_attention
    with pytest.raises(RuntimeError, match="intentional"):
        with capture_attention(model, horizon=3):
            integrate(model)
            raise RuntimeError("intentional user-code failure")
    assert model.denoise_step is saved_denoise
    assert expert.forward_attn_layer is saved_layer
    assert expert.eager_attention_forward is saved_attention


def test_capture_clears_active_layer_and_restores_hooks_if_original_attention_raises():
    model = Model()

    def fail_attention(*args, **kwargs):
        raise RuntimeError("intentional attention failure")

    model.vlm_with_expert.eager_attention_forward = fail_attention
    with pytest.raises(RuntimeError, match="intentional attention"):
        with capture_attention(model, horizon=3) as capture:
            integrate(model)
    assert capture["active_layer"] is None
    assert model.vlm_with_expert.eager_attention_forward is fail_attention
    assert "forward_attn_layer" not in model.vlm_with_expert.__dict__
    assert "denoise_step" not in model.__dict__


def test_dimension_error_restores_hooks_after_original_result_is_computed():
    model = Model()
    with pytest.raises(ValueError, match="dimensions"):
        with capture_attention(model, selected_fm_step=1, horizon=4):
            integrate(model)
    assert len(model.vlm_with_expert.original_outputs) == 1
    assert_hooks_removed(model)


def make_policy(self_layers=(0, 2)):
    policy = AutoHorizonSmolVLAPolicy.__new__(AutoHorizonSmolVLAPolicy)
    policy.policy = SimpleNamespace(model=Model(self_layers=self_layers))
    policy.horizon = 3
    policy.selected_fm_step = 3
    policy.num_steps = 10
    policy.tail_length = 5
    policy.action_dim = 7
    policy.torch = torch
    policy.last_diagnostics = None
    return policy


def mock_base_predict(self, observation, seed):
    return integrate(self.policy.model)[0].tolist()


def test_policy_averages_heads_then_layers_and_preserves_dvac_capture(monkeypatch):
    monkeypatch.setattr(SmolVLAPolicy, "predict", mock_base_predict)
    policy = make_policy()
    expected_actions = integrate(Model())[0].tolist()
    actions = policy.predict(None, 11)
    assert actions == expected_actions
    diagnostics = policy.last_diagnostics
    expected_layers = torch.stack([expected_attention(policy.policy.model, i) for i in (0, 2)])
    torch.testing.assert_close(torch.tensor(diagnostics["attention_layers"]), expected_layers)
    torch.testing.assert_close(torch.tensor(diagnostics["attention"]), expected_layers.mean(0))
    assert diagnostics["velocity_evaluations"] == 10
    assert len(diagnostics["clean_estimates"]) == 5
    assert diagnostics["fm_times"] == pytest.approx([0.5, 0.4, 0.3, 0.2, 0.1])
    assert diagnostics["attention_capture"]["selected_fm_step"] == 3
    assert diagnostics["attention_capture"]["fm_time"] == pytest.approx(0.8)
    assert diagnostics["attention_capture"]["layer_indices"] == [0, 2]
    assert diagnostics["attention_capture"]["head_count"] == 4
    assert diagnostics["attention_capture"]["statistics_dtype"] == "float32"
    assert_hooks_removed(policy.policy.model)


def test_policy_fails_if_attention_was_not_observed(monkeypatch):
    monkeypatch.setattr(SmolVLAPolicy, "predict", mock_base_predict)
    policy = make_policy(self_layers=())
    with pytest.raises(RuntimeError, match="Missing"):
        policy.predict(None, 11)
    assert_hooks_removed(policy.policy.model)


def test_policy_fails_if_a_self_attention_layer_is_observed_twice(monkeypatch):
    monkeypatch.setattr(SmolVLAPolicy, "predict", mock_base_predict)
    policy = make_policy(self_layers=(0, 0))
    with pytest.raises(RuntimeError, match="Duplicate"):
        policy.predict(None, 11)
    assert_hooks_removed(policy.policy.model)


@pytest.mark.parametrize("kwargs", [
    {"selected_fm_step": 0}, {"selected_fm_step": True}, {"selected_fm_step": 3.5},
    {"horizon": 0}, {"horizon": True}, {"horizon": 3.5},
])
def test_invalid_capture_configuration_is_rejected_before_patching(kwargs):
    model = Model()
    with pytest.raises(ValueError):
        with capture_attention(model, **kwargs):
            pass
    assert_hooks_removed(model)


def test_policy_rejects_sampling_step_beyond_solver_length(monkeypatch):
    def initialize_stub(self, experiment, device="cuda", tail_length=5):
        self.num_steps = experiment.num_denoising_steps

    monkeypatch.setattr(InstrumentedSmolVLAPolicy, "__init__", initialize_stub)
    with pytest.raises(ValueError):
        AutoHorizonSmolVLAPolicy(SimpleNamespace(chunk_size=50, num_denoising_steps=10), selected_fm_step=11)


def test_selector_adapter_ignores_variance_and_forwards_current_attention():
    class SpySelector:
        execution_cap = 25

        def __init__(self):
            self.last_decision = None
            self.received = None
            self.resets = 0

        def reset(self):
            self.resets += 1
            self.last_decision = None

        def select(self, values):
            self.received = values
            self.last_decision = {"chosen_k": 7}
            return self.last_decision

    class UnusableVariance:
        def __iter__(self):
            raise AssertionError("variance must not be accessed")

    attention = [[1.0, 0.0], [0.0, 1.0]]
    selector = SpySelector()
    adapter = SelectorAdapter(selector, SimpleNamespace(last_diagnostics={"attention": attention}))
    decision = adapter.select(UnusableVariance())
    assert selector.received is attention
    assert decision is selector.last_decision is adapter.last_decision
    assert adapter.execution_cap == 25
    adapter.reset()
    assert selector.resets == 1
    assert adapter.last_decision is None
