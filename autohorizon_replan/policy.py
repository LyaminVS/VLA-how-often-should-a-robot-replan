"""Observe selected-step action self-attention without changing model outputs."""

import inspect
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from adaptive_replan.integrity import file_sha256
from adaptive_replan.policy import InstrumentedSmolVLAPolicy


@contextmanager
def capture_attention(model, *, selected_fm_step=3, horizon=50):
    """Recompute only diagnostic probabilities from unchanged post-RoPE Q/K.

    Cross-attention layers contain no action keys and are excluded. The original
    eager attention output is always returned verbatim; no attention weights,
    hidden states, caches, sampling noise or final actions are replaced.
    """
    import torch

    expert = model.vlm_with_expert
    if (type(selected_fm_step) is not int or selected_fm_step < 1
            or type(horizon) is not int or horizon < 1):
        raise ValueError("FM step and horizon must be positive integers")
    heads, kv_heads = expert.num_attention_heads, expert.num_key_value_heads
    if (type(heads) is not int or type(kv_heads) is not int
            or min(heads, kv_heads) < 1 or heads % kv_heads):
        raise ValueError("Invalid query/key head counts")
    originals = {}
    capture = {"calls": 0, "layer_indices": [], "layers": [], "active_layer": None,
               "fm_time": None, "head_count": expert.num_attention_heads}

    def patch(obj, name, replacement):
        originals[(obj, name)] = (name in obj.__dict__, obj.__dict__.get(name))
        setattr(obj, name, replacement)

    denoise = model.denoise_step
    denoise_signature = inspect.signature(denoise)

    @wraps(denoise)
    def observe_denoise(*args, **kwargs):
        capture["calls"] += 1
        if capture["calls"] == selected_fm_step:
            bound = denoise_signature.bind(*args, **kwargs)
            capture["fm_time"] = bound.arguments["timestep"].detach()
        return denoise(*args, **kwargs)

    self_layer = expert.forward_attn_layer
    layer_signature = inspect.signature(self_layer)

    @wraps(self_layer)
    def observe_layer(*args, **kwargs):
        bound = layer_signature.bind(*args, **kwargs)
        capture["active_layer"] = bound.arguments["layer_idx"]
        try:
            return self_layer(*args, **kwargs)
        finally:
            capture["active_layer"] = None

    attention = expert.eager_attention_forward

    @wraps(attention)
    def observe_attention(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        output = attention(attention_mask, batch_size, head_dim, query_states, key_states, value_states)
        if capture["calls"] != selected_fm_step or capture["active_layer"] is None:
            return output
        if batch_size != 1 or query_states.shape[1] != horizon or key_states.shape[1] < horizon:
            raise ValueError("Unexpected action self-attention dimensions")
        heads, kv_heads = expert.num_attention_heads, expert.num_key_value_heads
        length = key_states.shape[1]
        keys = key_states[:, :, :, None, :].expand(batch_size, length, kv_heads, heads // kv_heads, head_dim)
        keys = keys.reshape(batch_size, length, heads, head_dim).float().transpose(1, 2)
        queries = query_states.float().transpose(1, 2)
        logits = torch.matmul(queries, keys.transpose(2, 3))
        logits *= head_dim ** -0.5
        masked = torch.where(attention_mask[:, None, :, :], logits, torch.finfo(logits.dtype).min)
        probabilities = torch.nn.functional.softmax(masked, dim=-1)
        layer = probabilities[:, :, -horizon:, -horizon:].mean(dim=(0, 1))
        capture["layers"].append(layer.detach())
        capture["layer_indices"].append(int(capture["active_layer"]))
        return output

    patch(model, "denoise_step", observe_denoise)
    patch(expert, "forward_attn_layer", observe_layer)
    patch(expert, "eager_attention_forward", observe_attention)
    try:
        yield capture
    finally:
        for (obj, name), (had_override, previous) in reversed(list(originals.items())):
            if had_override:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)


class AutoHorizonSmolVLAPolicy(InstrumentedSmolVLAPolicy):
    def __init__(self, experiment, device="cuda", tail_length=5, selected_fm_step=3):
        super().__init__(experiment, device=device, tail_length=tail_length)
        self.horizon = experiment.chunk_size
        if type(selected_fm_step) is not int or not 1 <= selected_fm_step <= self.num_steps:
            raise ValueError("Selected FM step is outside the solver schedule")
        self.selected_fm_step = selected_fm_step

    def predict(self, observation, seed):
        with capture_attention(self.policy.model, selected_fm_step=self.selected_fm_step,
                               horizon=self.horizon) as capture:
            actions = super().predict(observation, seed)
        if capture["calls"] != self.num_steps or not capture["layers"]:
            raise RuntimeError("Missing AutoHorizon attention capture")
        if len(set(capture["layer_indices"])) != len(capture["layer_indices"]):
            raise RuntimeError("Duplicate self-attention layer at selected FM step")
        layers = self.torch.stack(capture["layers"])
        self.last_diagnostics.update({
            "attention": layers.mean(dim=0).cpu().tolist(),
            "attention_layers": layers.cpu().tolist(),
            "attention_capture": {
                "selected_fm_step": self.selected_fm_step,
                "fm_time": float(capture["fm_time"].item()),
                "layer_indices": capture["layer_indices"], "head_count": capture["head_count"],
                "horizon": self.horizon, "statistics_dtype": "float32",
                "extraction": "self_attention_layers_only; mean_heads_then_layers_before_row_normalization",
            },
        })
        return actions

    def sampler_manifest(self):
        from lerobot.policies.smolvla import smolvlm_with_expert
        return {**super().sampler_manifest(),
                Path(smolvlm_with_expert.__file__).name: file_sha256(smolvlm_with_expert.__file__)}


class SelectorAdapter:
    """Connect the frozen executor's selector interface to captured attention."""

    def __init__(self, selector, policy):
        self.selector, self.policy = selector, policy
        self.execution_cap = selector.execution_cap

    @property
    def last_decision(self):
        return self.selector.last_decision

    def reset(self):
        self.selector.reset()

    def select(self, _variance):
        return self.selector.select(self.policy.last_diagnostics["attention"])
