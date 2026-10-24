"""Audit diagnostic capture without loading a model, CUDA, or a simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from adaptive_replan.policy import (
    InstrumentedSmolVLAPolicy,
    capture_denoising,
    variance_from_estimates,
)
from vla_replan.backend import SmolVLAPolicy


class Tensor(np.ndarray):
    """Tiny tensor stand-in for the observer's arithmetic/transfer interface."""

    def detach(self):
        return self

    def cpu(self):
        return self


def tensor(values):
    return np.asarray(values, dtype=np.float32).view(Tensor)


class Model:
    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        self.last_velocity = (x_t * 0.2 + timestep[:, None, None]).view(Tensor)
        return self.last_velocity


def integrate(model, noise, *, count=10, time_offset=0):
    x = noise.copy()
    for i in range(count):
        t = tensor([1 - i / 10 + time_offset])
        velocity = model.denoise_step(None, None, x, t)
        assert velocity is model.last_velocity
        x = x - 0.1 * velocity
    return x


def test_variance_is_per_future_action_summed_over_coordinates():
    # First action: coordinate variances 1 and 4. Second action: both zero.
    estimates = [[[0, 0], [1, 5]], [[2, 4], [1, 5]]]
    assert variance_from_estimates(estimates) == [5, 0]
    # Population, not sample variance: mean squared deviations are 8/3.
    estimates = [[[0], [3]], [[2], [3]], [[4], [3]]]
    assert variance_from_estimates(estimates) == pytest.approx([8 / 3, 0])


def test_capture_keeps_last_five_preupdate_estimates_and_preserves_output():
    model = Model()
    noise = tensor(np.arange(27).reshape(1, 3, 9))
    baseline = integrate(model, noise)
    expected = []
    x = noise.copy()
    for i in range(10):
        t = tensor([1 - i / 10])
        velocity = model.denoise_step(None, None, x, t)
        if i >= 5:
            expected.append(x[..., :7] - t[:, None, None] * velocity[..., :7])
        x = x - 0.1 * velocity
    with capture_denoising(model, num_steps=10, tail_length=5, action_dim=7) as captured:
        observed = integrate(model, noise)
    np.testing.assert_array_equal(observed, baseline)
    assert captured["calls"] == 10
    assert len(captured["estimates"]) == 5
    np.testing.assert_array_equal(np.stack(captured["estimates"]), expected)
    np.testing.assert_allclose(np.stack(captured["times"]).ravel(), [.5, .4, .3, .2, .1])
    assert np.stack(captured["estimates"]).shape == (5, 1, 3, 7)
    assert "denoise_step" not in model.__dict__


def test_padding_channels_cannot_change_variance():
    results = []
    for padding in (0, 100000):
        noise = tensor(np.zeros((1, 2, 9)))
        noise[..., 7:] = padding
        model = Model()
        with capture_denoising(model, num_steps=10, tail_length=5, action_dim=7) as captured:
            integrate(model, noise)
        estimates = np.stack(captured["estimates"])[:, 0].tolist()
        results.append(variance_from_estimates(estimates))
    assert results[0] == results[1]


def test_hook_restores_an_existing_instance_override_after_error():
    model = Model()

    def fail(x_t, timestep):
        raise RuntimeError("intentional inference failure")

    model.denoise_step = fail
    with pytest.raises(RuntimeError, match="intentional"):
        with capture_denoising(model, num_steps=10, tail_length=5, action_dim=7):
            model.denoise_step(x_t=tensor(np.zeros((1, 2, 7))), timestep=tensor([1]))
    assert model.denoise_step is fail


def test_hook_restores_class_method_after_extra_velocity_evaluation():
    model = Model()
    with pytest.raises(RuntimeError, match="extra velocity"):
        with capture_denoising(model, num_steps=10, tail_length=5, action_dim=7):
            integrate(model, tensor(np.zeros((1, 2, 7))), count=11)
    assert "denoise_step" not in model.__dict__


@pytest.mark.parametrize(
    "estimates",
    [[], [[[1]]], [[], []], [[[]], [[]]], [[[1]], [[1, 2]]],
     [[[1]], [[float("nan")]]], [[[1]], [[float("inf")]]]],
)
def test_invalid_clean_estimates_fail_closed(estimates):
    with pytest.raises(ValueError):
        variance_from_estimates(estimates)


def make_policy_without_model_load():
    policy = InstrumentedSmolVLAPolicy.__new__(InstrumentedSmolVLAPolicy)
    policy.policy = SimpleNamespace(model=Model())
    policy.num_steps = 10
    policy.tail_length = 5
    policy.action_dim = 7
    policy.torch = SimpleNamespace(stack=lambda values: np.stack(values).view(Tensor))
    return policy


@pytest.mark.parametrize(("count", "offset"), [(9, 0), (10, .01)])
def test_policy_rejects_changed_sampler_schedule(monkeypatch, count, offset):
    policy = make_policy_without_model_load()

    def base_predict(self, observation, seed):
        return integrate(self.policy.model, tensor(np.zeros((1, 2, 7))),
                         count=count, time_offset=offset)[0].tolist()

    monkeypatch.setattr(SmolVLAPolicy, "predict", base_predict)
    with pytest.raises(RuntimeError):
        policy.predict(None, 1)
    assert policy.last_diagnostics is None
    assert "denoise_step" not in policy.policy.model.__dict__
