"""Observe the existing FM solver without replacing its integration or action output."""

import inspect
import math
import time
from contextlib import contextmanager
from pathlib import Path

from vla_replan.backend import SmolVLAPolicy

from .integrity import file_sha256


def variance_from_estimates(estimates):
    """Population variance across solver iterations, summed over valid action dimensions.

    Captured z values are float32. Statistics use Python float64 and math.fsum so
    saved diagnostics can be independently replayed without a CUDA dependency.
    """
    if len(estimates) < 2:
        raise ValueError("Variance requires at least two clean-action estimates")
    horizon = len(estimates[0])
    dimensions = len(estimates[0][0]) if horizon else 0
    if not horizon or not dimensions:
        raise ValueError("Empty clean-action estimates")
    if any(len(item) != horizon or any(len(row) != dimensions for row in item) for item in estimates):
        raise ValueError("Inconsistent estimate shapes")
    if any(not math.isfinite(v) for item in estimates for row in item for v in row):
        raise ValueError("Nonfinite clean-action estimate")
    count = len(estimates)
    result = []
    for j in range(horizon):
        terms = []
        for d in range(dimensions):
            values = [float(item[j][d]) for item in estimates]
            mean = math.fsum(values) / count
            terms.append(math.fsum((v - mean) ** 2 for v in values) / count)
        result.append(math.fsum(terms))
    return result


@contextmanager
def capture_denoising(model, *, num_steps, tail_length, action_dim):
    """Temporarily wrap denoise_step; preserve the original velocity tensor verbatim."""
    if not 2 <= tail_length <= num_steps or action_dim < 1:
        raise ValueError("Invalid denoising capture dimensions")
    original = model.denoise_step
    previous = model.__dict__.get("denoise_step")
    had_instance_override = "denoise_step" in model.__dict__
    signature = inspect.signature(original)
    capture = {"calls": 0, "estimates": [], "times": []}

    def observe(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        x = bound.arguments["x_t"]
        timestep = bound.arguments["timestep"]
        velocity = original(*args, **kwargs)
        capture["calls"] += 1
        if capture["calls"] > num_steps:
            raise RuntimeError("Unexpected extra velocity evaluation")
        if capture["calls"] > num_steps - tail_length:
            if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] < action_dim:
                raise ValueError("Expected one action chunk with enough valid dimensions")
            clean = x[..., :action_dim] - timestep[:, None, None] * velocity[..., :action_dim]
            capture["estimates"].append(clean.detach())
            capture["times"].append(timestep.detach())
        return velocity

    model.denoise_step = observe
    try:
        yield capture
    finally:
        if had_instance_override:
            model.denoise_step = previous
        else:
            delattr(model, "denoise_step")


class InstrumentedSmolVLAPolicy(SmolVLAPolicy):
    def __init__(self, experiment, device="cuda", tail_length=5):
        super().__init__(experiment, device)
        if not 2 <= tail_length <= experiment.num_denoising_steps:
            raise ValueError("Require 2 <= tail length <= number of FM iterations")
        if self.policy._rtc_enabled():
            raise ValueError("This synchronous study requires RTC disabled")
        self.tail_length = tail_length
        self.num_steps = experiment.num_denoising_steps
        self.action_dim = self.policy.config.action_feature.shape[0]
        if self.action_dim != 7:
            raise ValueError("The LIBERO experiment requires seven valid action dimensions")
        self.last_diagnostics = None

    def predict(self, observation, seed):
        self.last_diagnostics = None
        with capture_denoising(self.policy.model, num_steps=self.num_steps,
                               tail_length=self.tail_length, action_dim=self.action_dim) as capture:
            actions = super().predict(observation, seed)
        if capture["calls"] != self.num_steps:
            raise RuntimeError("Sampler used an unexpected number of velocity evaluations")
        start = time.perf_counter()
        estimates = self.torch.stack(capture["estimates"])[:, 0].cpu().tolist()
        times = self.torch.stack(capture["times"])[:, 0].cpu().tolist()
        expected = [1 - i / self.num_steps for i in range(self.num_steps - self.tail_length, self.num_steps)]
        if any(abs(a - b) > 1e-6 for a, b in zip(times, expected, strict=True)):
            raise RuntimeError("Unexpected flow-time schedule")
        self.last_diagnostics = {
            "clean_estimates": estimates,
            "fm_times": times,
            "variance": variance_from_estimates(estimates),
            "velocity_evaluations": capture["calls"],
            "action_dimensions": self.action_dim,
            "statistics_dtype": "float64",
            "diagnostics_finalize_s": time.perf_counter() - start,
        }
        return actions

    def sampler_manifest(self):
        from lerobot.policies.common import flow_matching
        from lerobot.policies.smolvla import modeling_smolvla

        return {Path(module.__file__).name: file_sha256(module.__file__)
                for module in (flow_matching, modeling_smolvla)}
