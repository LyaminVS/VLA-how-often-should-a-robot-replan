"""Apply independent translation-command dropout around the frozen executor."""

import hashlib
import json
import math

from capped_replan.core import run_episode as frozen_run_episode

from .config import validate_probability, validate_seed


DROPOUT_MODEL = "translation_dropout_v1"


def uniform_for_step(pair_id, disturbance_seed, step):
    """A private counter stream: method, probability and policy calls are absent.

    Use the leading 53 bits of SHA-256's first eight bytes, big endian, divided
    by 2**53. This maps exactly to a binary64 number in [0,1), with no RNG state
    consumed by the model, simulator, selector or collection order.
    """
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("A nonempty matched-pair identity is required")
    validate_seed(disturbance_seed)
    if type(step) is not int or step < 0:
        raise ValueError("Simulation step must be a nonnegative integer")
    payload = json.dumps([DROPOUT_MODEL, pair_id, disturbance_seed, step],
                         separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 11
    return value / 2**53


def dropout_mask_sha256(pair_id, disturbance_seed, probability, max_steps):
    """Hash all planned mask bits, even when an episode succeeds early."""
    probability = validate_probability(probability)
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("Maximum steps must be a positive integer")
    mask = bytes(int(uniform_for_step(pair_id, disturbance_seed, step) < probability)
                 for step in range(max_steps))
    return hashlib.sha256(mask).hexdigest()


def transform_action(action, uniform, probability):
    """Zero only XYZ of the clipped environment command; preserve rotation/grip."""
    probability = validate_probability(probability)
    if (type(uniform) not in (int, float) or not math.isfinite(uniform)
            or not 0 <= uniform < 1):
        raise ValueError("Dropout uniform must be in [0, 1)")
    if not isinstance(action, (list, tuple)) or len(action) != 7:
        raise ValueError("Expected a seven-coordinate clipped command")
    commanded = list(action)
    if any(type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1
           for value in commanded):
        raise ValueError("Dropout requires finite, already-clipped action coordinates")
    dropped = uniform < probability
    applied = [0.0, 0.0, 0.0, *commanded[3:]] if dropped else list(commanded)
    return applied, dropped


class DropoutEnvironment:
    """Transform only post-reset control actions; failed steps do not advance."""

    def __init__(self, environment, episode):
        self.environment = environment
        self.env = environment
        self.episode = episode
        self.next_step = 0
        self.applied_trace = []

    def reset(self, episode):
        for field in ("pair_id", "dropout_p", "disturbance_seed", "max_steps"):
            if getattr(episode, field) != getattr(self.episode, field):
                raise ValueError(f"Dropout reset identity differs at {field}")
        # LiberoEnvironment.reset performs settling internally, before any
        # wrapper.step call. The disturbance starts at policy-controlled step 0.
        observation, info = self.environment.reset(episode)
        self.next_step = 0
        self.applied_trace = []
        return observation, info

    def preview(self, action):
        if self.next_step >= self.episode.max_steps:
            raise ValueError("Dropout step exceeds the configured episode limit")
        uniform = uniform_for_step(self.episode.pair_id, self.episode.disturbance_seed, self.next_step)
        applied, dropped = transform_action(action, uniform, self.episode.dropout_p)
        return {"step": self.next_step, "commanded_action": list(action), "applied_action": applied,
                "translation_dropped": dropped, "dropout_u": uniform}

    def step(self, action):
        record = self.preview(action)
        # Record the successfully submitted environment input only after the
        # underlying step returns. An exception aborts this attempt; no retry
        # is silently advanced to a different disturbance counter.
        result = self.environment.step(list(record["applied_action"]))
        self.applied_trace.append(record)
        self.next_step += 1
        return result


def run_episode(env, policy, episode, *, selector=None, reference=None, record=None, **kwargs):
    """Keep policy evidence commanded, while logging independently applied inputs."""
    disturbance = DropoutEnvironment(env, episode)

    def recording(frame, row, outcome):
        annotated = dict(row)
        if "action" in row:
            if row["step"] != disturbance.next_step:
                raise ValueError("Pre-action recording and disturbance counters differ")
            annotated.update(disturbance.preview(row["action"]))
        annotated.update(dropout_p=episode.dropout_p, disturbance_seed=episode.disturbance_seed)
        record(frame, annotated, outcome)

    summary, rows, chunks = frozen_run_episode(
        disturbance, policy, episode, selector=selector, reference=reference,
        record=recording if record is not None else None, **kwargs,
    )
    for row, applied in zip(rows, disturbance.applied_trace, strict=True):
        if row["step"] != applied["step"] or row["action"] != applied["commanded_action"]:
            raise ValueError("Executed command differs from the disturbance audit trail")
        row.update({key: value for key, value in applied.items() if key != "step"})
    summary.update(
        dropout_model=DROPOUT_MODEL,
        dropout_mask_sha256=dropout_mask_sha256(episode.pair_id, episode.disturbance_seed,
                                               episode.dropout_p, episode.max_steps),
        dropout_mask_length=episode.max_steps,
        translation_drop_count=sum(item["translation_dropped"] for item in disturbance.applied_trace),
        translation_changed_count=sum(item["commanded_action"][:3] != item["applied_action"][:3]
                                      for item in disturbance.applied_trace),
    )
    return summary, rows, chunks
