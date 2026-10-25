"""Paired translation-dropout settings without modifying the frozen policy setup."""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from adaptive_replan.config import DVACSettings
from capped_replan.config import CappedEpisode, CappedExperiment
from vla_replan.config import Experiment, digest


ALL_METHODS = ("fixed_1", "fixed_5", "fixed_10", "fixed_25", "fixed_50",
               "dvac_cap_5", "dvac_cap_10", "dvac_cap_25", "dvac_cap_50")


def validate_probability(value):
    """Use integer percentage probabilities so episode identifiers are unique."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Dropout probability must be finite and in [0, 1]")
    percent = round(float(value) * 100)
    if not math.isclose(float(value) * 100, percent, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Dropout probability must be an integer percentage")
    return percent / 100.0


def validate_seed(value):
    if type(value) is not int or value < 0:
        raise ValueError("Disturbance seed must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class DropoutEpisode(CappedEpisode):
    dropout_p: float
    disturbance_seed: int

    def __post_init__(self):
        object.__setattr__(self, "dropout_p", validate_probability(self.dropout_p))
        validate_seed(self.disturbance_seed)

    @property
    def id(self):
        return f"{super().id}__p{round(self.dropout_p * 100):03d}"


@dataclass(frozen=True)
class DropoutExperiment:
    base: Experiment
    methods: tuple[str, ...]
    dvac: DVACSettings
    probabilities: tuple[float, ...]
    disturbance_seed: int

    @classmethod
    def load(cls, path):
        return cls.fromdict(json.loads(Path(path).read_text()))

    @classmethod
    def fromdict(cls, data):
        data = dict(data)
        capped = CappedExperiment.fromdict({name: data.pop(name) for name in ("base", "methods", "dvac")})
        probabilities = tuple(validate_probability(value) for value in data.pop("probabilities"))
        result = cls(capped.base, capped.methods, capped.dvac, probabilities,
                     validate_seed(data.pop("disturbance_seed")))
        if data:
            raise ValueError(f"Unexpected dropout configuration keys: {sorted(data)}")
        result.validate()
        return result

    def validate(self):
        CappedExperiment(self.base, self.methods, self.dvac).validate()
        if not set(self.methods).issubset(ALL_METHODS):
            raise ValueError("Dropout methods must match the retained nine-method comparison")
        if not self.probabilities:
            raise ValueError("At least one dropout probability is required")
        probabilities = tuple(validate_probability(value) for value in self.probabilities)
        if len(set(probabilities)) != len(probabilities):
            raise ValueError("Dropout probabilities must be unique")
        validate_seed(self.disturbance_seed)

    def asdict(self):
        capped = CappedExperiment(self.base, self.methods, self.dvac).asdict()
        return {**capped, "probabilities": [validate_probability(value) for value in self.probabilities],
                "disturbance_seed": self.disturbance_seed}

    @property
    def fingerprint(self):
        return digest(self.asdict())

    def episodes(self):
        self.validate()
        blocks = [(task, state, seed) for task in self.base.tasks
                  for state in self.base.initial_states for seed in self.base.policy_seeds]
        rng = random.Random(self.base.order_seed)
        rng.shuffle(blocks)
        result = []
        for task, state, seed in blocks:
            variants = [(method, probability) for method in self.methods for probability in self.probabilities]
            rng.shuffle(variants)
            result.extend(DropoutEpisode(task.suite, task.name, state, seed, method, task.max_steps,
                                         probability, self.disturbance_seed)
                          for method, probability in variants)
        return result

    def budget(self):
        episodes = self.episodes()
        return {
            "episodes": len(episodes),
            "physical_task_state_settings": len({(e.suite, e.task, e.state) for e in episodes}),
            "matched_task_state_seed_blocks": len({e.pair_id for e in episodes}),
            "control_steps_upper_bound": sum(e.max_steps for e in episodes),
            "model_calls_upper_bound": sum((e.max_steps + (e.k or 1) - 1) // (e.k or 1)
                                            for e in episodes),
        }
