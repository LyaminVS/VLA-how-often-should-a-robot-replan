"""Frozen paired comparison of fixed horizons and adaptive DVAC horizons."""

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from vla_replan.config import Experiment, Task, digest


@dataclass(frozen=True)
class AdaptiveEpisode:
    suite: str
    task: str
    state: int
    policy_seed: int
    method: str
    max_steps: int

    @property
    def k(self):
        return int(self.method.split("_", 1)[1]) if self.method.startswith("fixed_") else None

    @property
    def pair_id(self):
        return f"{self.suite}/{self.task}/state{self.state}/seed{self.policy_seed}"

    @property
    def id(self):
        return f"{self.suite}__{self.task}__s{self.state}__r{self.policy_seed}__{self.method}"

    @property
    def env_seed(self):
        return int(digest([self.suite, self.task, self.state])[:8], 16)

    def asdict(self):
        return {**asdict(self), "k": self.k}


@dataclass(frozen=True)
class DVACSettings:
    tail_length: int = 5
    window_size: int = 5
    alpha: float = 2.0
    min_steps: int = 1
    max_steps: int = 50
    bootstrap: str = "first_call_k1"

    def validate(self, base):
        if type(self.tail_length) is not int or not 2 <= self.tail_length <= base.num_denoising_steps:
            raise ValueError("tail_length must be in [2, num_denoising_steps]")
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if type(self.alpha) not in (float, int) or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if type(self.min_steps) is not int or self.min_steps != 1:
            raise ValueError("This experiment fixes min_steps=1")
        if type(self.max_steps) is not int or self.max_steps != base.chunk_size:
            raise ValueError("This experiment fixes max_steps=H")
        if self.bootstrap != "first_call_k1":
            raise ValueError("Use the preregistered first-call K=1 initialization")


@dataclass(frozen=True)
class AdaptiveExperiment:
    base: Experiment
    methods: tuple[str, ...]
    dvac: DVACSettings

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        base_data = data.pop("base")
        base_data["tasks"] = tuple(Task(**task) for task in base_data["tasks"])
        for field in ("intervals", "initial_states", "policy_seeds"):
            base_data[field] = tuple(base_data[field])
        result = cls(
            base=Experiment(**base_data),
            methods=tuple(data.pop("methods")),
            dvac=DVACSettings(**data.pop("dvac")),
        )
        if data:
            raise ValueError(f"Unexpected configuration keys: {sorted(data)}")
        result.validate()
        return result

    def validate(self):
        self.base.validate()
        self.dvac.validate(self.base)
        if self.base.precision != "fp32":
            raise ValueError("The DVAC comparison requires fp32")
        if not self.methods or len(set(self.methods)) != len(self.methods) or "dvac" not in self.methods:
            raise ValueError("Methods must be unique and include dvac")
        intervals = []
        for method in self.methods:
            if method == "dvac":
                continue
            match = re.fullmatch(r"fixed_([1-9][0-9]*)", method)
            if not match or not 1 <= int(match[1]) <= self.base.chunk_size:
                raise ValueError(f"Invalid method: {method}")
            intervals.append(int(match[1]))
        if sorted(intervals) != sorted(self.base.intervals):
            raise ValueError("base.intervals must match the fixed methods")

    def asdict(self):
        return {"base": self.base.asdict(), "methods": list(self.methods), "dvac": asdict(self.dvac)}

    @property
    def fingerprint(self):
        return digest(self.asdict())

    def episodes(self):
        blocks = [
            (task, state, seed)
            for task in self.base.tasks
            for state in self.base.initial_states
            for seed in self.base.policy_seeds
        ]
        rng = random.Random(self.base.order_seed)
        rng.shuffle(blocks)
        episodes = []
        for task, state, seed in blocks:
            methods = list(self.methods)
            rng.shuffle(methods)
            episodes.extend(
                AdaptiveEpisode(task.suite, task.name, state, seed, method, task.max_steps)
                for method in methods
            )
        return episodes

    def budget(self):
        episodes = self.episodes()
        return {
            "episodes": len(episodes),
            "matched_blocks": len(episodes) // len(self.methods),
            "control_steps_upper_bound": sum(e.max_steps for e in episodes),
            "model_calls_upper_bound": sum(
                (e.max_steps + (e.k or self.dvac.min_steps) - 1) // (e.k or self.dvac.min_steps)
                for e in episodes
            ),
        }
