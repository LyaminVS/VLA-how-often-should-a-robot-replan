"""Paired fixed and capped-DVAC schedules with an unchanged generated horizon."""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from adaptive_replan.config import DVACSettings
from vla_replan.config import Experiment, Task, digest


def method_parameters(method, horizon=50):
    match = re.fullmatch(r"(fixed|dvac_cap)_([1-9][0-9]*)", method)
    if not match or not 1 <= int(match[2]) <= horizon:
        raise ValueError(f"Invalid method: {method}")
    return match[1], int(match[2])


@dataclass(frozen=True)
class CappedEpisode:
    suite: str
    task: str
    state: int
    policy_seed: int
    method: str
    max_steps: int

    @property
    def adaptive(self):
        return method_parameters(self.method)[0] == "dvac_cap"

    @property
    def k(self):
        kind, value = method_parameters(self.method)
        return value if kind == "fixed" else None

    @property
    def execution_cap(self):
        kind, value = method_parameters(self.method)
        return value if kind == "dvac_cap" else None

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
        return {**asdict(self), "k": self.k, "execution_cap": self.execution_cap}


@dataclass(frozen=True)
class CappedExperiment:
    base: Experiment
    methods: tuple[str, ...]
    dvac: DVACSettings

    @classmethod
    def load(cls, path):
        return cls.fromdict(json.loads(Path(path).read_text()))

    @classmethod
    def fromdict(cls, data):
        data = dict(data)
        base = dict(data.pop("base"))
        base["tasks"] = tuple(Task(**task) for task in base["tasks"])
        for field in ("intervals", "initial_states", "policy_seeds"):
            base[field] = tuple(base[field])
        result = cls(Experiment(**base), tuple(data.pop("methods")), DVACSettings(**data.pop("dvac")))
        if data:
            raise ValueError(f"Unexpected configuration keys: {sorted(data)}")
        result.validate()
        return result

    def validate(self):
        self.base.validate()
        self.dvac.validate(self.base)
        if self.base.precision != "fp32":
            raise ValueError("The capped comparison requires fp32")
        if not self.methods or len(self.methods) != len(set(self.methods)):
            raise ValueError("Methods must be nonempty and unique")
        fixed = []
        for method in self.methods:
            kind, value = method_parameters(method, self.base.chunk_size)
            if kind == "fixed":
                fixed.append(value)
        if fixed and not set(fixed).issubset(self.base.intervals):
            raise ValueError("Fixed controls must be listed in base.intervals")
        # base.intervals also describes separately retained historical controls.
        # Adaptive-only collection is therefore intentional and supported.

    @property
    def adaptive_caps(self):
        return tuple(method_parameters(method)[1] for method in self.methods if method.startswith("dvac_cap_"))

    def asdict(self):
        return {"base": self.base.asdict(), "methods": list(self.methods), "dvac": asdict(self.dvac)}

    @property
    def fingerprint(self):
        return digest(self.asdict())

    def episodes(self):
        blocks = [(task, state, seed) for task in self.base.tasks
                  for state in self.base.initial_states for seed in self.base.policy_seeds]
        rng = random.Random(self.base.order_seed)
        rng.shuffle(blocks)
        episodes = []
        for task, state, seed in blocks:
            methods = list(self.methods)
            rng.shuffle(methods)
            episodes.extend(CappedEpisode(task.suite, task.name, state, seed, method, task.max_steps)
                            for method in methods)
        return episodes

    def budget(self):
        episodes = self.episodes()
        return {
            "episodes": len(episodes), "matched_blocks": len(episodes) // len(self.methods),
            "control_steps_upper_bound": sum(episode.max_steps for episode in episodes),
            "model_calls_upper_bound": sum((episode.max_steps + (episode.k or 1) - 1) // (episode.k or 1)
                                           for episode in episodes),
        }
