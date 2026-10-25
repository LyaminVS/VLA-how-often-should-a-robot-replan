"""Frozen paired evaluation of AutoHorizon execution caps."""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from adaptive_replan.config import DVACSettings
from vla_replan.config import Experiment, Task, digest


@dataclass(frozen=True)
class AutoHorizonSettings:
    selected_fm_step: int = 3
    hold_thr: float = 0.3
    max_entropy_q: float = 0.9
    implementation: str = "official_c7504f1"


@dataclass(frozen=True)
class AutoEpisode:
    suite: str
    task: str
    state: int
    policy_seed: int
    method: str
    max_steps: int

    @property
    def adaptive(self):
        return True

    @property
    def k(self):
        return None

    @property
    def execution_cap(self):
        match = re.fullmatch(r"autohorizon_cap_([1-9][0-9]*)", self.method)
        if not match:
            raise ValueError("Invalid AutoHorizon method")
        return int(match[1])

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
        return {**asdict(self), "k": None, "execution_cap": self.execution_cap}


@dataclass(frozen=True)
class AutoHorizonExperiment:
    base: Experiment
    methods: tuple[str, ...]
    dvac: DVACSettings
    autohorizon: AutoHorizonSettings

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
        result = cls(Experiment(**base), tuple(data.pop("methods")),
                     DVACSettings(**data.pop("dvac")), AutoHorizonSettings(**data.pop("autohorizon")))
        if data:
            raise ValueError(f"Unexpected settings: {sorted(data)}")
        result.validate()
        return result

    def validate(self):
        self.base.validate()
        self.dvac.validate(self.base)
        if self.base.precision != "fp32":
            raise ValueError("This comparison requires FP32")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("Methods must be nonempty and unique")
        settings = self.autohorizon
        if (type(settings.selected_fm_step) is not int
                or settings.selected_fm_step != 3
                or settings.selected_fm_step > self.base.num_denoising_steps
                or settings.hold_thr != 0.3 or settings.max_entropy_q != 0.9
                or settings.implementation != "official_c7504f1"):
            raise ValueError("Use the preregistered official AutoHorizon settings")
        for method in self.methods:
            if not 1 <= AutoEpisode("", "", 0, 0, method, 1).execution_cap <= self.base.chunk_size:
                raise ValueError("Execution cap exceeds prediction horizon")

    def asdict(self):
        return {"base": self.base.asdict(), "methods": list(self.methods),
                "dvac": asdict(self.dvac), "autohorizon": asdict(self.autohorizon)}

    @property
    def fingerprint(self):
        return digest(self.asdict())

    def episodes(self):
        blocks = [(task, state, seed) for task in self.base.tasks
                  for state in self.base.initial_states for seed in self.base.policy_seeds]
        rng = random.Random(self.base.order_seed)
        rng.shuffle(blocks)
        result = []
        for task, state, seed in blocks:
            methods = list(self.methods)
            rng.shuffle(methods)
            result.extend(AutoEpisode(task.suite, task.name, state, seed, method, task.max_steps)
                          for method in methods)
        return result
