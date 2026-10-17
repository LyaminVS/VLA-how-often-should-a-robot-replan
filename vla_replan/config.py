"""Experiment specification, immutable episode identities, and balanced run ordering."""

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    suite: str
    name: str
    max_steps: int


@dataclass(frozen=True)
class Episode:
    suite: str
    task: str
    state: int
    policy_seed: int
    k: int
    max_steps: int

    @property
    def pair_id(self):
        return f"{self.suite}/{self.task}/state{self.state}/seed{self.policy_seed}"

    @property
    def id(self):
        return f"{self.suite}__{self.task}__s{self.state}__r{self.policy_seed}__k{self.k}"

    @property
    def env_seed(self):
        # Policy sampling replicates must not alter the physical starting situation.
        return int(digest([self.suite, self.task, self.state])[:8], 16)

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    name: str
    model_id: str
    model_revision: str
    backbone_id: str
    backbone_revision: str
    tasks: tuple[Task, ...]
    intervals: tuple[int, ...]
    initial_states: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    order_seed: int = 20261017
    chunk_size: int = 50
    num_denoising_steps: int = 10
    control_hz: int = 20
    settle_steps: int = 10
    precision: str = "fp32"
    record_video: bool = True
    warmup_calls: int = 2

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        data["tasks"] = tuple(Task(**t) for t in data["tasks"])
        for field in ("intervals", "initial_states", "policy_seeds"):
            data[field] = tuple(data[field])
        result = cls(**data)
        result.validate()
        return result

    def validate(self):
        for revision in (self.model_revision, self.backbone_revision):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Pin model and backbone revisions to full 40-character commit hashes")
        for values, field, minimum in (
            (self.intervals, "intervals", 1),
            (self.initial_states, "initial_states", 0),
            (self.policy_seeds, "policy_seeds", 0),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field} must be nonempty and contain no duplicates")
            if any(type(x) is not int or x < minimum for x in values):
                raise ValueError(f"Invalid {field}")
        if self.chunk_size != 50 or max(self.intervals) > self.chunk_size:
            raise ValueError("This checkpoint uses H=50; require 1 <= K <= H")
        if not self.tasks or len({(t.suite, t.name) for t in self.tasks}) != len(self.tasks):
            raise ValueError("Tasks must be nonempty and unique")
        for task in self.tasks:
            if not re.fullmatch(r"[A-Za-z0-9_]+", task.name) or task.max_steps < 1:
                raise ValueError("Invalid task name or horizon")
            if task.suite not in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
                raise ValueError("Unsupported suite")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("Use fp32 or fp16; RTX 2060 does not natively support bfloat16")
        if self.control_hz != 20 or self.settle_steps < 0 or self.num_denoising_steps < 1:
            raise ValueError("Use 20 Hz and nonnegative settling / positive sampling steps")
        if self.warmup_calls < 1 or type(self.record_video) is not bool:
            raise ValueError("Require warmup_calls >= 1 and a boolean record_video")

    def asdict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return digest(self.asdict())

    def episodes(self):
        blocks = [(t, s, seed) for t in self.tasks for s in self.initial_states for seed in self.policy_seeds]
        rng = random.Random(self.order_seed)
        rng.shuffle(blocks)
        result = []
        for task, state, seed in blocks:
            intervals = list(self.intervals)
            rng.shuffle(intervals)
            result.extend(Episode(task.suite, task.name, state, seed, k, task.max_steps) for k in intervals)
        return result

    def budget(self):
        episodes = self.episodes()
        return {
            "episodes": len(episodes),
            "control_steps_upper_bound": sum(e.max_steps for e in episodes),
            "model_calls_upper_bound": sum((e.max_steps + e.k - 1) // e.k for e in episodes),
            "replanning_hz": {k: self.control_hz / k for k in self.intervals},
        }
