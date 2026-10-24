"""Validated moving-target protocol and paired, randomized episode identities."""

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vla_replan.config import Episode, Experiment, Task, digest

CONDITIONS = ("static", "slow", "fast")
SOUP_TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"


def _finite_number(value, name, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0 if positive else value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class MotionSpec:
    amplitude_m: float = 0.02
    periods_s: dict = field(default_factory=lambda: {"static": None, "slow": 8.0, "fast": 4.0})
    start_delay_s: float = 0.5
    ramp_s: float = 1.0
    kp_n_per_m: float = 100.0
    kd_ns_per_m: float = 0.5
    max_force_n: float = 1.0
    force_application_offset_z_m: float = -0.03
    force_axis: str = "x"
    stop_on_contact: bool = True

    def validate(self):
        for name in ("amplitude_m", "ramp_s", "kp_n_per_m", "max_force_n"):
            _finite_number(getattr(self, name), name)
        for name in ("start_delay_s", "kd_ns_per_m"):
            _finite_number(getattr(self, name), name, positive=False)
        offset = self.force_application_offset_z_m
        if type(offset) not in (int, float) or not math.isfinite(offset) or not -0.05 <= offset <= 0:
            raise ValueError("force_application_offset_z_m must be finite and between -0.05 and 0 metres")
        if not isinstance(self.periods_s, dict) or set(self.periods_s) != set(CONDITIONS):
            raise ValueError("periods_s must contain exactly static, slow and fast")
        if self.periods_s["static"] is not None:
            raise ValueError("The static condition must have a null period")
        for name in ("slow", "fast"):
            _finite_number(self.periods_s[name], f"periods_s.{name}")
        if self.periods_s["slow"] <= self.periods_s["fast"]:
            raise ValueError("The slow period must exceed the fast period")
        if self.force_axis != "x":
            raise ValueError("This protocol applies forces along the world x axis")
        if self.stop_on_contact is not True:
            raise ValueError("The motion force must stop permanently at first finger contact")

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class MovingEpisode:
    base_episode: Episode
    condition: str

    @property
    def id(self):
        return f"{self.base_episode.id}__motion_{self.condition}"

    @property
    def pair_id(self):
        return self.base_episode.pair_id

    @property
    def env_seed(self):
        return self.base_episode.env_seed

    def asdict(self):
        return {**self.base_episode.asdict(), "condition": self.condition}


@dataclass(frozen=True)
class MovingExperiment:
    base_config: Experiment
    motion: MotionSpec

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"base_config", "motion"}:
            raise ValueError("Expected exactly base_config and motion in the moving-target config")
        base_data = dict(data["base_config"])
        base_data["tasks"] = tuple(Task(**task) for task in base_data["tasks"])
        for name in ("intervals", "initial_states", "policy_seeds"):
            base_data[name] = tuple(base_data[name])
        result = cls(Experiment(**base_data), MotionSpec(**data["motion"]))
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self):
        self.base_config.validate()
        self.motion.validate()
        tasks = self.base_config.tasks
        if len(tasks) != 1 or (tasks[0].suite, tasks[0].name) != ("libero_object", SOUP_TASK):
            raise ValueError("This moving-target protocol requires the single alphabet-soup task")
        for name in ("chunk_size", "num_denoising_steps", "control_hz", "settle_steps", "warmup_calls", "order_seed"):
            value = getattr(self.base_config, name)
            if type(value) is not int or value < (0 if name in ("settle_steps", "order_seed") else 1):
                raise ValueError(f"Invalid integer {name}")
        if type(tasks[0].max_steps) is not int or tasks[0].max_steps < 1:
            raise ValueError("Task max_steps must be a positive integer")

    def asdict(self):
        return {"base_config": self.base_config.asdict(), "motion": self.motion.asdict()}

    @property
    def fingerprint(self):
        return digest(self.asdict())

    def episodes(self):
        base = self.base_config
        blocks = [(task, state, seed) for task in base.tasks for state in base.initial_states for seed in base.policy_seeds]
        rng = random.Random(base.order_seed)
        rng.shuffle(blocks)
        result = []
        for task, state, seed in blocks:
            treatments = [(condition, k) for condition in CONDITIONS for k in base.intervals]
            rng.shuffle(treatments)
            result.extend(
                MovingEpisode(Episode(task.suite, task.name, state, seed, k, task.max_steps), condition)
                for condition, k in treatments
            )
        return result

    def budget(self):
        episodes = self.episodes()
        return {
            "data_kind": "moving_target_stress_test",
            "episodes": len(episodes),
            "paired_state_seed_blocks": len(self.base_config.initial_states) * len(self.base_config.policy_seeds),
            "conditions": list(CONDITIONS),
            "control_steps_upper_bound": sum(e.base_episode.max_steps for e in episodes),
            "model_calls_upper_bound": sum((e.base_episode.max_steps + e.base_episode.k - 1) // e.base_episode.k for e in episodes),
            "replanning_hz": {k: self.base_config.control_hz / k for k in self.base_config.intervals},
        }
