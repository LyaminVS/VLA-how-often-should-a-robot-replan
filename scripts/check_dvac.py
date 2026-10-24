#!/usr/bin/env python3
"""Check instrumented/plain policy equality on held-out inputs before collection."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_replan.integrity import source_manifest
from adaptive_replan.config import AdaptiveExperiment
from adaptive_replan.policy import InstrumentedSmolVLAPolicy
from vla_replan.backend import LiberoEnvironment, SmolVLAPolicy
from vla_replan.config import Episode, Experiment, Task
from vla_replan.storage import atomic_json, runtime_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    adaptive_config = AdaptiveExperiment.load(args.config)
    base = raw["base"]
    base["tasks"] = tuple(Task(**task) for task in base["tasks"])
    for field in ("intervals", "initial_states", "policy_seeds"):
        base[field] = tuple(base[field])
    config = Experiment(**base)
    config.validate()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    policy = InstrumentedSmolVLAPolicy(config, tail_length=5)
    env = LiberoEnvironment(config)
    results = []
    try:
        for task in config.tasks:
            episode = Episode(task.suite, task.name, 21, 101, 1, task.max_steps)
            observation, reset_info = env.reset(episode)
            policy.reset()
            start = time.perf_counter()
            plain = SmolVLAPolicy.predict(policy, observation, 101)
            plain_seconds = time.perf_counter() - start
            policy.reset()
            start = time.perf_counter()
            observed = policy.predict(observation, 101)
            observed_seconds = time.perf_counter() - start
            error = max(abs(x - y) for a, b in zip(plain, observed, strict=True)
                        for x, y in zip(a, b, strict=True))
            if plain != observed or error != 0:
                raise AssertionError(f"Instrumentation altered actions for {task.name}: {error}")
            diagnostics = policy.last_diagnostics
            row = {
                "task": task.name, "suite": task.suite, "initial_state": 21,
                "sampling_seed": 101, "exact_action_equality": True,
                "max_absolute_action_error": error,
                "action_sha256": hashlib.sha256(json.dumps(plain).encode()).hexdigest(),
                "plain_s": plain_seconds, "instrumented_s": observed_seconds,
                "timing_note": "Order and initialization make these unsuitable as an overhead estimate.",
                "diagnostics": diagnostics, **reset_info,
            }
            results.append(row)
            print(f"PASS {task.name}: identical actions, {len(diagnostics['fm_times'])} tail estimates", flush=True)
    finally:
        env.close()
    atomic_json(output, {
        "passed": True, "checks": results, "source_manifest": source_manifest(),
        "sampler_manifest": policy.sampler_manifest(), "runtime": runtime_metadata(),
        "hardware": policy.runtime,
        "config_fingerprint": adaptive_config.fingerprint,
        "config": adaptive_config.asdict(),
    })


if __name__ == "__main__":
    main()
