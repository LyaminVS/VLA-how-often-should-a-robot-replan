#!/usr/bin/env python3
"""Check historical reset and first-action compatibility without executing a rollout."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_replan.policy import InstrumentedSmolVLAPolicy
from capped_replan.config import CappedExperiment
from capped_replan.integrity import file_sha256, source_manifest
from capped_replan.reference import load_references, validate_reference_runtime
from vla_replan.backend import LiberoEnvironment
from vla_replan.core import sampling_seed
from vla_replan.storage import atomic_json, runtime_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--reference-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = CappedExperiment.load(args.config)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    references, audit = load_references(args.reference_roots, cfg)
    sources = source_manifest()
    policy = InstrumentedSmolVLAPolicy(cfg.base, tail_length=cfg.dvac.tail_length)
    validate_reference_runtime(audit, runtime_metadata(), policy.runtime)
    env = LiberoEnvironment(cfg.base)
    pairs = {}
    for episode in cfg.episodes():
        pairs.setdefault(episode.pair_id, episode)
    checks = []
    try:
        for pair_id, episode in sorted(pairs.items()):
            reference = references[pair_id]
            observation, reset = env.reset(episode)
            for field in ("initial_state_sha256", "initial_observation_sha256"):
                if reset[field] != reference[field]:
                    raise ValueError(f"Historical {field} mismatch: {pair_id}")
            seed = sampling_seed(episode.policy_seed, 0)
            if seed != reference["sampling_seed"]:
                raise ValueError(f"Historical sampling seed mismatch: {pair_id}")
            policy.reset()
            actions = policy.predict(observation, seed)
            previous = reference["actions"]
            error = max(abs(a-b) for current, old in zip(actions, previous, strict=True)
                        for a, b in zip(current, old, strict=True))
            if actions != previous or error != 0:
                raise ValueError(f"Historical first actions changed: {pair_id}, maximum error={error}")
            checks.append({
                "pair_id": pair_id, "suite": episode.suite, "task": episode.task,
                "state": episode.state, "policy_seed": episode.policy_seed,
                "sampling_seed": seed, "initial_state_sha256": reset["initial_state_sha256"],
                "initial_observation_sha256": reset["initial_observation_sha256"],
                "exact_action_equality": True, "max_absolute_action_error": error,
                "action_sha256": hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
                "reference_episode_ids": reference["reference_episode_ids"],
            })
            print(f"PASS {len(checks)}/{len(pairs)} {pair_id}: exact historical first chunk", flush=True)
    finally:
        env.close()
    if sources != source_manifest():
        raise ValueError("Collection sources changed during compatibility check")
    for path, expected in audit["files"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"Historical source changed during compatibility check: {path}")
    atomic_json(output, {
        "passed": True, "check_kind": "historical_first_prediction_compatibility",
        "robot_actions_executed": 0, "pairs": len(checks), "checks": checks,
        "config_fingerprint": cfg.fingerprint, "config": cfg.asdict(),
        "source_manifest": sources, "runtime": runtime_metadata(),
        "hardware": policy.runtime, "sampler_manifest": policy.sampler_manifest(),
        "references": audit,
        "scope": "Exact reset and first prediction equality on all selected starts; not a proof for all later observations.",
    })


if __name__ == "__main__":
    main()
