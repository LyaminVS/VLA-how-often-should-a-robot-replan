#!/usr/bin/env python3
"""Collect the frozen DVAC comparison on the Linux simulator host."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_replan.config import AdaptiveExperiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dvac.json")
    parser.add_argument("--output", default="outputs/dvac")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--integrity-check", default="outputs/dvac-check/invariance.json")
    parser.add_argument("--limit", type=int, help="Collect at most this many additional episodes")
    parser.add_argument("--plan", action="store_true", help="Print the frozen order without loading the model")
    args = parser.parse_args()
    config = AdaptiveExperiment.load(args.config)
    if args.plan:
        print(json.dumps({
            "config_fingerprint": config.fingerprint,
            "budget": config.budget(),
            "episodes": [{"episode_id": e.id, **e.asdict()} for e in config.episodes()],
        }, indent=2))
        return
    from adaptive_replan.runner import run

    run(config, args.output, device=args.device, limit=args.limit, integrity_check=args.integrity_check)


if __name__ == "__main__":
    main()
