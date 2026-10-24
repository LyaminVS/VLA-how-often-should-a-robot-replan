#!/usr/bin/env python3
"""Plan or execute the separately registered moving-target stress test."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moving_target.config import MovingExperiment
from vla_replan.storage import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Validate protocol and show budget without loading a model")
    plan.add_argument("config")
    plan.add_argument("--output")
    run_parser = sub.add_parser("run", help="Run on Linux/CUDA; resume only checksum-verified episodes")
    run_parser.add_argument("config")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--device", choices=["cuda"], default="cuda")
    run_parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        config = MovingExperiment.load(args.config)
        if args.command == "plan":
            record = {
                "config_fingerprint": config.fingerprint,
                "config": config.asdict(),
                "budget": config.budget(),
                "episodes": [{**episode.asdict(), "id": episode.id, "pair_id": episode.pair_id, "env_seed": episode.env_seed} for episode in config.episodes()],
            }
            print(json.dumps(record["budget"], indent=2))
            if args.output:
                if Path(args.output).exists():
                    raise FileExistsError(args.output)
                atomic_json(args.output, record)
        else:
            from moving_target.runner import run

            print(json.dumps(run(config, args.output, args.device, args.limit), indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError, TypeError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
