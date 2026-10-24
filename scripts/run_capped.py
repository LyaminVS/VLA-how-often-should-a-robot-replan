#!/usr/bin/env python3
"""Collect capped-DVAC variants while retaining the generated H=50 action chunk."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capped_replan.config import CappedExperiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capped.json")
    parser.add_argument("--output", default="outputs/capped")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--integrity-check", default="outputs/capped-check/invariance.json")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--episode-ids", type=Path, help="JSON file containing selected episode IDs")
    parser.add_argument("--reference-roots", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, help="Collect at most this many additional episodes")
    parser.add_argument("--plan", action="store_true", help="Print the frozen order without loading the model")
    args = parser.parse_args()
    config = CappedExperiment.load(args.config)
    if args.plan:
        print(json.dumps({
            "config_fingerprint": config.fingerprint,
            "budget": config.budget(),
            "episodes": [{"episode_id": e.id, **e.asdict()} for e in config.episodes()],
        }, indent=2))
        return
    from capped_replan.runner import run

    ids = json.loads(args.episode_ids.read_text()) if args.episode_ids else None
    run(config, args.output, device=args.device, limit=args.limit, integrity_check=args.integrity_check,
        methods=args.methods, episode_ids=ids, reference_roots=args.reference_roots)


if __name__ == "__main__":
    main()
