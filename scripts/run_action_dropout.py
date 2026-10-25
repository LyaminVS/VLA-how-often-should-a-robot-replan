#!/usr/bin/env python3
"""Run paired translation-command dropout without changing the frozen policy."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from action_dropout.config import DropoutExperiment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/action-dropout.json")
    parser.add_argument("--output", default="outputs/action-dropout")
    parser.add_argument("--integrity-check", default="outputs/action-dropout-check/invariance.json")
    parser.add_argument("--baseline-root", default="outputs/capped")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--deadline", help="ISO date/time with offset; stop between episodes")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    cfg = DropoutExperiment.load(args.config)
    if args.plan:
        episodes = sorted(cfg.episodes(), key=lambda episode: episode.method == "fixed_1")
        print(json.dumps({"fingerprint": cfg.fingerprint, "episodes": [
            {"episode_id": episode.id, **episode.asdict()} for episode in episodes]}, indent=2))
    else:
        from action_dropout.runner import check, run
        if args.check:
            check(cfg, args.integrity_check, args.baseline_root)
        else:
            run(cfg, args.output, args.integrity_check, args.baseline_root,
                methods=args.methods, limit=args.limit, deadline=args.deadline)


if __name__ == "__main__":
    main()
