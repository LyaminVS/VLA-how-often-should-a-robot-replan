#!/usr/bin/env python3
"""Run frozen AutoHorizon caps on the matched LIBERO starts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autohorizon_replan.config import AutoHorizonExperiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autohorizon.json")
    parser.add_argument("--output", default="outputs/autohorizon")
    parser.add_argument("--integrity-check", default="outputs/autohorizon-check/invariance.json")
    parser.add_argument("--reference-root", default="outputs/capped")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    cfg = AutoHorizonExperiment.load(args.config)
    if args.plan:
        print(json.dumps({"config_fingerprint": cfg.fingerprint, "episodes": [
            {"episode_id": episode.id, **episode.asdict()} for episode in cfg.episodes()]}, indent=2))
    else:
        from autohorizon_replan.runner import check, run
        if args.check:
            check(cfg, args.integrity_check, args.reference_root)
        else:
            run(cfg, args.output, args.integrity_check, args.reference_root, args.limit)


if __name__ == "__main__":
    main()
