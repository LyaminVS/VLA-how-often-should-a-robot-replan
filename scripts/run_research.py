"""Run the separately specified adaptive/causal study from the repository root."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".cache/libero"))

from vla_replan.research import run_research  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/research_v1.json")
    parser.add_argument("--output", default="outputs/research_v1")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_research(args.plan, args.output, args.device, args.limit)
