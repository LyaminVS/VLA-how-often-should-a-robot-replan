"""Build reports and paired demos after a complete real experiment.

Usage: .venv/bin/python scripts/finish_run.py outputs/compact
Existing manual annotations and completed demo files are preserved on retries.
"""

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

# Direct script execution must use the adjacent sources, even if an older package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_replan.analysis import analyze, annotation_template, trace_episode  # noqa: E402
from vla_replan.storage import exclusive_run  # noqa: E402
from vla_replan.video import compare_videos  # noqa: E402


def pair_filename(pair_id):
    name = pair_id.replace("/", "__")
    if len(name.encode()) > 200:
        name = name[:180] + "__" + hashlib.sha256(pair_id.encode()).hexdigest()[:16]
    return name


def generate_once(destination, generate):
    """Publish only complete artifacts, so an interrupted retry never skips a partial file."""
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size == 0:
            raise ValueError(f"Existing artifact is not a nonempty file: {destination}")
        print(f"Already complete: {destination}", flush=True)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finish-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        generate(staged)
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError(f"Artifact generation produced no data: {destination}")
        staged.replace(destination)
    print(f"Created: {destination}", flush=True)
    return True


def finish_run(root):
    root = Path(root).resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"Run manifest not found: {root / 'manifest.json'}")
    # Use the runner's lock to prevent both concurrent finish jobs and analysis of an active run.
    with exclusive_run(root):
        annotations = root / "annotations.csv"
        report = root / "report"
        coverage = analyze(root, report, annotations=annotations if annotations.exists() else None)
        if not annotations.exists():
            annotation_template(root, annotations)
        with (report / "demo_candidates.csv").open(newline="") as handle:
            candidates = list(csv.DictReader(handle))
        if not candidates:
            raise ValueError("The completed run has no paired demo candidates")
        created = 0
        for candidate in candidates:
            left = root / "episodes" / candidate["low_k_episode"]
            right = root / "episodes" / candidate["high_k_episode"]
            summaries = [json.loads((episode / "summary.json").read_text()) for episode in (left, right)]
            if [summary["k"] for summary in summaries] != [1, 50]:
                raise ValueError("This finishing script requires paired K=1 and K=50 episodes")
            if any(summary["pair_id"] != candidate["pair_id"] for summary in summaries):
                raise ValueError(f"Demo candidate identity mismatch: {candidate['pair_id']}")
            name = pair_filename(candidate["pair_id"])
            demo = root / "demos" / f"{name}.mp4"
            created += generate_once(demo, lambda output, a=left, b=right: compare_videos(a, b, output))
            for episode, summary in zip((left, right), summaries, strict=True):
                trace = root / "demos" / f"{name}__k{summary['k']}_trace.png"
                created += generate_once(trace, lambda output, source=episode: trace_episode(source, output))
        result = {
            "run": str(root),
            "coverage": coverage,
            "report": str(report),
            "annotations": str(annotations),
            "comparison_videos": len(candidates),
            "motion_traces": 2 * len(candidates),
            "new_demo_artifacts": created,
        }
        print(json.dumps(result, indent=2), flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing the completed run's manifest.json")
    args = parser.parse_args(argv)
    finish_run(args.root)


if __name__ == "__main__":
    main()
