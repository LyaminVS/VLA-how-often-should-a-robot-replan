import argparse
import json
import platform
from pathlib import Path

from .config import Experiment
from .storage import atomic_json, runtime_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="VLA replanning experiment: K executed actions per observation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Validate config and calculate experiment budget; no model imports")
    plan.add_argument("config")
    plan.add_argument("--output")
    plan.add_argument("--seconds-per-call", type=float)
    sub.add_parser("doctor", help="Inspect local capabilities without downloading model weights")
    estimate = sub.add_parser("estimate", help="Estimate runtime from measured pilot timings")
    estimate.add_argument("config")
    estimate.add_argument("--from-run", required=True)
    run = sub.add_parser("run", help="Run real SmolVLA + LIBERO on Linux/CUDA; automatically resume")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cuda", choices=["cuda"])
    run.add_argument(
        "--limit", type=int, help="Maximum new episodes this invocation; does not change experiment"
    )
    analyze = sub.add_parser("analyze", help="Paired cluster bootstrap and figures from real completed runs")
    analyze.add_argument("root")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--allow-incomplete", action="store_true")
    analyze.add_argument("--draws", type=int, default=5000)
    analyze.add_argument("--reference", type=int, default=10)
    analyze.add_argument("--annotations")
    annotate = sub.add_parser(
        "annotations", help="Create a manual video-review worksheet; never overwrite it"
    )
    annotate.add_argument("root")
    annotate.add_argument("--output", required=True)
    compare = sub.add_parser("compare", help="Combine paired episode videos at identical simulation time")
    compare.add_argument("left", help="First episode directory")
    compare.add_argument("right", help="Second episode directory")
    compare.add_argument("--output", required=True)
    trace = sub.add_parser("trace", help="Plot one episode's motion, gripper actions and replan times")
    trace.add_argument("episode")
    trace.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            config = Experiment.load(args.config)
            budget = config.budget()
            if args.seconds_per_call is not None:
                if args.seconds_per_call <= 0:
                    raise ValueError("seconds-per-call must be positive")
                budget["inference_hours_upper_bound"] = (
                    budget["model_calls_upper_bound"] * args.seconds_per_call / 3600
                )
            print(json.dumps(budget, indent=2))
            if args.output:
                path = Path(args.output)
                if path.exists():
                    raise FileExistsError(path)
                atomic_json(
                    path,
                    {
                        "config_fingerprint": config.fingerprint,
                        "budget": budget,
                        "episodes": [
                            {**e.asdict(), "id": e.id, "env_seed": e.env_seed} for e in config.episodes()
                        ],
                    },
                )
        elif args.command == "doctor":
            info = runtime_metadata()
            info["real_rollouts_supported_os"] = platform.system() == "Linux"
            try:
                import torch

                info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    info["gpu"] = torch.cuda.get_device_name(0)
                    info["vram_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
            except ImportError:
                info["cuda_available"] = False
            print(json.dumps(info, indent=2))
        elif args.command == "run":
            from .runner import run

            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            run(Experiment.load(args.config), args.output, args.device, args.limit)
        elif args.command == "estimate":
            from .analysis import estimate_budget

            print(json.dumps(estimate_budget(Experiment.load(args.config), args.from_run), indent=2))
        elif args.command == "analyze":
            from .analysis import analyze

            print(
                json.dumps(
                    analyze(
                        args.root,
                        args.output,
                        allow_incomplete=args.allow_incomplete,
                        draws=args.draws,
                        reference=args.reference,
                        annotations=args.annotations,
                    ),
                    indent=2,
                )
            )
        elif args.command == "annotations":
            from .analysis import annotation_template

            annotation_template(args.root, args.output)
        elif args.command == "compare":
            from .video import compare_videos

            compare_videos(args.left, args.right, args.output)
        elif args.command == "trace":
            from .analysis import trace_episode

            trace_episode(args.episode, args.output)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
