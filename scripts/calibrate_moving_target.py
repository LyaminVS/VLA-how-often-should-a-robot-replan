"""Measure physical target motion without loading the VLA (run on Linux host)."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moving_target.config import MotionSpec
from moving_target.environment import MovingTargetEnvironment
from vla_replan.config import Episode, Experiment
from vla_replan.storage import atomic_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--force", type=float, default=1.0)
    parser.add_argument("--force-offset-z", type=float, default=-0.03)
    parser.add_argument("--state", type=int, default=20)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    base = Experiment.load("configs/study.json")
    base = replace(base, tasks=(base.tasks[0],), initial_states=(args.state,))
    motion = MotionSpec(kp_n_per_m=args.kp, kd_ns_per_m=args.kd, max_force_n=args.force,
                        force_application_offset_z_m=args.force_offset_z)
    motion.validate()
    env = MovingTargetEnvironment(base, motion)
    episode = Episode(base.tasks[0].suite, base.tasks[0].name, args.state, 101, 10, args.steps)
    results = {}
    try:
        for condition in ("static", "slow", "fast"):
            env.set_condition(condition)
            obs, reset = env.reset(episode)
            writer = None
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(output / f"{condition}.mp4"), fps=20,
                    codec="libx264", quality=7, ffmpeg_params=["-threads", "1"])
            records = []
            try:
                for step in range(args.steps):
                    records.append({"step": step, **env.telemetry()})
                    if writer:
                        writer.append_data(obs.frame)
                    obs, _, done = env.step([0, 0, 0, 0, 0, 0, -1])
                    if done:
                        break
                records.append({"step": len(records), **env.telemetry()})
                if writer:
                    writer.append_data(obs.frame)
            finally:
                if writer:
                    writer.close()
            results[condition] = {"reset": reset, "final": env.telemetry()}
            write_jsonl(output / f"{condition}.jsonl", records)
            print(json.dumps({"condition": condition, **results[condition]}), flush=True)
        atomic_json(output / "calibration.json", {
            "role": "physics calibration without policy, excluded from experiment results",
            "motion": motion.asdict(), "state": args.state, "steps": args.steps, "results": results,
        })
    finally:
        env.close()


if __name__ == "__main__":
    main()
