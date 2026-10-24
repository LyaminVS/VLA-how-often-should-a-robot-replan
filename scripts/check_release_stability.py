"""Additional success diagnostic: replay, open gripper and test goal persistence.

No learned-model calls. Original LIBERO outcomes remain unchanged. Only successful
heldout trials need replay; an official failure is false for the stricter metric.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".cache/libero"))

from vla_replan.accelerator import prepare_platform  # noqa: E402
from vla_replan.backend import LiberoEnvironment, observation_hash  # noqa: E402
from vla_replan.config import Episode, Experiment  # noqa: E402
from vla_replan.research import load_trial, simulation_hash  # noqa: E402
from vla_replan.storage import atomic_json, exclusive_run, runtime_metadata, write_jsonl  # noqa: E402
from vla_replan.video import EpisodeVideo  # noqa: E402


def run(roots, output):
    prepare_platform()
    config = Experiment.load(ROOT / "configs/compact.json")
    trials = []
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["runtime"]["code_sha256"] == runtime_metadata()["code_sha256"]
        assert manifest["runtime"]["packages"] == runtime_metadata()["packages"]
        trials.extend(p.parent for p in root.glob("episodes/*/summary.json")
                      if json.loads(p.read_text())["phase"] == "heldout")
    trials.sort(key=lambda p: p.name)
    output = Path(output)
    with exclusive_run(output):
        signature = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "runtime": runtime_metadata(), "reference_manifests": [hashlib.sha256((r / "manifest.json").read_bytes()).hexdigest() for r in roots],
                    "post_steps": 40, "evaluation_tail_steps": 10, "finger_open_threshold_m": .035}
        if (output / "manifest.json").exists():
            if json.loads((output / "manifest.json").read_text()) != signature:
                raise ValueError("Diagnostic code, runtime or sources changed; use a new output directory")
        else:
            atomic_json(output / "manifest.json", signature)
        env = LiberoEnvironment(config)
        try:
            for i, path in enumerate(trials):
                t = load_trial(path)
                s = t["summary"]
                destination = output / s["trial_id"]
                if (destination / "summary.json").exists():
                    previous = json.loads((destination / "summary.json").read_text())
                    assert previous["reference_steps_sha256"] == hashlib.sha256((path / "steps.jsonl").read_bytes()).hexdigest()
                    continue
                audit = {"trial_id": s["trial_id"], "method": s["method"], "pair_id": s["pair_id"],
                         "official_success": s["success"], "post_open_stable_success": False,
                         "reference_steps_sha256": hashlib.sha256((path / "steps.jsonl").read_bytes()).hexdigest(),
                         "protocol": "40 zero-motion/open-gripper actions after standard success; goal true on last 10, both finger |qpos|>=0.035 on last 10"}
                if not s["success"]:
                    atomic_json(destination / "summary.json", audit | {"replayed": False, "reason": "official_failure"})
                    continue
                ep = Episode(s["suite"], s["task"], s["state"], s["policy_seed"], 1, s["max_steps"])
                obs, initial = env.reset(ep)
                assert initial["initial_state_sha256"] == s["initial_state_sha256"]
                assert initial["initial_observation_sha256"] == s["initial_observation_sha256"]
                for row in t["steps"]:
                    assert observation_hash(obs.payload["raw"]) == row["observation_sha256"]
                    assert simulation_hash(env) == row["physics_state_sha256"]
                    obs, success, _ = env.step(row["action"])
                    assert not success or row["step"] == s["steps"]-1
                assert success
                destination.mkdir(parents=True, exist_ok=True)
                rows = []
                video = EpisodeVideo(destination / "video.mp4", ep, 20, label="post-success / open")
                try:
                    for step in range(40):
                        video(obs.frame, {"step": step, "sim_time_s": step/20, "replanned": False,
                              "model_calls": 0, "active_k": "-", "instruction": initial["instruction"]}, None)
                        obs, success, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                        assert len(obs.gripper) == 2, "This diagnostic is specified for Panda's two fingers"
                        rows.append({"step": step+1, "goal_success": success, "gripper_qpos": obs.gripper,
                                     "observation_sha256": observation_hash(obs.payload["raw"]),
                                     "physics_state_sha256": simulation_hash(env)})
                    stable = all(r["goal_success"] and all(abs(q)>=.035 for q in r["gripper_qpos"]) for r in rows[-10:])
                    video(obs.frame, {"step":40,"sim_time_s":2,"replanned":False,"model_calls":0,"active_k":"-"}, stable)
                finally:
                    video.close()
                write_jsonl(destination / "steps.jsonl", rows)
                atomic_json(destination / "summary.json", audit | {"replayed": True, "post_open_stable_success": stable,
                            "goal_true_on_last_10": all(r["goal_success"] for r in rows[-10:]),
                            "fingers_open_on_last_10": all(all(abs(q)>=.035 for q in r["gripper_qpos"]) for r in rows[-10:])})
                atomic_json(output / "status.json", {"phase":"running","completed":i+1,"total":len(trials)})
                print(f"[{i+1}/{len(trials)}] {s['method']} {s['suite']} s{s['state']} post_open_stable={stable}", flush=True)
            atomic_json(output / "status.json", {"phase":"completed","completed":len(trials),"total":len(trials)})
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run([args.run]+([args.extra] if args.extra else []), args.output or args.run / "release_checks")
