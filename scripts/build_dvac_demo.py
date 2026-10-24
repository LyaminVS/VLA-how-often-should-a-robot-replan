#!/usr/bin/env python3
"""Show all three preselected fixed-10 / DVAC pairs, aligned by simulation step."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_replan.integrity import file_sha256
from vla_replan.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/dvac")
    parser.add_argument("--output", default="outputs/dvac-report")
    args = parser.parse_args()
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    root, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    base = manifest["config"]["base"]
    state, seed = base["initial_states"][0], base["policy_seeds"][0]
    summaries = [json.loads(path.read_text()) for path in root.glob("episodes/*/summary.json")]
    selected = []
    frames = 0
    fps = base["control_hz"]
    target = output / "demo.mp4"
    with imageio.get_writer(str(target), fps=fps, codec="libx264", quality=7,
                           ffmpeg_log_level="error", ffmpeg_params=["-threads", "1"]) as writer:
        for task in base["tasks"]:
            pair = []
            for method in ("fixed_10", "dvac"):
                match = [s for s in summaries if s["suite"] == task["suite"] and s["task"] == task["name"]
                         and s["state"] == state and s["policy_seed"] == seed and s["method"] == method]
                if len(match) != 1:
                    raise ValueError(f"Missing or ambiguous preselected demo pair: {task['name']}/{method}")
                pair.append(match[0])
            if any(pair[0][field] != pair[1][field] for field in (
                "pair_id", "initial_state_sha256", "initial_observation_sha256", "config_fingerprint",
            )):
                raise ValueError("Demo pairs differ in initial conditions")
            title = Image.new("RGB", (1536, 512), "#101a2c")
            draw = ImageDraw.Draw(title)
            font = ImageFont.load_default(size=28)
            small = ImageFont.load_default(size=20)
            import textwrap
            draw.text((48, 60), "Fixed K=10  |  DVAC", font=font, fill="white")
            for i, line in enumerate(textwrap.wrap(pair[0]["instruction"], width=94)):
                draw.text((48, 130 + 35 * i), line, font=font, fill="white")
            draw.text((48, 290), f"Matched initial state {state}, policy seed {seed}; simulation time at {fps} Hz",
                      font=small, fill="#97abc7")
            draw.text((48, 330), "DVAC chooses K from internal flow-matching variance. Yellow marks replanning.",
                      font=small, fill="#f6c550")
            draw.text((48, 370), "These three pairs were selected before examining outcomes.",
                      font=small, fill="#97abc7")
            for _ in range(fps):
                writer.append_data(np.asarray(title))
                frames += 1
            paths = [root / "episodes" / s["episode_id"] / "video.mp4" for s in pair]
            readers = [imageio.get_reader(str(path), input_params=["-threads", "1"]) for path in paths]
            start_frame = frames
            try:
                if any(abs(reader.get_meta_data()["fps"] - fps) > 1e-6 for reader in readers):
                    raise ValueError("Source video frame rate mismatch")
                iterators = [iter(reader) for reader in readers]
                ended = [False, False]
                last = [None, None]
                while True:
                    for i in range(2):
                        if not ended[i]:
                            try:
                                last[i] = next(iterators[i])
                            except StopIteration:
                                ended[i] = True
                    if all(ended):
                        break
                    if any(frame is None for frame in last):
                        raise ValueError("Empty source video")
                    writer.append_data(np.concatenate(last, axis=1))
                    frames += 1
            finally:
                for reader in readers:
                    reader.close()
            selected.append({
                "task": task["name"], "pair_id": pair[0]["pair_id"],
                "start_frame": start_frame, "end_frame_exclusive": frames,
                "episodes": [{"episode_id": s["episode_id"], "success": s["success"],
                              "model_calls": s["model_calls"], "video_sha256": file_sha256(path)}
                             for s, path in zip(pair, paths, strict=True)],
            })
    atomic_json(output / "demo.json", {
        "selection_rule": "First configured initial state and policy seed, every configured task, fixed_10 versus dvac.",
        "fps": fps, "frames": frames, "duration_s": frames / fps, "width": 1536, "height": 512,
        "sha256": file_sha256(target), "pairs": selected,
    })
    print(f"Saved {target}: {frames} frames, {frames / fps:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
