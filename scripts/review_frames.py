"""Create contact sheets from recorded robot episodes for human video review.

Frames are decoded sequentially. Captions report existing logs and task outcomes;
this script does not infer grasp errors, drops, or other review labels.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 3, 6
TILE_WIDTH, IMAGE_HEIGHT, HEADER_HEIGHT = 512, 256, 28
TILE_HEIGHT = IMAGE_HEIGHT + HEADER_HEIGHT
PAGE_SIZE = (COLS * TILE_WIDTH, ROWS * TILE_HEIGHT)
CROP = (0, 128, 768, 512)


def select_action_frames(rows, stride):
    if stride < 1:
        raise ValueError("stride must be positive")
    if not rows:
        raise ValueError("No action records to review")
    selected = set(range(0, len(rows), stride)) | {len(rows) - 1}
    transitions = []
    previous_sign = None
    for index, row in enumerate(rows):
        if row["step"] != index:
            raise ValueError("Action records must contain consecutive zero-based steps")
        grip = float(row["action"][6])
        if not math.isfinite(grip):
            raise ValueError(f"Nonfinite gripper command at step {index}")
        sign = (grip > 0) - (grip < 0)
        if previous_sign is not None and sign != previous_sign:
            transitions.append(index)
            selected.update(range(max(0, index - 2), min(len(rows), index + 3)))
        previous_sign = sign
    return sorted(selected), transitions


class ContactSheets:
    def __init__(self, directory, episode_id, expected_frames):
        self.directory = directory
        self.episode_id = episode_id
        self.page_count = math.ceil(expected_frames / (COLS * ROWS))
        self.pages = []
        self.records = []
        self.canvas = None
        self.font = ImageFont.load_default(size=14)
        self.top_font = ImageFont.load_default(size=12)

    def add(self, frame, frame_index, row, outcome=None):
        if frame.shape[:2] != (512, 768):
            raise ValueError(f"Expected a 768x512 episode video, got {frame.shape}")
        ordinal = len(self.records)
        slot = ordinal % (COLS * ROWS)
        page_number = ordinal // (COLS * ROWS) + 1
        if slot == 0:
            self.canvas = Image.new("RGB", PAGE_SIZE, "#101a2c")
            title = f"{self.episode_id} | page {page_number:02d}/{self.page_count:02d}"
            ImageDraw.Draw(self.canvas).text((6, 0), title, font=self.top_font, fill="white")
        column, tile_row = slot % COLS, slot // COLS
        x, y = column * TILE_WIDTH, tile_row * TILE_HEIGHT
        cameras = Image.fromarray(frame).convert("RGB").crop(CROP)
        cameras = cameras.resize((TILE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        self.canvas.paste(cameras, (x, y + HEADER_HEIGHT))
        grip = None if outcome is not None else float(row["action"][6])
        replan = None if outcome is not None else bool(row["replanned"])
        grip_text = "-" if grip is None else f"{grip:+.3f}"
        replan_text = "-" if replan is None else "yes" if replan else "no"
        caption = (
            f"{ordinal + 1:03d} step {row['step']} t {row['sim_time_s']:.2f}s "
            f"replan {replan_text} grip {grip_text}"
        )
        if outcome is not None:
            caption += f" | {outcome}"
        draw = ImageDraw.Draw(self.canvas)
        color = "#f6c550" if replan else "#d0dce9"
        # The top row shares its 28px header with the episode title; all camera tiles remain 512x256.
        draw.text((x + 5, y + (14 if tile_row == 0 else 6)), caption, font=self.top_font if tile_row == 0 else self.font, fill=color)
        draw.line((x, y + HEADER_HEIGHT - 1, x + TILE_WIDTH - 1, y + HEADER_HEIGHT - 1), fill="#536174")
        self.records.append(
            {
                "number": ordinal + 1,
                "frame_index": frame_index,
                "step": row["step"],
                "sim_time_s": row["sim_time_s"],
                "replanned": replan,
                "gripper_command": grip,
                "recorded_outcome": outcome,
                "page": f"sheet_{page_number:03d}.png",
                "row": tile_row,
                "column": column,
            }
        )
        if slot == COLS * ROWS - 1:
            self.flush()

    def flush(self):
        if self.canvas is not None:
            name = f"sheet_{len(self.pages) + 1:03d}.png"
            self.canvas.save(self.directory / name)
            self.pages.append(name)
            self.canvas = None


def review_episode(episode_dir, output, stride=5):
    episode_dir, output = Path(episode_dir).resolve(), Path(output).resolve()
    summary = json.loads((episode_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (episode_dir / "steps.jsonl").read_text().splitlines() if line.strip()]
    if summary["steps"] != len(rows):
        raise ValueError("Summary step count differs from the action log")
    selected, transitions = select_action_frames(rows, stride)
    selected_set = set(selected)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-", dir=output) as temporary:
        staging = Path(temporary)
        sheets = ContactSheets(staging, summary["episode_id"], len(selected) + 1)
        reader = imageio.get_reader(str(episode_dir / "video.mp4"))
        last_frame = None
        frame_index = -1
        try:
            fps = float(reader.get_meta_data()["fps"])
            for frame_index, frame in enumerate(reader):
                if frame_index in selected_set:
                    sheets.add(frame, frame_index, rows[frame_index])
                last_frame = frame
        finally:
            reader.close()
        # EpisodeVideo writes one frame per action and a one-second final outcome hold.
        expected_count = len(rows) + round(fps)
        decoded_count = frame_index + 1
        if last_frame is None or decoded_count != expected_count:
            raise ValueError(f"Video/log alignment failed: decoded {decoded_count} frames, expected {expected_count}")
        final_row = {"step": len(rows), "sim_time_s": summary["sim_time_s"]}
        outcome = "SUCCESS" if summary["success"] else f"END: {summary['termination']}"
        sheets.add(last_frame, frame_index, final_row, outcome=outcome)
        sheets.flush()
        index = {
            "episode_id": summary["episode_id"],
            "episode_dir": str(episode_dir),
            "video": str(episode_dir / "video.mp4"),
            "parameters": {
                "stride": stride,
                "gripper_transition_window_steps": 2,
                "grid": [COLS, ROWS],
                "page_size": list(PAGE_SIZE),
                "camera_crop": list(CROP),
                "camera_tile_size": [TILE_WIDTH, IMAGE_HEIGHT],
                "header_height": HEADER_HEIGHT,
                "selection": "stride over action frames, last action, sign-change neighborhoods, last video frame",
                "decode": "sequential",
            },
            "fps": fps,
            "decoded_video_frames": decoded_count,
            "action_steps": len(rows),
            "gripper_sign_change_steps": transitions,
            "selected_frame_indices": [record["frame_index"] for record in sheets.records],
            "pages": [str(output / name) for name in sheets.pages],
            "frames": sheets.records,
            "note": "Action frames precede their logged action; the final frame follows the last action. No review labels are inferred.",
        }
        (staging / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        for name in sheets.pages:
            (staging / name).replace(output / name)
        (staging / "index.json").replace(output / "index.json")
    print(json.dumps({"index": str(output / "index.json"), "pages": len(index["pages"]), "frames": len(index["frames"])}, indent=2))
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args(argv)
    review_episode(args.episode_dir, args.output, args.stride)


if __name__ == "__main__":
    main()
