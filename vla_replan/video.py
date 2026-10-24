"""Streaming videos: no full episode is kept in RAM."""

import json
import textwrap
from pathlib import Path


class EpisodeVideo:
    def __init__(self, path, episode, fps, label=None):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=7, macro_block_size=16, ffmpeg_log_level="error"
        )
        self.episode = episode
        self.label = label
        self.fps = fps
        self.instruction = episode.task.replace("_", " ")
        self.last_replan = 0

    def __call__(self, frame, row, outcome):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        if frame is None:
            raise ValueError("Recording requested but simulator returned no frame")
        if "instruction" in row:
            self.instruction = row["instruction"]
        if row["replanned"]:
            self.last_replan = row["step"]
        canvas = Image.new("RGB", (768, 512), "#101a2c")
        canvas.paste(Image.fromarray(frame).resize((768, 384)), (0, 128))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=16)
        small = ImageFont.load_default(size=13)
        color = "#f6c550" if row["replanned"] else "#97abc7"
        for i, line in enumerate(textwrap.wrap(self.instruction, width=84)[:3]):
            draw.text((16, 10 + i * 20), line, font=font, fill="white")
        status = "REPLAN" if row["replanned"] else f"Executing | last replan: step {self.last_replan}"
        if outcome is not None:
            status = "SUCCESS" if outcome else "FAILURE / EPISODE ENDED"
            color = "#71e4ac" if outcome else "#ff8d91"
        execution = f"{self.label} | K={row.get('active_k', '?')}" if self.label else f"K={self.episode.k}"
        draw.text(
            (16, 76),
            f"{execution} | step {row['step']} | t={row['sim_time_s']:.2f}s | calls={row['model_calls']}",
            font=font,
            fill="white",
        )
        draw.text((16, 102), status, font=small, fill=color)
        draw.text((540, 102), "SIMULATION TIME / 20 Hz", font=small, fill="#97abc7")
        draw.rectangle((0, 124, 767, 127), fill=color)
        encoded = np.asarray(canvas)
        for _ in range(self.fps if outcome is not None else 1):
            self.writer.append_data(encoded)

    def close(self):
        self.writer.close()


def compare_videos(left, right, output):
    import imageio.v2 as imageio
    import numpy as np

    left, right, output = Path(left), Path(right), Path(output)
    summaries = [json.loads((p / "summary.json").read_text()) for p in (left, right)]
    if summaries[0]["pair_id"] != summaries[1]["pair_id"]:
        raise ValueError("Compare the same task, initial state and policy seed")
    for field in ("config_fingerprint", "initial_state_sha256", "initial_observation_sha256"):
        if summaries[0][field] != summaries[1][field]:
            raise ValueError(f"Cannot compare runs with differing {field}")
    if output.exists():
        raise FileExistsError(output)
    readers = [imageio.get_reader(str(p / "video.mp4")) for p in (left, right)]
    try:
        fps = [r.get_meta_data()["fps"] for r in readers]
        if fps[0] != fps[1]:
            raise ValueError("Video frame rates differ")
        output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(output), fps=fps[0], codec="libx264", quality=7, ffmpeg_log_level="error"
        ) as writer:
            iterators = [iter(r) for r in readers]
            ended = [False, False]
            frames = [None, None]
            while True:
                for i in range(2):
                    if not ended[i]:
                        try:
                            frames[i] = next(iterators[i])
                        except StopIteration:
                            ended[i] = True
                if all(ended):
                    break
                if any(frame is None for frame in frames):
                    raise ValueError("Empty input video")
                writer.append_data(np.concatenate(frames, axis=1))
    finally:
        for reader in readers:
            reader.close()
