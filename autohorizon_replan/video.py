"""Stream action-aligned frames with the selected execution horizon."""

import textwrap


class EpisodeVideo:
    def __init__(self, path, episode, fps):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=7, macro_block_size=16,
            ffmpeg_log_level="error", ffmpeg_params=["-threads", "1"],
        )
        self.episode = episode
        self.fps = fps
        self.instruction = episode.task.replace("_", " ")
        self.last_replan = 0
        self.closed = False

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
        method = f"AutoHorizon cap={self.episode.execution_cap}"
        draw.text((16, 76), (
            f"{method} | K={row['chosen_k']} | step {row['step']} | "
            f"t={row['sim_time_s']:.2f}s | calls={row['model_calls']}"
        ), font=font, fill="white")
        status = "REPLAN" if row["replanned"] else f"Executing | last replan: step {self.last_replan}"
        if row.get("raw_chosen_k") is not None:
            status += f" | raw K={row['raw_chosen_k']}"
        if outcome is not None:
            status = "SUCCESS" if outcome else "FAILURE / EPISODE ENDED"
            color = "#71e4ac" if outcome else "#ff8d91"
        draw.text((16, 102), status, font=small, fill=color)
        draw.text((568, 102), f"SIMULATION / {self.fps} Hz", font=small, fill="#97abc7")
        draw.rectangle((0, 124, 767, 127), fill=color)
        encoded = np.asarray(canvas)
        for _ in range(self.fps if outcome is not None else 1):
            self.writer.append_data(encoded)

    def close(self):
        if not self.closed:
            self.writer.close()
            self.closed = True
