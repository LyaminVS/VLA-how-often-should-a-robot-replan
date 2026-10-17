"""Build one silent demo from paired cases in a complete real compact experiment.

Example:
    .venv/bin/python scripts/build_demo.py --run outputs/compact \
        --discussion outputs/compact/discussion.json --output outputs/compact/demo.mp4

Each case uses its first two episode_ids, in that order. All source frames are retained
at 20 fps. There is a two-second title before each pair and a three-second results card.
The 1536 x 512 comparison remains intact below a 96-pixel header. No model is run.
"""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_report import TASK_NAMES, font_paths, validate_discussion  # noqa: E402
from vla_replan.analysis import load_results  # noqa: E402
from vla_replan.video import compare_videos  # noqa: E402

WIDTH = 1536
HEIGHT = 608
HEADER_HEIGHT = 96
FPS = 20
BACKGROUND = "#101A2C"
WHITE = "#F3F7FB"
MUTED = "#A9BBD1"
ACCENT = "#71E4AC"
TIME_NOTE = "Video shows simulation time at 20 Hz. Model computation pauses are omitted."


def inspect_video(path, expected_size, expected_frames=None):
    import imageio.v2 as imageio

    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"Missing or empty video: {path}")
    with imageio.get_reader(str(path)) as reader:
        metadata = reader.get_meta_data()
        if not math.isclose(metadata.get("fps", 0), FPS, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Expected {FPS} fps: {path}")
        if tuple(metadata.get("size", ())) != expected_size:
            raise ValueError(f"Expected video size {expected_size}: {path}")
        frames = reader.count_frames()
    if frames < 1 or (expected_frames is not None and frames != expected_frames):
        raise ValueError(f"Unexpected frame count: {path}: {frames}, expected {expected_frames}")
    return frames


def load_inputs(root, discussion_path):
    cfg, rows, coverage = load_results(root, allow_incomplete=False)
    if (
        cfg.name != "compact"
        or coverage["completed_episodes"] != 30
        or len(cfg.tasks) != 3
        or len(cfg.initial_states) != 2
        or len(cfg.policy_seeds) != 1
        or tuple(sorted(cfg.intervals)) != (1, 5, 10, 25, 50)
        or cfg.control_hz != FPS
        or not cfg.record_video
    ):
        raise ValueError("Demo requires the complete compact30 protocol with videos at 20 Hz")
    discussion = json.loads(discussion_path.read_text(encoding="utf-8"))
    validate_discussion(discussion, rows)
    by_id = {row["episode_id"]: row for row in rows}
    cases = []
    for index, case in enumerate(discussion["cases"], 1):
        if len(case["episode_ids"]) < 2:
            raise ValueError(f"Case {index} needs at least two episode_ids for comparison")
        selected = [by_id[identity] for identity in case["episode_ids"][:2]]
        if selected[0]["k"] == selected[1]["k"]:
            raise ValueError(f"Case {index} must compare different K values")
        for field in ("pair_id", "config_fingerprint", "initial_state_sha256", "initial_observation_sha256"):
            if selected[0][field] != selected[1][field]:
                raise ValueError(f"Case {index}: comparison differs in {field}")
        for row in selected:
            video = root / "episodes" / row["episode_id"] / "video.mp4"
            # EpisodeVideo appends one second of outcome frames to its control steps.
            inspect_video(video, (768, 512), expected_frames=row["steps"] + FPS)
        cases.append({"title": case["title"], "episodes": selected})
    return cfg, rows, cases


def wrapped_lines(draw, text, font, width):
    """Wrap by measured pixels, refusing to silently truncate a long word."""
    lines, line = [], ""
    for word in text.split():
        if draw.textlength(word, font=font) > width:
            raise ValueError(f"Word does not fit in title card: {word}")
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def build_demo(args):
    root = args.run.resolve()
    output = args.output.resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have the .mp4 extension")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {output}")
    if output == args.discussion.resolve() or root / "episodes" in output.parents:
        raise ValueError("Refusing to overwrite discussion or an original episode")
    cfg, rows, cases = load_inputs(root, args.discussion)

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    regular_path, bold_path = font_paths(args.font, args.font_bold)

    def font(size, bold=False):
        return ImageFont.truetype(bold_path if bold else regular_path, size=size)

    def line(draw, text, y, size, *, bold=False, color=WHITE, x=64, max_width=WIDTH - 128):
        selected_font = font(size, bold)
        if draw.textlength(text, font=selected_font) > max_width:
            raise ValueError(f"Text does not fit at {size}px: {text}")
        draw.text((x, y), text, font=selected_font, fill=color)

    def paragraph(draw, text, y, size, *, bold=False, color=WHITE, max_lines=3):
        selected_font = font(size, bold)
        lines = wrapped_lines(draw, text, selected_font, WIDTH - 128)
        if len(lines) > max_lines:
            raise ValueError("Title card text is too long; shorten the case title")
        spacing = round(size * 1.3)
        for index, item in enumerate(lines):
            draw.text((64, y + index * spacing), item, font=selected_font, fill=color)
        return y + spacing * len(lines)

    def title_card(case, index):
        card = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(card)
        left, right = case["episodes"]
        line(draw, f"SMOLVLA / LIBERO   ·   PAIRED CASE {index} / {len(cases)}", 40, 25, color=ACCENT)
        title_end = paragraph(draw, case["title"], 108, 46, bold=True, max_lines=2)
        task = TASK_NAMES.get(left["task"], left["task"].replace("_", " "))
        paragraph(draw, task, max(265, title_end + 30), 32, max_lines=2)
        line(
            draw, f"Same initial conditions: state={left['state']}, seed={left['policy_seed']}", 366, 29, color=MUTED
        )
        line(draw, f"Left: K={left['k']}     |     Right: K={right['k']}", 419, 35, bold=True)
        paragraph(draw, TIME_NOTE, 524, 25, color=MUTED, max_lines=2)
        return np.asarray(card)

    def header(case, index):
        bar = Image.new("RGB", (WIDTH, HEADER_HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(bar)
        text = f"{index}/{len(cases)}. {case['title']}"
        selected_size = 30
        while selected_size > 19 and draw.textlength(text, font=font(selected_size, True)) > WIDTH - 48:
            selected_size -= 1
        line(draw, text, 7, selected_size, bold=True, x=24, max_width=WIDTH - 48)
        line(draw, TIME_NOTE, 53, 23, color=MUTED, x=24, max_width=WIDTH - 48)
        draw.rectangle((0, HEADER_HEIGHT - 3, WIDTH - 1, HEADER_HEIGHT - 1), fill=ACCENT)
        return np.asarray(bar)

    def results_card():
        card = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(card)
        line(draw, "Results across all 30 episodes", 43, 48, bold=True)
        line(draw, "Successful episodes / all episodes at each K", 112, 30, color=MUTED)
        cell = (WIDTH - 128) / len(cfg.intervals)
        for index, k in enumerate(sorted(cfg.intervals)):
            group = [row for row in rows if row["k"] == k]
            center = 64 + (index + 0.5) * cell
            draw.text((center, 205), f"K={k}", font=font(40, True), anchor="mm", fill=MUTED)
            draw.text(
                (center, 283),
                f"{sum(r['success'] for r in group)}/{len(group)}",
                font=font(62, True),
                anchor="mm",
                fill=ACCENT,
            )
        paragraph(
            draw,
            "6 initial situations, one model; results do not generalize universally.",
            380,
            35,
            bold=True,
            max_lines=2,
        )
        line(draw, "Report: output/pdf/vla-replanning-report.pdf", 490, 28, color=MUTED)
        line(draw, f"3 tasks × 2 states × 5 K values; seed={cfg.policy_seeds[0]}", 548, 25, color=MUTED)
        return np.asarray(card)

    # Render static cards before any output file is created, so bad text cannot produce a partial demo.
    cards = [title_card(case, index) for index, case in enumerate(cases, 1)]
    headers = [header(case, index) for index, case in enumerate(cases, 1)]
    ending = results_card()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    with tempfile.TemporaryDirectory(prefix=".demo-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        staged = temporary / "demo.mp4"
        with imageio.get_writer(
            str(staged),
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=16,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        ) as writer:
            for index, (case, card, bar) in enumerate(zip(cases, cards, headers, strict=True), 1):
                left, right = case["episodes"]
                comparison = temporary / f"case-{index}.mp4"
                compare_videos(
                    root / "episodes" / left["episode_id"],
                    root / "episodes" / right["episode_id"],
                    comparison,
                )
                expected_frames = max(left["steps"], right["steps"]) + FPS
                inspect_video(comparison, (1536, 512), expected_frames=expected_frames)
                for _ in range(2 * FPS):
                    writer.append_data(card)
                    frame_count += 1
                with imageio.get_reader(str(comparison)) as reader:
                    for frame in reader:
                        if frame.shape != (512, WIDTH, 3):
                            raise ValueError("Unexpected comparison frame dimensions")
                        canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
                        canvas[:HEADER_HEIGHT] = bar
                        canvas[HEADER_HEIGHT:] = frame
                        writer.append_data(canvas)
                        frame_count += 1
                print(f"Added case {index}: K={left['k']} vs K={right['k']}", flush=True)
            for _ in range(3 * FPS):
                writer.append_data(ending)
                frame_count += 1
        inspect_video(staged, (WIDTH, HEIGHT), expected_frames=frame_count)
        if args.overwrite:
            staged.replace(output)
        else:
            # Atomic no-clobber publication, including a concurrently created destination.
            output.hardlink_to(staged)
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(cases),
                "frames": frame_count,
                "duration_s": frame_count / FPS,
                "fps": FPS,
                "size": [WIDTH, HEIGHT],
                "audio": False,
                "visual_review_required": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--discussion", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/compact/demo.mp4"))
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace an existing demo")
    parser.add_argument("--font", type=Path, help="Regular TTF font")
    parser.add_argument("--font-bold", type=Path, help="Matching bold TTF font")
    build_demo(parser.parse_args(argv))


if __name__ == "__main__":
    main()
