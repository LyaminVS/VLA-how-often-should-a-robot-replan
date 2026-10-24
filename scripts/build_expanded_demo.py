"""Build the English demo for the complete 150-episode study from saved recordings.

The introduction reports all 150 episodes, followed by reviewed, paired cases.
Original frames and REPLAN labels are retained at 20 fps. The simulator's inference
pauses are omitted from the videos.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.build_demo import (  # noqa: E402
    ACCENT,
    BACKGROUND,
    FPS,
    HEADER_HEIGHT,
    HEIGHT,
    MUTED,
    TIME_NOTE,
    WHITE,
    WIDTH,
    inspect_video,
    wrapped_lines,
)
from scripts.build_report import TASK_NAMES, font_paths  # noqa: E402
from scripts.package_submission import sha256_file  # noqa: E402
from vla_replan.video import compare_videos  # noqa: E402


def load_inputs(args):
    from scripts.build_expanded_figures import load_bundle, validate_expanded_discussion

    bundle = load_bundle(args.run, args.pilot, args.validation)
    validation = bundle["sources"]["validation"]
    cfg, rows = validation["config"], validation["rows"]
    if cfg.control_hz != FPS or not cfg.record_video:
        raise ValueError("Expanded demo requires recorded videos at 20 Hz")
    discussion = json.loads(args.discussion.read_text(encoding="utf-8"))
    validate_expanded_discussion(discussion, rows)
    by_id = {row["episode_id"]: row for row in rows}
    inspected = {label["episode_id"]: label.get("inspected") for label in bundle["labels"]["validation"]}
    cases = []
    for index, case in enumerate(discussion["cases"], 1):
        if len(case["episode_ids"]) < 2:
            raise ValueError(f"Case {index} must identify at least two paired episodes")
        selected = [by_id[identity] for identity in case["episode_ids"][:2]]
        if selected[0]["k"] == selected[1]["k"]:
            raise ValueError(f"Case {index} must compare different K values")
        for field in ("pair_id", "config_fingerprint", "initial_state_sha256", "initial_observation_sha256"):
            if selected[0][field] != selected[1][field]:
                raise ValueError(f"Case {index}: paired episodes differ in {field}")
        for row in selected:
            if row["state"] not in (2, 3):
                raise ValueError("These illustrative cases must come from reviewed states 2 and 3")
            if inspected.get(row["episode_id"]) is not True:
                raise ValueError("Demo case episodes require explicit visual-inspection evidence")
            inspect_video(
                validation["root"] / "episodes" / row["episode_id"] / "video.mp4",
                (768, 512), expected_frames=row["steps"] + FPS,
            )
        cases.append({"title": case["title"], "episodes": selected})
    return bundle, cfg, cases


def build_demo(args):
    output = args.output.resolve()
    evidence = output.with_suffix(".json")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have the .mp4 extension")
    if not args.overwrite and any(path.exists() for path in (output, evidence)):
        raise FileExistsError(f"Output or its evidence sidecar exists; use --overwrite: {output}")
    if args.discussion.resolve() in (output, evidence) or any(
        output.is_relative_to(path.resolve()) for path in (args.pilot, args.validation)
    ):
        raise ValueError("Expanded demo must not overwrite a source-phase artifact")
    bundle, cfg, cases = load_inputs(args)
    validation = bundle["sources"]["validation"]
    rows = bundle["rows"]["combined"]
    statistics_sha256 = sha256_file(bundle["statistics_path"])
    discussion_sha256 = sha256_file(args.discussion)
    script_sha256 = sha256_file(Path(__file__))
    source_videos = {
        row["episode_id"]: validation["root"] / "episodes" / row["episode_id"] / "video.mp4"
        for case in cases for row in case["episodes"]
    }
    source_hashes = {identity: sha256_file(path) for identity, path in source_videos.items()}

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
            raise ValueError("Card text is too long; shorten the case title")
        spacing = round(size * 1.3)
        for index, item in enumerate(lines):
            draw.text((64, y + index * spacing), item, font=selected_font, fill=color)
        return y + spacing * len(lines)

    def overview_card():
        card = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(card)
        line(draw, "How often should a robot replan?", 40, 46, bold=True)
        line(draw, "Results from all 150 episodes with the same model and matched starts.", 108, 28, color=MUTED)
        cell = (WIDTH - 128) / len(cfg.intervals)
        for index, k in enumerate(sorted(cfg.intervals)):
            group = [row for row in rows if row["k"] == k]
            center = 64 + (index + 0.5) * cell
            draw.text((center, 202), f"K={k}", font=font(38, True), anchor="mm", fill=MUTED)
            draw.text(
                (center, 280), f"{sum(row['success'] for row in group)}/{len(group)}",
                font=font(59, True), anchor="mm", fill=ACCENT,
            )
        line(draw, "Successful episodes / all 30 episodes at each K", 349, 28, color=MUTED)
        line(draw, "3 tasks × 10 states × 5 K values; policy seed 11", 409, 31, bold=True)
        paragraph(draw, "30 matched task-state situations, one model and one seed.",
                  478, 29, max_lines=2)
        return np.asarray(card)

    def title_card(case, index):
        card = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(card)
        left, right = case["episodes"]
        line(draw, f"PAIRED CASE {index} / {len(cases)}", 40, 25, color=ACCENT)
        title_end = paragraph(draw, case["title"], 108, 46, bold=True, max_lines=2)
        task = TASK_NAMES.get(left["task"], left["task"].replace("_", " "))
        paragraph(draw, task, max(265, title_end + 30), 32, max_lines=2)
        line(draw, f"Same initial conditions: state={left['state']}, seed={left['policy_seed']}",
             366, 29, color=MUTED)
        line(draw, f"Left: K={left['k']}     |     Right: K={right['k']}", 419, 35, bold=True)
        paragraph(draw, TIME_NOTE, 524, 25, color=MUTED, max_lines=2)
        return np.asarray(card)

    def header(case, index):
        bar = Image.new("RGB", (WIDTH, HEADER_HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(bar)
        text = f"Case {index}/{len(cases)}. {case['title']}"
        size = 30
        while size > 19 and draw.textlength(text, font=font(size, True)) > WIDTH - 48:
            size -= 1
        line(draw, text, 7, size, bold=True, x=24, max_width=WIDTH - 48)
        line(draw, TIME_NOTE, 53, 23, color=MUTED, x=24, max_width=WIDTH - 48)
        draw.rectangle((0, HEADER_HEIGHT - 3, WIDTH - 1, HEADER_HEIGHT - 1), fill=ACCENT)
        return np.asarray(bar)

    def closing_card():
        card = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(card)
        line(draw, "Replanning trades computation for feedback", 42, 48, bold=True)
        summary = []
        for k in (1, 5, 10):
            selected = [row for row in rows if row['k'] == k]
            summary.append(f"K={k}: {sum(row['success'] for row in selected)}/{len(selected)}")
        line(draw, "    |    ".join(summary), 139, 36, bold=True)
        line(draw, "The most frequent updates did not give the highest success count.",
             206, 30, color=MUTED)
        line(draw, "No universal best K follows from three tasks and one seed.", 277, 31, color=ACCENT)
        paragraph(draw, "LIBERO success may precede object release. Visible corrections do not establish causality.",
                  352, 31, max_lines=2)
        line(draw, "Report: output/pdf/vla-replanning-report.pdf", 490, 27, color=MUTED)
        line(draw, "Cases illustrate reviewed states 2 and 3; they do not replace the full results.",
             548, 25, color=MUTED)
        return np.asarray(card)

    introduction = overview_card()
    cards = [title_card(case, index) for index, case in enumerate(cases, 1)]
    headers = [header(case, index) for index, case in enumerate(cases, 1)]
    ending = closing_card()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    with tempfile.TemporaryDirectory(prefix=".expanded-demo-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        staged = temporary / "demo.mp4"
        with imageio.get_writer(
            str(staged), fps=FPS, codec="libx264", quality=8, macro_block_size=16,
            ffmpeg_log_level="error", output_params=["-movflags", "+faststart"],
        ) as writer:
            for _ in range(4 * FPS):
                writer.append_data(introduction)
                frame_count += 1
            for index, (case, card, bar) in enumerate(zip(cases, cards, headers, strict=True), 1):
                left, right = case["episodes"]
                comparison = temporary / f"case-{index}.mp4"
                compare_videos(
                    validation["root"] / "episodes" / left["episode_id"],
                    validation["root"] / "episodes" / right["episode_id"], comparison,
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
                print(f"Added paired case {index}: K={left['k']} vs K={right['k']}", flush=True)
            for _ in range(3 * FPS):
                writer.append_data(ending)
                frame_count += 1
        inspect_video(staged, (WIDTH, HEIGHT), expected_frames=frame_count)
        inputs = {bundle["statistics_path"]: statistics_sha256, args.discussion: discussion_sha256,
                  Path(__file__): script_sha256}
        inputs.update({path: source_hashes[identity] for identity, path in source_videos.items()})
        if any(sha256_file(path) != digest for path, digest in inputs.items()):
            raise RuntimeError("Demo inputs changed during rendering")
        staged_evidence = temporary / "demo.json"
        staged_evidence.write_text(json.dumps({
            "schema_version": 1, "artifact_kind": "combined_demo", "primary_phase": "combined",
            "statistics_sha256": statistics_sha256, "discussion_sha256": discussion_sha256,
            "build_script_sha256": script_sha256, "video_sha256": sha256_file(staged),
            "source_video_sha256": source_hashes,
            "case_episode_ids": [[row["episode_id"] for row in case["episodes"]] for case in cases],
            "frames": frame_count, "fps": FPS, "size": [WIDTH, HEIGHT],
            "note": "Rendered at simulation rate; model-inference pauses are omitted. Visual QA is separate.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.overwrite:
            staged.replace(output)
            staged_evidence.replace(evidence)
        else:
            output.hardlink_to(staged)
            evidence.hardlink_to(staged_evidence)
    print(json.dumps({
        "output": str(output), "primary_phase": "combined", "episodes": len(rows),
        "cases": len(cases), "frames": frame_count,
        "duration_s": frame_count / FPS, "fps": FPS, "size": [WIDTH, HEIGHT],
        "audio": False, "visual_review_required": True,
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--pilot", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--discussion", type=Path, default=Path("outputs/expanded/discussion.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expanded/demo.mp4"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--font", type=Path, help="Regular TTF font")
    parser.add_argument("--font-bold", type=Path, help="Matching bold TTF font")
    build_demo(parser.parse_args(argv))


if __name__ == "__main__":
    main()
