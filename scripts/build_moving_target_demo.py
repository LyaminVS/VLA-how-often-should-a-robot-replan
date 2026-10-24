"""Render a fixed, paired illustration and all 135 moving-target results.

The illustrated state (10) and policy seed (11) are selected before collection.
Videos advance together at simulated time; terminated panels hold their final
frame. The unchanged raw recordings and their outcome labels remain visible.
"""

import argparse
import json
import math
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.analyze_moving_target import DATA_KIND, load_run, sha256

FPS = 20
WIDTH, HEIGHT = 1536, 512
PANEL_WIDTH, PANEL_HEIGHT = 512, 384
HEADER_HEIGHT = FOOTER_HEIGHT = 64
STATE, POLICY_SEED = 10, 11
BACKGROUND = "#101a2c"
WHITE, MUTED, ACCENT = "#f2f6fc", "#a4b7cf", "#f6c550"


def inspect_video(path, size, expected_frames):
    import imageio.v2 as imageio

    with imageio.get_reader(str(path), input_params=["-threads", "1"]) as reader:
        metadata = reader.get_meta_data()
        if not math.isclose(metadata.get("fps", 0), FPS, abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"Expected {FPS} fps: {path}")
        if tuple(metadata.get("size", ())) != tuple(size):
            raise ValueError(f"Unexpected video dimensions: {path}")
        if reader.count_frames() != expected_frames:
            raise ValueError(f"Unexpected source video frame count: {path}")


def font_paths(regular=None, bold=None):
    if bool(regular) != bool(bold):
        raise ValueError("Provide both --font and --font-bold")
    candidates = [(regular, bold)] if regular else [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for normal, heavy in candidates:
        if Path(normal).is_file() and Path(heavy).is_file():
            return str(normal), str(heavy)
    raise FileNotFoundError("No matching regular/bold fonts found; use --font and --font-bold")


def load_inputs(input_path, report_path):
    source, report = Path(input_path).resolve(), Path(report_path).resolve()
    cfg, rows, provenance = load_run(source)
    base = cfg.base_config
    if (len(rows) != 135 or tuple(sorted(base.intervals)) != (5, 10, 50)
            or tuple(sorted(base.initial_states)) != (10, 11, 12, 13, 14)
            or tuple(sorted(base.policy_seeds)) != (11, 22, 33)
            or base.control_hz != FPS or not base.record_video):
        raise ValueError("This demo requires the complete registered 135-episode plan with recorded 20-Hz videos")
    statistics_path = report / "statistics.json"
    statistics = json.loads(statistics_path.read_text())
    if (statistics.get("data_kind") != DATA_KIND or statistics.get("complete") is not True
            or statistics.get("episodes") != 135
            or statistics.get("config") != json.loads(json.dumps(cfg.asdict()))):
        raise ValueError("Statistics do not describe this complete moving-target experiment")
    recorded = statistics.get("provenance", {})
    for field in ("manifest_sha256", "config_fingerprint", "runtime", "hardware", "sources", "episodes"):
        if recorded.get(field) != provenance.get(field):
            raise ValueError(f"Statistics provenance mismatch: {field}")
    if statistics.get("analysis_source_sha256") != sha256(PROJECT / "scripts/analyze_moving_target.py"):
        raise ValueError("Analysis source changed after the statistics were generated")
    lookup = {(row["state"], row["policy_seed"], row["condition"], row["k"]): row for row in rows}
    sections = [
        {"title": "Same K=10: changing target motion", "episodes": [lookup[(STATE, POLICY_SEED, condition, 10)] for condition in ("static", "slow", "fast")]},
        {"title": "Fast reference: changing replanning interval", "episodes": [lookup[(STATE, POLICY_SEED, "fast", k)] for k in (5, 10, 50)]},
    ]
    selected = {row["episode_id"]: row for section in sections for row in section["episodes"]}
    logs, summaries = {}, {}
    for identity, row in selected.items():
        directory = source / "episodes" / identity
        inspect_video(directory / "video.mp4", (768, 576), row["steps"] + FPS)
        logs[identity] = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        summaries[identity] = json.loads((directory / "summary.json").read_text())
    if len({row["pair_id"] for row in selected.values()}) != 1:
        raise ValueError("Illustrative episodes do not share the preselected state/seed pair")
    return cfg, rows, provenance, statistics_path, sections, logs, summaries


def build_demo(args):
    output = args.output.resolve()
    sidecar = output.with_suffix(".json")
    source = args.input.resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("The demo output must use the .mp4 extension")
    if output.is_relative_to(source):
        raise ValueError("Demo output must be outside the immutable raw run directory")
    if not args.overwrite and (output.exists() or sidecar.exists()):
        raise FileExistsError("Demo or sidecar exists; use --overwrite to replace derived outputs")
    cfg, rows, provenance, statistics_path, sections, logs, summaries = load_inputs(args.input, args.report)
    source_hashes = {record["episode_id"]: record for record in provenance["episodes"]}
    selected_ids = sorted(summaries)
    input_hashes = {
        statistics_path: sha256(statistics_path),
        source / "manifest.json": provenance["manifest_sha256"],
        Path(__file__).resolve(): sha256(__file__),
        PROJECT / "scripts/analyze_moving_target.py": sha256(PROJECT / "scripts/analyze_moving_target.py"),
    }
    for record in provenance["episodes"]:
        directory = source / "episodes" / record["episode_id"]
        input_hashes[directory / "summary.json"] = record["files"]["summary.json"]
        input_hashes[directory / "evidence.json"] = record["evidence_sha256"]
    for identity in selected_ids:
        input_hashes.update({source / "episodes" / identity / filename: checksum for filename, checksum in source_hashes[identity]["files"].items()})

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    regular_path, bold_path = font_paths(args.font, args.font_bold)
    fonts = {}

    def font(size, bold=False):
        key = size, bold
        if key not in fonts:
            fonts[key] = ImageFont.truetype(bold_path if bold else regular_path, size=size)
        return fonts[key]

    def text(draw, message, xy, size=28, color=WHITE, bold=False, max_width=None, anchor=None):
        selected_font = font(size, bold)
        if max_width is not None and draw.textlength(message, font=selected_font) > max_width:
            raise ValueError(f"Demo text exceeds its layout: {message}")
        draw.text(xy, message, font=selected_font, fill=color, anchor=anchor)

    def card(title, lines):
        image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        text(draw, title, (56, 50), 46, bold=True, max_width=WIDTH - 112)
        for index, line in enumerate(lines):
            text(draw, line, (56, 158 + 70 * index), 29, MUTED if index else ACCENT, max_width=WIDTH - 112)
        return np.asarray(image)

    introductory = card("Can replanning help catch a moving target?", [
        "SmolVLA / modified LIBERO | 135 episodes | K = 5, 10, 50",
        "Instruction: pick up the alphabet soup and place it in the basket.",
        "Illustration fixed in advance: state 10, policy seed 11.",
        "Simulation time shown; inference pauses omitted. Force stops at finger contact.",
    ])
    divider = card("Now compare three replanning intervals", [
        "Fast motion setting; same state 10 and policy seed 11.",
        "K=5: 4 updates/s     K=10: 2 updates/s     K=50: 0.4 updates/s",
        "Each panel runs to its outcome; completed panels hold their final frame.",
        "This preselected example illustrates behavior; all episodes inform the results.",
    ])

    closing = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(closing)
    text(draw, "All 135 episodes", (56, 26), 43, bold=True)
    text(draw, "Successful transfers / 15 episodes; mean model calls in parentheses", (56, 91), 25, MUTED)
    column_centers = [600, 952, 1304]
    for center, k in zip(column_centers, (5, 10, 50), strict=True):
        text(draw, f"K={k}", (center, 151), 29, ACCENT, True, anchor="mm")
    counts = []
    for index, condition in enumerate(("static", "slow", "fast")):
        y = 215 + 72 * index
        text(draw, condition.upper(), (56, y), 29, bold=True, anchor="lm")
        for center, k in zip(column_centers, (5, 10, 50), strict=True):
            group = [row for row in rows if row["condition"] == condition and row["k"] == k]
            successes = sum(row["success"] for row in group)
            calls = sum(row["model_calls"] for row in group) / len(group)
            text(draw, f"{successes}/{len(group)}  ({calls:.1f})", (center, y), 33, anchor="mm", max_width=330)
            counts.append({"condition": condition, "k": k, "successes": successes, "episodes": len(group), "mean_model_calls": calls})
    text(draw, "5 initial states x 3 policy seeds x 3 intervals x 3 conditions", (56, 418), 24, MUTED)
    text(draw, "One artificial-motion task; contact is not a verified grasp. See the report for uncertainty.", (56, 460), 23, MUTED, max_width=WIDTH - 112)
    closing = np.asarray(closing)

    def composite(section, frames, step):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        text(draw, section["title"] + " | Soup can to basket", (16, 4), 22, bold=True, max_width=1180)
        last_step = max(row["steps"] for row in section["episodes"])
        clock_label = f"Simulation t={step / FPS:.2f} s" if step < last_step else f"Final outcomes | t={last_step / FPS:.2f} s"
        text(draw, clock_label, (WIDTH - 16, 5), 20, MUTED, anchor="ra")
        for index, (row, frame) in enumerate(zip(section["episodes"], frames, strict=True)):
            left = index * PANEL_WIDTH
            canvas.paste(Image.fromarray(frame).resize((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS), (left, HEADER_HEIGHT))
            text(draw, f"{row['condition'].upper()} | K={row['k']}", (left + PANEL_WIDTH / 2, 46), 23, ACCENT, True, anchor="mm")
            identity = row["episode_id"]
            ended = step >= row["steps"]
            if ended:
                status = f"{'SUCCESS' if row['success'] else 'FAILURE'} | final frame held | {row['model_calls']} calls"
                color = "#71e4ac" if row["success"] else "#ff8d91"
                motion = summaries[identity]["motion_final"]
            else:
                logged = logs[identity][step]
                last_replan = step - step % row["k"]
                status = (f"REPLAN at step {step} | {logged['model_calls']} calls" if logged["replanned"]
                          else f"step {step} | last replan {last_replan} | {logged['model_calls']} calls")
                color = ACCENT if step - last_replan < 3 else WHITE
                motion = logged["motion"]
            contact_time = motion.get("first_contact_sim_time")
            contact = f"Finger contact: {contact_time:.2f} s; force off" if contact_time is not None else "No finger contact yet"
            text(draw, status, (left + 10, 450), 17, color, max_width=PANEL_WIDTH - 20)
            text(draw, contact, (left + 10, 480), 16, MUTED, max_width=PANEL_WIDTH - 20)
            if index:
                draw.line((left, HEADER_HEIGHT, left, HEIGHT), fill="#40536b", width=2)
        return np.asarray(canvas)

    output.parent.mkdir(parents=True, exist_ok=True)
    timeline = []
    frame_count = 0
    with tempfile.TemporaryDirectory(prefix=".moving-target-demo-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        staged = temporary / "demo.mp4"
        with imageio.get_writer(str(staged), fps=FPS, codec="libx264", quality=8, macro_block_size=16,
                                ffmpeg_log_level="error", ffmpeg_params=["-threads", "1", "-movflags", "+faststart"]) as writer:
            for _ in range(2 * FPS):
                writer.append_data(introductory)
                frame_count += 1
            timeline.append({"kind": "intro", "start_frame": 0, "frames": 2 * FPS})
            for section_index, section in enumerate(sections):
                if section_index:
                    timeline.append({"kind": "divider", "start_frame": frame_count, "frames": 2 * FPS})
                    for _ in range(2 * FPS):
                        writer.append_data(divider)
                        frame_count += 1
                start = frame_count
                duration_frames = max(row["steps"] for row in section["episodes"]) + FPS
                with ExitStack() as stack:
                    readers = [stack.enter_context(imageio.get_reader(str(source / "episodes" / row["episode_id"] / "video.mp4"), input_params=["-threads", "1"])) for row in section["episodes"]]
                    iterators, frames = [iter(reader) for reader in readers], [None] * 3
                    for step in range(duration_frames):
                        for index, row in enumerate(section["episodes"]):
                            if step <= row["steps"]:
                                try:
                                    frames[index] = next(iterators[index])
                                except StopIteration as exc:
                                    raise ValueError(f"Source video ended before its recorded terminal state: {row['episode_id']}") from exc
                                if frames[index].shape != (576, 768, 3):
                                    raise ValueError("Unexpected decoded source video dimensions")
                        writer.append_data(composite(section, frames, step))
                        frame_count += 1
                timeline.append({"kind": "comparison", "title": section["title"], "start_frame": start, "frames": duration_frames, "episode_ids": [row["episode_id"] for row in section["episodes"]]})
                print(f"Rendered {section['title']}: {duration_frames / FPS:.2f} seconds", flush=True)
            timeline.append({"kind": "results", "start_frame": frame_count, "frames": 4 * FPS})
            for _ in range(4 * FPS):
                writer.append_data(closing)
                frame_count += 1
        inspect_video(staged, (WIDTH, HEIGHT), frame_count)
        if any(sha256(path) != checksum for path, checksum in input_hashes.items()):
            raise RuntimeError("Demo inputs changed during rendering")
        evidence = {
            "schema_version": 1,
            "artifact_kind": "moving_target_demo",
            "data_kind": DATA_KIND,
            "episodes_in_results": len(rows),
            "config_fingerprint": cfg.fingerprint,
            "manifest_sha256": provenance["manifest_sha256"],
            "statistics_sha256": input_hashes[statistics_path],
            "generator_sha256": input_hashes[Path(__file__).resolve()],
            "analysis_source_sha256": input_hashes[PROJECT / "scripts/analyze_moving_target.py"],
            "video_sha256": sha256(staged),
            "selected_episodes": [source_hashes[identity] for identity in selected_ids],
            "illustration_selection": {"state": STATE, "policy_seed": POLICY_SEED, "rule": "Fixed before collection; no selection by outcome"},
            "full_results_table": counts,
            "timeline": timeline,
            "frames": frame_count,
            "fps": FPS,
            "size": [WIDTH, HEIGHT],
            "duration_s": frame_count / FPS,
            "audio": False,
            "visual_review_required": True,
            "note": "Aligned simulation time; model-inference pauses omitted; one-second terminal hold retained; shorter episodes hold their terminal frame.",
        }
        staged_sidecar = temporary / "demo.json"
        staged_sidecar.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if args.overwrite:
            staged.replace(output)
            staged_sidecar.replace(sidecar)
        else:
            output.hardlink_to(staged)
            sidecar.hardlink_to(staged_sidecar)
    print(json.dumps({"output": str(output), "frames": frame_count, "duration_s": frame_count / FPS, "size": [WIDTH, HEIGHT], "episodes_in_results": 135}, indent=2))
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/moving-target"))
    parser.add_argument("--report", type=Path, default=Path("outputs/moving-target-report"))
    parser.add_argument("--output", type=Path, default=Path("outputs/moving-target-report/demo.mp4"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-bold", type=Path)
    build_demo(parser.parse_args(argv))


if __name__ == "__main__":
    main()
