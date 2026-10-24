"""Render every predetermined state-0/seed-11 fixed-10/cap-10/cap-50 comparison."""

import argparse
import json
import math
import sys
import tempfile
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.analyze_capped import label, sha256
from scripts.verify_dvac import read_jsonl
from vla_replan.storage import atomic_json

METHODS = ("fixed_10", "dvac_cap_10", "dvac_cap_50")
STATE, SEED = 0, 11
PANEL_WIDTH, PANEL_HEIGHT = 768, 512
WIDTH, HEADER, FOOTER = 2304, 128, 64
HEIGHT = HEADER + PANEL_HEIGHT + FOOTER
BACKGROUND, WHITE, MUTED, ACCENT = "#101a2c", "#f2f6fc", "#a4b7cf", "#f6c550"


def record_directory(record):
    root = Path(record["raw_source_root"]).resolve()
    directory = (root / "episodes" / record["summary"]["episode_id"]).resolve()
    if not directory.is_relative_to(root / "episodes"):
        raise ValueError("Episode path escapes its raw source")
    return directory


def verified_inputs(raw, report):
    from scripts.verify_capped import load_and_verify

    raw, report = Path(raw).resolve(), Path(report).resolve()
    cfg, records, current = load_and_verify(raw, decode_videos=False)
    if (len(records) != 270 or current.get("complete") is not True
            or current.get("fixed_episodes") != 150 or current.get("capped_episodes") != 120):
        raise ValueError("Publishing requires all 270 verified matched episodes")
    statistics_path, verification_path = report / "analysis.json", report / "verification.json"
    statistics, stored = json.loads(statistics_path.read_text()), json.loads(verification_path.read_text())
    artifact_path = report / "artifacts.json"
    artifacts = json.loads(artifact_path.read_text())
    if (statistics.get("complete") is not True or statistics.get("episodes") != 270
            or statistics.get("config_fingerprint") != cfg.fingerprint
            or artifacts.get("config_fingerprint") != cfg.fingerprint
            or artifacts.get("statistics_sha256") != sha256(statistics_path)
            or artifacts.get("files", {}).get("analysis.json") != sha256(statistics_path)
            or artifacts.get("files", {}).get("verification.json") != sha256(verification_path)
            or statistics.get("analysis_source_sha256") != sha256(PROJECT / "scripts/analyze_capped.py")):
        raise ValueError("Analysis artifacts do not match the complete verified comparison")
    for key in ("complete", "episodes", "fixed_episodes", "capped_episodes", "manifest_sha256", "config_fingerprint",
                "invariance_sha256", "verifier_source_sha256", "verification_dependency_sha256",
                "first_prediction_equality_all_methods", "max_variance_error"):
        if stored.get(key) != current.get(key):
            raise ValueError(f"Stored verification no longer matches current evidence: {key}")
    if stored.get("videos_decoded") is not True:
        raise ValueError("Run analyze_capped.py --decode-videos before publishing a demo or review")
    observed = {row["method"]: row for row in statistics["methods"]}
    methods = {record["summary"]["method"] for record in records}
    if set(observed) != methods:
        raise ValueError("Reported methods differ from verified evidence")
    for method in methods:
        group = [record["summary"] for record in records if record["summary"]["method"] == method]
        if (observed[method]["episodes"] != len(group)
                or observed[method]["successes"] != sum(row["success"] for row in group)
                or not math.isclose(observed[method]["model_calls_mean"],
                                    sum(row["model_calls"] for row in group) / len(group), rel_tol=1e-12)):
            raise ValueError("Reported outcomes differ from verified evidence")
    binding = {str(path): sha256(path) for path in (raw / "manifest.json", statistics_path, verification_path, artifact_path)}
    return cfg, records, current, binding


def predetermined_sections(cfg, records):
    sections = []
    for task in cfg.base.tasks:
        group = []
        for method in METHODS:
            matches = [record for record in records if record["summary"]["suite"] == task.suite
                       and record["summary"]["task"] == task.name and record["summary"]["state"] == STATE
                       and record["summary"]["policy_seed"] == SEED and record["summary"]["method"] == method]
            if len(matches) != 1:
                raise ValueError(f"Missing predetermined example: {task.name}/{method}")
            group.append(matches[0])
        for field in ("pair_id", "initial_state_sha256", "initial_observation_sha256", "instruction"):
            if len({record["summary"][field] for record in group}) != 1:
                raise ValueError(f"Predetermined examples disagree on {field}")
        sections.append(group)
    return sections


def protected_output(output, records):
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(record["raw_source_root"]).resolve()) for record in records):
        raise ValueError("Derived media must not overwrite immutable raw evidence")


@lru_cache(maxsize=24)
def font(size, bold=False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)


def wrap_text(draw, value, selected_font, width):
    lines, current = [], ""
    for word in value.split():
        candidate = (current + " " + word).strip()
        if draw.textlength(candidate, font=selected_font) <= width:
            current = candidate
        elif current:
            lines.append(current)
            current = word
        else:
            raise ValueError("An instruction word exceeds the available layout width")
    if current:
        lines.append(current)
    if any(draw.textlength(line, font=selected_font) > width for line in lines):
        raise ValueError("Instruction text exceeds the available layout width")
    return lines


def input_hashes(records):
    hashes = {}
    for record in records:
        directory = record_directory(record)
        hashes.update({str(directory / name): sha256(directory / name)
                       for name in ("summary.json", "steps.jsonl", "chunks.jsonl", "video.mp4")})
    return hashes


def check_unchanged(hashes):
    for path, checksum in hashes.items():
        if sha256(path) != checksum:
            raise ValueError(f"Source evidence changed while rendering: {path}")


def build_demo(raw, report, output, *, overwrite=False):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    cfg, records, verification, binding = verified_inputs(raw, report)
    sections = predetermined_sections(cfg, records)
    output = Path(output).resolve()
    protected_output(output, records)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Demo output must be an MP4 file")
    sidecar = output.with_suffix(".json")
    if not overwrite and (output.exists() or sidecar.exists()):
        raise FileExistsError("Derived demo exists; use --overwrite to replace it")
    fps = cfg.base.control_hz
    selected = [record for section in sections for record in section]
    hashes = {**binding, **input_hashes(selected), str(Path(__file__).resolve()): sha256(__file__)}
    logs = {record["summary"]["episode_id"]: read_jsonl(record_directory(record) / "steps.jsonl") for record in selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    timeline, total_frames = [], 0

    def title_card(section):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((64, 70), "Fixed K=10  |  DVAC cap 10  |  DVAC cap 50", font=font(44, True), fill=WHITE)
        instruction = section[0]["summary"]["instruction"]
        for index, line in enumerate(wrap_text(draw, instruction, font(34), WIDTH - 128)):
            draw.text((64, 180 + index * 48), line, font=font(34), fill=WHITE)
        lines = [f"Same initial state {STATE}, policy seed {SEED}; all three task examples fixed in advance.",
                 "The generated action horizon remains H=50. Adaptive caps limit execution before a new observation.",
                 "Panels advance together in simulation time. Ended episodes hold their final frame.",
                 "Yellow marks replanning. Inference pauses are omitted; this is not a wall-clock speed comparison."]
        for index, line in enumerate(lines):
            draw.text((64, 390 + index * 58), line, font=font(25), fill=ACCENT if index == 3 else MUTED)
        return np.asarray(canvas)

    def composite(section, frames, step):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 8), "SmolVLA: fixed and adaptive replanning", font=font(28, True), fill=WHITE)
        lines = wrap_text(draw, section[0]["summary"]["instruction"], font(25), WIDTH - 48)
        if len(lines) > 2:
            raise ValueError("Instruction exceeds the two-line demo header")
        for index, line in enumerate(lines):
            draw.text((24, 45 + index * 28), line, font=font(25), fill=WHITE)
        maximum = max(record["summary"]["steps"] for record in section)
        metadata = (f"State {STATE} | seed {SEED} | shared simulation t={min(step, maximum) / fps:.2f} s | "
                    "yellow = REPLAN | ended panels held | inference pauses omitted")
        draw.text((24, 106), metadata, font=font(16), fill=MUTED)
        for index, (record, frame) in enumerate(zip(section, frames, strict=True)):
            row = record["summary"]
            left = index * PANEL_WIDTH
            canvas.paste(Image.fromarray(frame), (left, HEADER))
            draw.text((left + 16, HEADER + PANEL_HEIGHT + 4), label(row["method"]), font=font(21, True), fill=WHITE)
            if step >= row["steps"]:
                message = f"{'SUCCESS' if row['success'] else 'FAILURE'} | {row['model_calls']} calls | final frame held"
                color = "#71e4ac" if row["success"] else "#ff8d91"
            else:
                logged = logs[row["episode_id"]][step]
                chosen = logged.get("chosen_k", row["k"])
                message = f"{'REPLAN' if logged['replanned'] else 'Executing'} | step {step} | K={chosen} | calls={logged['model_calls']}"
                color = ACCENT if logged["replanned"] else MUTED
            draw.text((left + 16, HEADER + PANEL_HEIGHT + 34), message, font=font(18), fill=color)
            if index:
                draw.line((left, HEADER, left, HEIGHT), fill="#6e7f95", width=2)
        return np.asarray(canvas)

    with tempfile.TemporaryDirectory(prefix=".capped-demo-", dir=output.parent) as temporary:
        staged = Path(temporary) / "demo.mp4"
        with imageio.get_writer(str(staged), fps=fps, codec="libx264", quality=7, macro_block_size=16,
                               ffmpeg_log_level="error", ffmpeg_params=["-threads", "1", "-movflags", "+faststart"]) as writer:
            for section in sections:
                start = total_frames
                card = title_card(section)
                for _ in range(2 * fps):
                    writer.append_data(card)
                    total_frames += 1
                comparisons_start = total_frames
                duration = max(record["summary"]["steps"] for record in section) + fps
                decoded = [0] * len(section)
                with ExitStack() as stack:
                    readers = [stack.enter_context(imageio.get_reader(str(record_directory(record) / "video.mp4"),
                                                                      input_params=["-threads", "1"])) for record in section]
                    for reader in readers:
                        metadata = reader.get_meta_data()
                        if not math.isclose(metadata["fps"], fps, abs_tol=1e-6) or tuple(metadata["size"]) != (PANEL_WIDTH, PANEL_HEIGHT):
                            raise ValueError("Selected source video has incorrect size or frame rate")
                    iterators, frames = [iter(reader) for reader in readers], [None] * len(section)
                    for step in range(duration):
                        for index, record in enumerate(section):
                            if step <= record["summary"]["steps"]:
                                try:
                                    frames[index] = next(iterators[index])
                                except StopIteration as exc:
                                    raise ValueError("Selected source ended before its terminal observation") from exc
                                decoded[index] += 1
                                if frames[index].shape != (PANEL_HEIGHT, PANEL_WIDTH, 3):
                                    raise ValueError("Unexpected selected video frame shape")
                        writer.append_data(composite(section, frames, step))
                        total_frames += 1
                    for index, iterator in enumerate(iterators):
                        for frame in iterator:
                            if frame.shape != (PANEL_HEIGHT, PANEL_WIDTH, 3):
                                raise ValueError("Unexpected selected video frame shape")
                            decoded[index] += 1
                        if decoded[index] != section[index]["summary"]["steps"] + fps:
                            raise ValueError("Selected source video does not align with its action log")
                timeline.append({
                    "pair_id": section[0]["summary"]["pair_id"], "task": section[0]["summary"]["task"],
                    "title_start_frame": start, "comparison_start_frame": comparisons_start,
                    "end_frame_exclusive": total_frames,
                    "episodes": [{"episode_id": record["summary"]["episode_id"], "method": record["summary"]["method"],
                                  "success": record["summary"]["success"], "steps": record["summary"]["steps"],
                                  "model_calls": record["summary"]["model_calls"], "decoded_source_frames": decoded[index]}
                                 for index, record in enumerate(section)],
                })
                print(f"Rendered {section[0]['summary']['task']}: {duration / fps:.2f} simulated-video seconds", flush=True)
        decoded_output = 0
        with imageio.get_reader(str(staged), input_params=["-threads", "1"]) as reader:
            metadata = reader.get_meta_data()
            if tuple(metadata["size"]) != (WIDTH, HEIGHT) or not math.isclose(metadata["fps"], fps, abs_tol=1e-6):
                raise ValueError("Rendered demo metadata mismatch")
            for frame in reader:
                if frame.shape != (HEIGHT, WIDTH, 3):
                    raise ValueError("Rendered demo frame shape mismatch")
                decoded_output += 1
        if decoded_output != total_frames:
            raise ValueError("Rendered demo frame count mismatch")
        check_unchanged(hashes)
        metadata = {
            "selection_rule": "Predetermined state 0, policy seed 11, all three configured tasks, fixed_10 / dvac_cap_10 / dvac_cap_50 in that order; independent of outcomes.",
            "all_270_verified_before_publish": True, "config_fingerprint": cfg.fingerprint,
            "verification_manifest_sha256": verification["manifest_sha256"], "fps": fps, "frames": total_frames,
            "duration_s": total_frames / fps, "width": WIDTH, "height": HEIGHT, "decoded_output_frames": decoded_output,
            "sha256": sha256(staged), "sections": timeline, "input_sha256": hashes,
            "timing_note": "Panels align by simulation step. Final observations are held after termination. This is not a wall-clock latency comparison.",
        }
        staged.replace(output)
        atomic_json(sidecar, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/capped"))
    parser.add_argument("--report", type=Path, default=Path("outputs/capped-report"))
    parser.add_argument("--output", type=Path, default=Path("outputs/capped-report/demo.mp4"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_demo(args.input, args.report, args.output, overwrite=args.overwrite)
    print(json.dumps({key: result[key] for key in ("frames", "duration_s", "width", "height", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
