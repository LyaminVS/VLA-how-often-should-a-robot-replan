#!/usr/bin/env python3
"""Make contact sheets for prespecified episodes and an optional failure diagnosis.

This script only reads raw evidence. It never infers visual behavior or failure
causes; manual_review.json starts with empty observations and interpretations.
Run on the Linux host after collection and independent verification finish.
"""

import argparse
import json
import math
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_replan.integrity import file_sha256
from vla_replan.storage import atomic_json

STATE, SEED = 15, 11
METHODS = ("fixed_10", "dvac")
STRIDE = 25
CLOSE_OFFSETS = (-10, -3, 0, 3, 10)
COLUMNS, ROWS = 3, 4
THUMBNAIL = (384, 256)
CAPTION_HEIGHT, TITLE_HEIGHT = 44, 72
PAGE_SIZE = (COLUMNS * THUMBNAIL[0], TITLE_HEIGHT + ROWS * (THUMBNAIL[1] + CAPTION_HEIGHT))


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_frames(rows):
    """Return action frame indices and explicit reasons, plus the post-action end."""
    if not rows:
        raise ValueError("Cannot review an empty episode")
    reasons = {}

    def add(index, reason):
        if 0 <= index < len(rows):
            reasons.setdefault(index, set()).add(reason)

    add(0, "initial")
    add(len(rows) - 1, "last_action")
    for index in range(0, len(rows), STRIDE):
        add(index, "regular_25_actions")
    close_edges = []
    previous = None
    for index, row in enumerate(rows):
        if type(row.get("step")) is not int or row["step"] != index:
            raise ValueError("Action rows must have consecutive integer step indices")
        grip = float(row["action"][6])
        if not math.isfinite(grip):
            raise ValueError("Invalid gripper command")
        if grip > 0 and (previous is None or previous <= 0):
            close_edges.append(index)
            for offset in CLOSE_OFFSETS:
                add(index + offset, f"close_edge_{index}{offset:+d}")
        previous = grip
    # The first outcome frame is immediately after the final executed action.
    reasons[len(rows)] = {"final_post_action_outcome"}
    return {index: sorted(reasons[index]) for index in sorted(reasons)}, close_edges


class ContactSheets:
    def __init__(self, destination, summary, expected_frames, review_roles):
        from PIL import ImageFont

        self.destination = destination
        self.summary = summary
        self.expected_frames = expected_frames
        self.review_roles = review_roles
        self.page_count = math.ceil(expected_frames / (COLUMNS * ROWS))
        self.pages = []
        self.records = []
        self.canvas = None
        self.font = ImageFont.load_default(size=13)
        self.title_font = ImageFont.load_default(size=15)

    def append(self, frame, metadata):
        from PIL import Image, ImageDraw

        if tuple(frame.shape) != (512, 768, 3):
            raise ValueError(f"Unexpected video frame shape: {frame.shape}")
        ordinal = len(self.records)
        slot = ordinal % (COLUMNS * ROWS)
        page = ordinal // (COLUMNS * ROWS) + 1
        if slot == 0:
            self.canvas = Image.new("RGB", PAGE_SIZE, "#101a2c")
            draw = ImageDraw.Draw(self.canvas)
            method = "DVAC" if self.summary["method"] == "dvac" else "Fixed K=10"
            role = "PRESELECTED" if "prespecified" in self.review_roles else "POSTHOC FAILURE DIAGNOSTIC"
            title = (f"{method} | state {self.summary['state']}, seed {self.summary['policy_seed']} | "
                     f"{role} | page {page}/{self.page_count}")
            draw.text((12, 7), title, fill="white", font=self.title_font)
            for line_index, line in enumerate(textwrap.wrap(self.summary["instruction"], width=130)[:2]):
                draw.text((12, 30 + 18 * line_index), line, fill="#bdcbe1", font=self.font)
        column, tile_row = slot % COLUMNS, slot // COLUMNS
        x = column * THUMBNAIL[0]
        y = TITLE_HEIGHT + tile_row * (THUMBNAIL[1] + CAPTION_HEIGHT)
        thumbnail = Image.fromarray(frame).resize(THUMBNAIL, Image.Resampling.LANCZOS)
        self.canvas.paste(thumbnail, (x, y + CAPTION_HEIGHT))
        draw = ImageDraw.Draw(self.canvas)
        color = "#f6c550" if metadata["replanned"] else "#d7e1f0"
        if metadata["is_final_frame"]:
            line1 = f"step {metadata['step']} | t={metadata['sim_time_s']:.2f}s | FINAL"
            line2 = f"Recorded outcome: {metadata['recorded_outcome']}"
        else:
            line1 = (f"step {metadata['step']} | t={metadata['sim_time_s']:.2f}s | "
                     f"grip={metadata['gripper_command']:+.3f}")
            line2 = (f"K={metadata['chosen_k']} | call={metadata['model_call']} | "
                     f"replan={'yes' if metadata['replanned'] else 'no'}")
        draw.text((x + 7, y + 5), line1, fill=color, font=self.font)
        draw.text((x + 7, y + 23), line2, fill=color, font=self.font)
        draw.rectangle((x, y, x + THUMBNAIL[0] - 1, y + CAPTION_HEIGHT + THUMBNAIL[1] - 1), outline="#42516b")
        self.records.append({**metadata, "page": f"sheet_{page:03d}.png",
                             "row": tile_row, "column": column, "ordinal": ordinal + 1})
        if slot == COLUMNS * ROWS - 1:
            self.flush()

    def flush(self):
        if self.canvas is not None:
            name = f"sheet_{len(self.pages) + 1:03d}.png"
            self.canvas.save(self.destination / name)
            self.pages.append(name)
            self.canvas = None


def review_episode(directory, output, summary, fps, review_roles):
    import imageio.v2 as imageio

    evidence = {}
    for filename in ("steps.jsonl", "chunks.jsonl", "video.mp4"):
        actual = file_sha256(directory / filename)
        if actual != summary["evidence_sha256"].get(filename):
            raise ValueError(f"Changed source evidence: {directory / filename}")
        evidence[filename] = actual
    rows = read_jsonl(directory / "steps.jsonl")
    chunks = read_jsonl(directory / "chunks.jsonl")
    if len(rows) != summary["steps"] or len(chunks) != summary["model_calls"]:
        raise ValueError("Summary/log count mismatch")
    selections, edges = selected_frames(rows)
    destination = output / summary["episode_id"]
    if destination.exists():
        raise FileExistsError(f"Review exists; preserve it and choose another output: {destination}")
    with tempfile.TemporaryDirectory(prefix=".dvac-review-", dir=output) as temporary:
        staging = Path(temporary)
        sheets = ContactSheets(staging, summary, len(selections), review_roles)
        reader = imageio.get_reader(str(directory / "video.mp4"), format="ffmpeg",
                                   input_params=["-threads", "1"])
        decoded = 0
        try:
            if not math.isclose(float(reader.get_meta_data()["fps"]), fps, abs_tol=1e-6):
                raise ValueError("Video and control frequencies disagree")
            for frame_index, frame in enumerate(reader):
                decoded += 1
                if frame_index not in selections:
                    continue
                final = frame_index == len(rows)
                metadata = {
                    "frame_index": frame_index,
                    "step": frame_index,
                    "sim_time_s": frame_index / fps,
                    "selection_reasons": selections[frame_index],
                    "is_final_frame": final,
                    "replanned": False,
                    "recorded_outcome": "SUCCESS" if summary["success"] else summary["termination"].upper(),
                }
                if not final:
                    row = rows[frame_index]
                    call_index = row["model_calls"] - 1
                    chunk = chunks[call_index]
                    if not chunk["step"] <= frame_index < chunk["step"] + chunk["executed_steps"]:
                        raise ValueError("Review frame cannot be matched to its model call")
                    metadata.update({
                        "replanned": row["replanned"], "gripper_command": row["action"][6],
                        "chosen_k": row["chosen_k"], "model_call": row["model_calls"],
                        "chunk_start_step": chunk["step"], "chunk_index": row["chunk_index"],
                        "threshold": chunk["decision"]["threshold"],
                        "first_crossing": chunk["decision"]["first_crossing"],
                    })
                sheets.append(frame, metadata)
        finally:
            reader.close()
        if decoded != len(rows) + fps or len(sheets.records) != len(selections):
            raise ValueError("Video/log frame alignment failed")
        sheets.flush()
        result = {
            "episode_id": summary["episode_id"], "method": summary["method"],
            "suite": summary["suite"], "task": summary["task"],
            "state": summary["state"], "policy_seed": summary["policy_seed"],
            "pair_id": summary["pair_id"], "review_roles": review_roles,
            "source_directory": str(directory), "source_video": str(directory / "video.mp4"),
            "source_sha256": evidence,
            "summary_sha256": file_sha256(directory / "summary.json"),
            "success": summary["success"], "termination": summary["termination"],
            "action_steps": len(rows), "model_calls": len(chunks), "fps": fps,
            "decoded_video_frames": decoded, "close_command_edge_steps": edges,
            "selected_frame_indices": list(selections), "pages": sheets.pages,
            "frames": sheets.records,
            "sheet_sha256": {name: file_sha256(staging / name) for name in sheets.pages},
            "note": "Action frames show the observation before the indexed action. The final frame follows all actions. Positive gripper-command edges are not proof of a grasp.",
        }
        atomic_json(staging / "index.json", result)
        staging.rename(destination)
    return result


def validate_pair(pair):
    if any(pair[0][field] != pair[1][field] for field in (
        "pair_id", "initial_state_sha256", "initial_observation_sha256", "config_fingerprint",
    )):
        raise ValueError("Selected review episodes are not a matched pair")


def discordant_failure(manifest, summaries):
    """Find the first eligible pair in the complete manifest's original order."""
    by_id = {summary["episode_id"]: summary for summary in summaries}
    planned = manifest["planned_episode_ids"]
    if len(by_id) != len(summaries) or set(by_id) != set(planned):
        raise ValueError("Posthoc selection requires the complete, unique planned episode inventory")
    for identity in planned:
        candidate = by_id[identity]
        if candidate["method"] != "dvac" or candidate["success"] is not False:
            continue
        matches = [summary for summary in summaries
                   if summary["pair_id"] == candidate["pair_id"] and summary["method"] == "fixed_10"]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous fixed_10 partner for failure inspection")
        if matches[0]["success"] is True:
            pair = [matches[0], candidate]
            validate_pair(pair)
            return pair
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/dvac"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dvac-report/review"))
    parser.add_argument("--include-discordant-failure", action="store_true",
                        help="Also inspect the first DVAC-failure/fixed10-success pair in manifest order; explicitly posthoc.")
    args = parser.parse_args()
    root, output = args.input.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Review output must be outside the immutable evidence directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Review output already contains files: {output}")
    manifest = json.loads((root / "manifest.json").read_text())
    base = manifest["config"]["base"]
    if STATE not in base["initial_states"] or SEED not in base["policy_seeds"]:
        raise ValueError("The prespecified state15/seed11 pair is absent from the experiment")
    fps = base["control_hz"]
    if type(fps) is not int or fps < 1:
        raise ValueError("Expected an integer control frequency")
    summaries = [json.loads(path.read_text()) for path in root.glob("episodes/*/summary.json")]
    selected = []
    roles = {}
    for task in base["tasks"]:
        pair = []
        for method in METHODS:
            matches = [s for s in summaries if s["suite"] == task["suite"] and s["task"] == task["name"]
                       and s["state"] == STATE and s["policy_seed"] == SEED and s["method"] == method]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous review episode: {task['name']}/{method}")
            pair.append(matches[0])
        validate_pair(pair)
        selected.extend(pair)
        for summary in pair:
            roles[summary["episode_id"]] = ["prespecified"]
    planned_ids = [summary["episode_id"] for summary in selected]
    posthoc_pair = discordant_failure(manifest, summaries) if args.include_discordant_failure else []
    for summary in posthoc_pair:
        identity = summary["episode_id"]
        if identity not in roles:
            selected.append(summary)
            roles[identity] = []
        roles[identity].append("posthoc_discordant_failure")
    selection = {
        "prespecified": {
            "rule": "State15 and policy seed11 for every configured task; fixed_10 and DVAC. Outcomes never affect this group.",
            "episode_ids": planned_ids,
        },
        "posthoc_discordant_failure": {
            "requested": args.include_discordant_failure,
            "rule": "First DVAC failure in original manifest planned order whose matched fixed_10 episode succeeds; inspect both videos.",
            "eligible_pair_found": bool(posthoc_pair),
            "pair_id": posthoc_pair[0]["pair_id"] if posthoc_pair else None,
            "episode_ids": [summary["episode_id"] for summary in posthoc_pair],
            "interpretation_limit": "Outcome-selected diagnostic example, not a representative error-rate estimate or evidence of causality. The prespecified demo is unchanged.",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for summary in selected:
        record = review_episode(root / "episodes" / summary["episode_id"], output, summary, fps,
                                roles[summary["episode_id"]])
        records.append(record)
        print(f"Contact sheets prepared: {summary['method']} / {summary['task']}: "
              f"{len(record['frames'])} frames, {len(record['pages'])} sheets", flush=True)
    atomic_json(output / "index.json", {
        "selection": selection,
        "frame_rule": "Initial, last action, every25 actions, positive gripper-command edges at offsets -10,-3,0,+3,+10, final post-action outcome frame; no frame-count cap.",
        "thumbnail_size": list(THUMBNAIL), "grid": [COLUMNS, ROWS], "page_size": list(PAGE_SIZE),
        "source_manifest_sha256": file_sha256(root / "manifest.json"),
        "builder_source_sha256": file_sha256(__file__),
        "episodes": [{"episode_id": item["episode_id"], "index": item["episode_id"] + "/index.json",
                      "review_roles": item["review_roles"],
                      "state": item["state"], "policy_seed": item["policy_seed"],
                      "success": item["success"], "termination": item["termination"],
                      "video_sha256": item["source_sha256"]["video.mp4"],
                      "frames": len(item["frames"]), "pages": len(item["pages"])} for item in records],
    })
    atomic_json(output / "manual_review.json", {
        "status": "not_reviewed", "reviewer": None,
        "selection": selection,
        "instructions": "Enter only visually supported observations, link episode/frame references, and keep interpretations distinct from observed facts. Leave uncertain causes unresolved. A closing command does not establish a successful grasp.",
        "episodes": [{"episode_id": item["episode_id"], "reviewed": False, "observed_facts": [],
                      "review_roles": item["review_roles"],
                      "interpretations": [], "uncertain_observations": [], "frame_references": []}
                     for item in records],
    })


if __name__ == "__main__":
    main()
