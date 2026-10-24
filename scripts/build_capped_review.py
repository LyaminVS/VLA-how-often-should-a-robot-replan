"""Create logged, paired contact sheets; never infer grasp errors or failure causes."""

import argparse
import json
import math
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.build_capped_demo import (
    ACCENT, BACKGROUND, METHODS, MUTED, WHITE, check_unchanged, font, input_hashes,
    label, predetermined_sections, protected_output, record_directory, sha256, verified_inputs, wrap_text,
)
from scripts.verify_dvac import read_jsonl
from vla_replan.storage import atomic_json

COLS, ROWS = 3, 6
TILE_WIDTH, IMAGE_HEIGHT, CAPTION_HEIGHT, HEADER = 512, 256, 64, 144
TILE_HEIGHT = IMAGE_HEIGHT + CAPTION_HEIGHT
WIDTH, HEIGHT = COLS * TILE_WIDTH, HEADER + ROWS * TILE_HEIGHT
CAMERA_CROP = (0, 128, 768, 512)


def select_groups(cfg, records, frozen_episode_order):
    """Preselected examples plus first discordance in either direction, without ranking severity."""
    by_pair = {}
    by_id = {record["summary"]["episode_id"]: record for record in records}
    for record in records:
        row = record["summary"]
        by_pair.setdefault(row["pair_id"], {})[row["method"]] = record
    groups, positions = [], {}

    def add(pair_id, reason):
        if pair_id not in positions:
            positions[pair_id] = len(groups)
            group = by_pair[pair_id]
            if not all(method in group for method in METHODS):
                raise ValueError("Incomplete review comparison")
            groups.append({"group_id": f"comparison_{len(groups) + 1:02d}", "pair_id": pair_id,
                           "records": [group[method] for method in METHODS], "selection_reasons": []})
        groups[positions[pair_id]]["selection_reasons"].append(reason)

    for section in predetermined_sections(cfg, records):
        add(section[0]["summary"]["pair_id"], "predetermined_state0_seed11")
    ordered_pairs = []
    for identity in frozen_episode_order:
        if identity not in by_id:
            raise ValueError("Frozen collection order includes an unknown episode")
        pair_id = by_id[identity]["summary"]["pair_id"]
        if pair_id not in ordered_pairs:
            ordered_pairs.append(pair_id)
    if set(ordered_pairs) != set(by_pair):
        raise ValueError("Frozen collection order does not cover every matched pair")
    selections = []
    for name, desired in (("cap10_success_cap50_failure", (True, False)),
                          ("cap10_failure_cap50_success", (False, True))):
        candidates = [pair_id for pair_id in ordered_pairs
                      if (by_pair[pair_id]["dvac_cap_10"]["summary"]["success"],
                          by_pair[pair_id]["dvac_cap_50"]["summary"]["success"]) == desired]
        chosen = candidates[0] if candidates else None
        if chosen is not None:
            add(chosen, name)
        selections.append({"rule": name, "posthoc_outcome_selected": True,
                           "qualifying_pairs": len(candidates), "selected_pair_id": chosen,
                           "selected_frozen_pair_position": ordered_pairs.index(chosen) if chosen is not None else None,
                           "absence_reason": None if chosen is not None else "No matched pair has this outcome pattern."})
    return groups, selections, ordered_pairs


def select_steps(logs, stride=20):
    if type(stride) is not int or stride < 1 or not logs:
        raise ValueError("Review requires action logs and a positive integer stride")
    maximum = max(len(rows) for rows in logs.values())
    if not maximum:
        raise ValueError("Empty action log")
    selected = set(range(0, maximum, stride)) | {maximum}
    transitions = {}
    for method, rows in logs.items():
        if [row.get("step") for row in rows] != list(range(len(rows))):
            raise ValueError("Review logs must have consecutive zero-based actions")
        selected.update((len(rows) - 1, len(rows)))
        method_transitions, previous = [], None
        for row in rows:
            grip = row["action"][6]
            if type(grip) not in (int, float) or not math.isfinite(grip):
                raise ValueError("Invalid logged gripper command")
            sign = (grip > 0) - (grip < 0)
            if previous is not None and sign != previous:
                method_transitions.append(row["step"])
                selected.update(range(max(0, row["step"] - 1), min(maximum, row["step"] + 1) + 1))
            previous = sign
        transitions[method] = method_transitions
    return sorted(selected), transitions


def review_group(group, destination, fps, stride):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    records = group["records"]
    logs = {record["summary"]["method"]: read_jsonl(record_directory(record) / "steps.jsonl") for record in records}
    steps, transitions = select_steps(logs, stride)
    maximum = max(len(rows) for rows in logs.values())
    pages, frame_records = [], []
    canvas = None
    decoded = [0] * len(records)
    wanted = set(steps)
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(str(record_directory(record) / "video.mp4"),
                                                          input_params=["-threads", "1"])) for record in records]
        for reader in readers:
            metadata = reader.get_meta_data()
            if tuple(metadata["size"]) != (768, 512) or not math.isclose(metadata["fps"], fps, abs_tol=1e-6):
                raise ValueError("Review source video metadata mismatch")
        iterators, frames = [iter(reader) for reader in readers], [None] * len(records)
        for step in range(maximum + 1):
            for index, record in enumerate(records):
                if step <= record["summary"]["steps"]:
                    try:
                        frames[index] = next(iterators[index])
                    except StopIteration as exc:
                        raise ValueError("Review source ended before its terminal observation") from exc
                    decoded[index] += 1
                    if frames[index].shape != (512, 768, 3):
                        raise ValueError("Review source frame shape mismatch")
            if step not in wanted:
                continue
            ordinal = len(frame_records) // len(records)
            page_number, row_number = divmod(ordinal, ROWS)
            page_name = f"sheet_{page_number + 1:03d}.png"
            if row_number == 0:
                canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
                draw = ImageDraw.Draw(canvas)
                first = records[0]["summary"]
                reason = " / ".join(group["selection_reasons"])
                draw.text((16, 8), f"State {first['state']}, seed {first['policy_seed']} | page {page_number + 1}/{math.ceil(len(steps) / ROWS)}",
                          font=font(22, True), fill=WHITE)
                instruction_lines = wrap_text(draw, first["instruction"], font(19), WIDTH - 32)
                if len(instruction_lines) > 2:
                    raise ValueError("Review instruction exceeds its allocated header")
                for index, line in enumerate(instruction_lines):
                    draw.text((16, 39 + 24 * index), line, font=font(19), fill=WHITE)
                draw.text((16, 91), "Selection: " + reason, font=font(12), fill=MUTED)
                for column, record in enumerate(records):
                    draw.text((column * TILE_WIDTH + 12, 115), label(record["summary"]["method"]), font=font(20, True), fill=ACCENT)
            draw = ImageDraw.Draw(canvas)
            for column, (record, frame) in enumerate(zip(records, frames, strict=True)):
                summary = record["summary"]
                method = summary["method"]
                ended = step >= summary["steps"]
                source_index = min(step, summary["steps"])
                left, top = column * TILE_WIDTH, HEADER + row_number * TILE_HEIGHT
                camera = Image.fromarray(frame).crop(CAMERA_CROP).resize((TILE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
                canvas.paste(camera, (left, top + CAPTION_HEIGHT))
                if ended:
                    chosen, grip, replan = None, None, None
                    caption = f"t={step/fps:.2f}s | {'SUCCESS' if summary['success'] else 'FAILURE'} | held"
                    details = f"End at step {summary['steps']} | {summary['model_calls']} calls"
                    color = "#71e4ac" if summary["success"] else "#ff8d91"
                else:
                    row = logs[method][step]
                    chosen, grip, replan = row.get("chosen_k", summary["k"]), row["action"][6], row["replanned"]
                    caption = f"step {step} | t={step/fps:.2f}s | {'REPLAN' if replan else 'executing'}"
                    details = f"K={chosen} | calls={row['model_calls']} | grip={grip:+.3f}"
                    color = ACCENT if replan else WHITE
                draw.text((left + 9, top + 5), caption, font=font(17), fill=color)
                draw.text((left + 9, top + 31), details, font=font(16), fill=MUTED)
                draw.line((left, top + CAPTION_HEIGHT - 1, left + TILE_WIDTH - 1, top + CAPTION_HEIGHT - 1), fill="#61718a")
                frame_records.append({"method": method, "episode_id": summary["episode_id"], "shared_step": step,
                                      "sim_time_s": step / fps, "source_frame_index": source_index,
                                      "final_frame_held": ended, "chosen_k": chosen, "gripper_command": grip,
                                      "replanned": replan, "page": page_name, "row": row_number, "column": column})
            if row_number == ROWS - 1 or step == steps[-1]:
                canvas.save(destination / page_name)
                pages.append({"file": page_name, "sha256": sha256(destination / page_name)})
        for index, iterator in enumerate(iterators):
            for frame in iterator:
                if frame.shape != (512, 768, 3):
                    raise ValueError("Review source frame shape mismatch")
                decoded[index] += 1
            if decoded[index] != records[index]["summary"]["steps"] + fps:
                raise ValueError("Review source video/log frame count mismatch")
    first = records[0]["summary"]
    metadata = {"group_id": group["group_id"], "pair_id": group["pair_id"], "suite": first["suite"], "task": first["task"],
                "state": first["state"], "policy_seed": first["policy_seed"], "selection_reasons": group["selection_reasons"],
                "posthoc_outcome_selected": any(reason != "predetermined_state0_seed11" for reason in group["selection_reasons"]),
                "selected_shared_steps": steps, "gripper_sign_change_steps": transitions, "pages": pages, "frames": frame_records,
                "episodes": [{"episode_id": record["summary"]["episode_id"], "method": record["summary"]["method"],
                              "success": record["summary"]["success"], "steps": record["summary"]["steps"],
                              "model_calls": record["summary"]["model_calls"], "decoded_video_frames": decoded[index],
                              "raw_source_root": record["raw_source_root"]} for index, record in enumerate(records)],
                "reviewed": False, "observations": [], "hypotheses": [],
                "note": "Action frames precede the logged action; the terminal frame follows the last action. Gripper transitions are commands, not detected grasps. No behavioral interpretation is inferred."}
    atomic_json(destination / "index.json", metadata)
    return metadata


def build_review(raw, report, output, *, stride=20, overwrite=False):
    cfg, records, verification, binding = verified_inputs(raw, report)
    output = Path(output).resolve()
    protected_output(output, records)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Derived review exists; use --overwrite to replace it")
    old_managed = set()
    if (output / "review.json").is_file():
        previous = json.loads((output / "review.json").read_text())
        for group in previous.get("groups", []):
            names = ["index.json", *(page["file"] for page in group.get("pages", []))]
            for name in names:
                path = (output / group["group_id"] / name).resolve()
                if not path.is_relative_to(output):
                    raise ValueError("Previous derived review inventory escapes its output directory")
                old_managed.add(path)
    manifest = json.loads((Path(raw) / "manifest.json").read_text())
    groups, outcome_selection, pair_order = select_groups(cfg, records, manifest["planned_episode_ids"])
    selected = {record["summary"]["episode_id"]: record for group in groups for record in group["records"]}
    hashes = {**binding, **input_hashes(selected.values()), str(Path(__file__).resolve()): sha256(__file__),
              str(PROJECT / "scripts/build_capped_demo.py"): sha256(PROJECT / "scripts/build_capped_demo.py")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".capped-review-", dir=output.parent) as temporary:
        staging = Path(temporary)
        result_groups = []
        for group in groups:
            directory = staging / group["group_id"]
            directory.mkdir()
            result_groups.append(review_group(group, directory, cfg.base.control_hz, stride))
            print(f"Prepared {group['group_id']}: {len(result_groups[-1]['pages'])} paired sheets", flush=True)
        check_unchanged(hashes)
        metadata = {
            "all_270_verified_before_publish": True, "config_fingerprint": cfg.fingerprint,
            "verification_manifest_sha256": verification["manifest_sha256"],
            "selection_rule": "All three predetermined state-0/seed-11 pairs, plus the first cap-10-success/cap-50-failure pair and first reverse pair in the frozen collection order. A pair is ordered by its first episode occurrence; duplicate selections share one review group.",
            "outcome_selected_cases": outcome_selection, "frozen_pair_order": pair_order,
            "frame_selection_rule": "Shared simulation-time grid at the requested stride, each episode's last action and terminal observation, and one-step neighborhoods around every logged gripper-command sign change. Ended panels hold their terminal observation.",
            "stride": stride, "fps": cfg.base.control_hz, "page_size": [WIDTH, HEIGHT],
            "rows_per_page": ROWS, "columns": list(METHODS), "camera_crop": list(CAMERA_CROP),
            "input_sha256": hashes, "groups": result_groups, "reviewed": False,
            "interpretation_note": "These are review materials. Outcome-selected cases are illustrations, not an unbiased estimate of error frequency. No grasp labels, causes, or scientific conclusions are generated by this script.",
        }
        atomic_json(staging / "review.json", metadata)
        output.mkdir(parents=True, exist_ok=True)
        published = set()
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                relative = path.relative_to(staging)
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
                published.add(destination.resolve())
        for stale in old_managed - published:
            if stale.is_file():
                stale.unlink()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/capped"))
    parser.add_argument("--report", type=Path, default=Path("outputs/capped-report"))
    parser.add_argument("--output", type=Path, default=Path("outputs/capped-report/review"))
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_review(args.input, args.report, args.output, stride=args.stride, overwrite=args.overwrite)
    print(json.dumps({"groups": len(result["groups"]), "pages": sum(len(group["pages"]) for group in result["groups"]),
                      "outcome_selected_cases": result["outcome_selected_cases"]}, indent=2))


if __name__ == "__main__":
    main()
