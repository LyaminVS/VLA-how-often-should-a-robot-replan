"""Append five verified moving-target pages to the unchanged eight-page study PDF.

Run on the Linux experiment host with reportlab and pypdf. Authoring-marker handling
and visual inspection are external responsibilities; this script does not run the
marker. The intermediate five-page PDF is kept under tmp/pdfs, never published alone.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moving_target.config import CONDITIONS
from scripts.analyze_moving_target import (
    DATA_KIND, cluster_statistics, contact_and_motion_summary, interaction_statistics, load_run, read_jsonl, sha256,
)
from scripts.build_report import BLUE, INK, MUTED, PALE, TEAL, clean, font_paths, num
from vla_replan.config import digest
from vla_replan.storage import atomic_json

BASELINE_PDF_SHA256 = "694e08619bbd91583fdeafc1a7a157354801d4bd115df9b7f29709a124fcd0b8"


def snapshot_json(path, snapshots):
    path = Path(path).resolve()
    data = path.read_bytes()
    snapshots[path] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def verify_snapshots(snapshots):
    for path, expected in snapshots.items():
        if sha256(path) != expected:
            raise ValueError(f"Source changed while the report was built: {path}")


def checked_child(root, relative):
    result = (root / relative).resolve()
    if result == root or root not in result.parents:
        raise ValueError(f"Unsafe evidence path: {relative}")
    return result


def load_evidence(args):
    """Recompute the complete analysis and validate every figure and calibration input."""
    snapshots = {Path(__file__).resolve(): sha256(__file__)}
    cfg, rows, provenance = load_run(args.run)
    raw_root = args.run.resolve()
    snapshots[raw_root / "manifest.json"] = provenance["manifest_sha256"]
    for episode in provenance["episodes"]:
        directory = raw_root / "episodes" / episode["episode_id"]
        snapshots[directory / "evidence.json"] = episode["evidence_sha256"]
        snapshots.update({directory / filename: expected for filename, expected in episode["files"].items()})
    base = cfg.base_config
    if (len(rows) != 135 or tuple(sorted(base.initial_states)) != (10, 11, 12, 13, 14)
            or tuple(sorted(base.policy_seeds)) != (11, 22, 33)
            or tuple(sorted(base.intervals)) != (5, 10, 50) or len(base.tasks) != 1
            or base.tasks[0].max_steps != 280 or not base.record_video):
        raise ValueError("The complete report requires the fixed 135-episode moving-target protocol")
    analysis = args.analysis.resolve()
    stats_path = analysis / "statistics.json"
    stats = snapshot_json(stats_path, snapshots)
    if (stats.get("data_kind") != DATA_KIND or stats.get("complete") is not True
            or stats.get("episodes") != 135 or digest(stats.get("config")) != cfg.fingerprint
            or stats.get("intervals") != [5, 10, 50] or stats.get("provenance") != provenance):
        raise ValueError("Statistics do not describe the current complete raw run")
    bootstrap = stats.get("bootstrap", {})
    if (bootstrap.get("draws") != 5000 or bootstrap.get("unit") != "initial_state"
            or bootstrap.get("policy_seeds_retained_in_cluster") is not True):
        raise ValueError("Expected 5000 paired initial-state bootstrap draws retaining all seeds")
    estimates, differences = cluster_statistics(rows, stats["intervals"], draws=bootstrap["draws"], seed=bootstrap["seed"])
    expected = {
        "estimates": estimates, "paired_vs_static": differences,
        "condition_by_k_interactions": interaction_statistics(
            rows, stats["intervals"], draws=bootstrap["draws"], seed=bootstrap["seed"]),
        "contact_and_motion": contact_and_motion_summary(rows, stats["intervals"]),
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"Stored statistics are stale or changed: {key}")
    analyzer = Path(__file__).with_name("analyze_moving_target.py").resolve()
    snapshots[analyzer] = sha256(analyzer)
    if stats.get("analysis_source_sha256") != snapshots[analyzer]:
        raise ValueError("Rebuild analysis after changing its source")
    artifacts_path = analysis / "artifacts.json"
    artifacts = snapshot_json(artifacts_path, snapshots)
    if (artifacts.get("data_kind") != DATA_KIND
            or artifacts.get("statistics_sha256") != snapshots[stats_path]
            or artifacts.get("analysis_source_sha256") != snapshots[analyzer]):
        raise ValueError("Analysis artifact provenance is stale")
    required = {"success-and-cost.png", "success-and-cost.pdf", "motion-and-contact.png",
                "motion-and-contact.pdf", "report.md", "episodes.csv", "summary.csv",
                "paired-vs-static.csv", "condition-by-k-interactions.csv", "contact-coverage.csv"}
    if set(artifacts.get("files", {})) != required:
        raise ValueError("Analysis artifacts are incomplete")
    for relative, expected_hash in artifacts["files"].items():
        path = checked_child(analysis, relative)
        if sha256(path) != expected_hash:
            raise ValueError(f"Changed analysis artifact: {relative}")
        snapshots[path] = expected_hash

    calibration_path = args.calibration.resolve()
    calibration = snapshot_json(calibration_path, snapshots)
    if (calibration.get("motion") != cfg.motion.asdict() or calibration.get("state") != 20
            or calibration.get("steps") != 240 or set(calibration.get("results", {})) != set(CONDITIONS)):
        raise ValueError("Calibration must be the selected physical controller on excluded state 20")
    starts = set()
    for condition in CONDITIONS:
        record = calibration["results"][condition]
        reset, final = record["reset"], record["final"]
        starts.add((reset["initial_state_sha256"], reset["initial_observation_sha256"]))
        if final["first_contact_step"] is not None or final["first_contact_sim_time"] is not None:
            raise ValueError("Physical calibration should have no finger contact")
        for key in ("precontact_x_range_m", "precontact_y_range_m", "precontact_z_range_m",
                    "max_precontact_tilt_deg", "peak_force_n", "max_force_after_contact_n"):
            if not isinstance(final[key], (int, float)) or not math.isfinite(final[key]) or final[key] < 0:
                raise ValueError(f"Invalid calibration diagnostic: {condition}/{key}")
        if final["peak_force_n"] > cfg.motion.max_force_n + 1e-9 or final["max_force_after_contact_n"] > 1e-9:
            raise ValueError("Calibration force integrity check failed")
        if condition == "static" and (final["peak_force_n"] or final["active_physics_samples"]):
            raise ValueError("The static calibration applied intervention force")
    if len(starts) != 1:
        raise ValueError("Calibration physical starts were not matched")
    frozen_path = args.frozen_protocol.resolve()
    frozen = snapshot_json(frozen_path, snapshots)
    if (frozen.get("config_fingerprint") != cfg.fingerprint
            or digest(frozen.get("main_config")) != cfg.fingerprint
            or frozen.get("planned_episode_ids") != [episode.id for episode in cfg.episodes()]
            or frozen.get("sources") != provenance["sources"]
            or frozen.get("baseline_runtime") != provenance["runtime"]):
        raise ValueError("The completed run does not match the frozen protocol")
    checks = frozen.get("model_checks", {})
    if (checks.get("episodes") != 9 or checks.get("contacts") != 9
            or checks.get("all_intervention_and_schedule_checks_passed") is not True):
        raise ValueError("Missing pre-run contact-release and schedule checks")
    calibration_hashes = frozen.get("calibration_files", {})
    if not calibration_hashes:
        raise ValueError("Frozen protocol lacks physical calibration evidence")
    for relative, expected_hash in calibration_hashes.items():
        path = checked_child(frozen_path.parent, relative)
        if sha256(path) != expected_hash:
            raise ValueError(f"Calibration evidence differs from the frozen protocol: {relative}")
        snapshots[path] = expected_hash
    calibration_relative = str(calibration_path.relative_to(frozen_path.parent))
    if calibration_hashes.get(calibration_relative) != snapshots[calibration_path]:
        raise ValueError("Selected calibration is not covered by frozen evidence")
    protocol_path = args.protocol.resolve()
    snapshots[protocol_path] = sha256(protocol_path)
    if frozen.get("protocol_sha256") != snapshots[protocol_path]:
        raise ValueError("Protocol document changed after the experiment was frozen")

    verification_path = analysis / "evidence-verification.json"
    verification = snapshot_json(verification_path, snapshots)
    verifier_path = Path(__file__).with_name("verify_moving_target.py").resolve()
    snapshots[verifier_path] = sha256(verifier_path)
    required_checks = {
        "complete_plan_and_raw_checksums", "current_sources_match_frozen_run", "freeze_precedes_main_creation",
        "physics_b_bound_to_freeze", "full_50_by_7_chunks_finite", "executed_actions_equal_clipped_prefix",
        "sampling_seeds_and_fixed_k_schedule", "success_and_gripper_bookkeeping",
        "zero_force_before_onset_static_and_after_contact", "identical_pre_motion_actions_and_full_predictions",
        "all_video_frames_decoded",
    }
    checks = verification.get("checks", {})
    if (verification.get("schema_version") != 1
            or verification.get("artifact_kind") != "moving_target_evidence_verification"
            or verification.get("data_kind") != DATA_KIND or verification.get("all_checks_passed") is not True
            or verification.get("config_fingerprint") != cfg.fingerprint
            or verification.get("provenance") != provenance
            or verification.get("verifier_source_sha256") != snapshots[verifier_path]
            or verification.get("analysis_source_sha256") != snapshots[analyzer]
            or verification.get("verified_episodes") != 135 or verification.get("decoded_videos") != 135
            or verification.get("video_verification_mode") != "decoded_all_frames"
            or verification.get("prefix_pairs") != 45
            or verification.get("condition_vs_static_prefix_comparisons") != 90
            or not required_checks <= checks.keys() or any(value is not True for value in checks.values())):
        raise ValueError("Complete current action-prefix and video-decoding verification is required")
    for field in ("max_executed_action_error", "max_action_prefix_error", "max_prediction_prefix_error"):
        if type(verification.get(field)) not in (int, float) or verification[field] != 0:
            raise ValueError(f"Verification reports nonzero or unknown action error: {field}")
    total_steps = sum(row["steps"] for row in rows)
    total_calls = sum(row["model_calls"] for row in rows)
    expected_totals = {"steps": total_steps, "model_calls": total_calls,
        "executed_action_values": total_steps * 7, "predicted_action_values": total_calls * base.chunk_size * 7,
        "decoded_frames": total_steps + 135 * base.control_hz}
    if any(verification.get(key) != value for key, value in expected_totals.items()):
        raise ValueError("Verification totals do not match all completed action logs and videos")
    prefix_checks = verification.get("prefix_checks", [])
    expected_blocks = {(state, seed, k) for state in base.initial_states
                       for seed in base.policy_seeds for k in base.intervals}
    if (len(prefix_checks) != 45
            or {(item.get("state"), item.get("policy_seed"), item.get("k")) for item in prefix_checks} != expected_blocks
            or verification.get("prefix_action_steps") != list(range(math.ceil(cfg.motion.start_delay_s * base.control_hz)))):
        raise ValueError("Verification does not cover every pre-motion state/seed/K triplet")
    for item in prefix_checks:
        if (item.get("condition_vs_static_comparisons") != 2
                or item.get("max_action_prefix_error") != 0 or item.get("max_prediction_prefix_error") != 0
                or not isinstance(item.get("compared_action_vectors"), int) or item["compared_action_vectors"] <= 0
                or not isinstance(item.get("compared_prediction_chunks"), int) or item["compared_prediction_chunks"] <= 0):
            raise ValueError("A pre-motion triplet lacks exact action and full-prediction equality")
    from scripts.verify_moving_target import verify_frozen
    if sha256(raw_root / "manifest.json") != snapshots[raw_root / "manifest.json"]:
        raise ValueError("Raw manifest changed during report validation")
    manifest = json.loads((raw_root / "manifest.json").read_text())
    if verification.get("frozen_evidence") != verify_frozen(cfg, manifest, frozen_path):
        raise ValueError("Verification does not bind the current frozen calibration and collection source")

    review = snapshot_json(analysis / "demo-review.json", snapshots)
    if (review.get("schema_version") != 1 or review.get("role") != "manual_assistant_visual_review"
            or review.get("reviewer") != "assistant" or review.get("independent_human_review") is not False
            or review.get("selection_fixed_before_selected_outcomes") is not True
            or review.get("state") != 10 or review.get("policy_seed") != 11
            or set(review.get("cameras", [])) != {"agentview", "eye_in_hand"}):
        raise ValueError("Demo review must describe the preselected assistant-reviewed five cases")
    for field in ("selection_rule", "frame_indexing", "review_method", "overall_conclusion"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"Missing demo-review description: {field}")
    selected = {("static", 10), ("slow", 10), ("fast", 10), ("fast", 5), ("fast", 50)}
    by_id = {row["episode_id"]: row for row in rows}
    raw_evidence = {item["episode_id"]: item for item in provenance["episodes"]}
    expected_ids = {row["episode_id"] for row in rows if row["state"] == 10 and row["policy_seed"] == 11
                    and (row["condition"], row["k"]) in selected}
    reviewed = review.get("episodes", [])
    if len(reviewed) != 5 or {item.get("episode_id") for item in reviewed} != expected_ids:
        raise ValueError("Demo review must cover exactly the five prespecified recordings")
    for item in reviewed:
        row, evidence = by_id[item["episode_id"]], raw_evidence[item["episode_id"]]
        fields = ("condition", "k", "state", "policy_seed", "steps", "model_calls", "success",
                  "first_contact_step", "first_contact_sim_time_s")
        if any(item.get(field) != row[field] for field in fields):
            raise ValueError("Demo-review identity, outcome or contact counters differ from raw data")
        if item.get("raw_files") != evidence["files"] or item.get("evidence_sha256") != evidence["evidence_sha256"]:
            raise ValueError("Demo review is not bound to the current raw files")
        if any(field not in item or item[field] is not None for field in ("missed_grasps", "dropped_objects")):
            raise ValueError("Sampled demo review must not invent resolved grasp/drop event counts")
        inspected = item.get("inspected_frame_indices", [])
        if (not inspected or inspected != sorted(set(inspected))
                or any(type(frame) is not int or not 0 <= frame <= row["steps"] for frame in inspected)):
            raise ValueError("Invalid inspected video-frame inventory")
        window_frames = set()
        for window in item.get("inspected_windows", []):
            start, end, stride = (window.get(field) for field in ("start_frame", "end_frame", "stride"))
            if (any(type(value) is not int for value in (start, end, stride)) or stride <= 0
                    or not 0 <= start <= end <= row["steps"]):
                raise ValueError("Invalid sampled visual-review window")
            extra = window.get("additional_frames", [])
            if any(type(frame) is not int or not 0 <= frame <= row["steps"] for frame in extra):
                raise ValueError("Invalid extra reviewed frame")
            window_frames.update(range(start, end + 1, stride))
            window_frames.update(extra)
        if set(inspected) != window_frames:
            raise ValueError("Review windows and inspected frame inventory disagree")
        observations = item.get("observations", [])
        if not observations or not item.get("uncertainties"):
            raise ValueError("Demo review requires observations and explicit uncertainties")
        for observation in observations:
            frames = observation.get("frames", [])
            if (not frames or not set(frames) <= window_frames
                    or not isinstance(observation.get("text"), str) or not observation["text"].strip()
                    or observation.get("sim_time_s") != [min(frames) / base.control_hz, max(frames) / base.control_hz]):
                raise ValueError("Visual observation is not supported by its inspected frames and times")

    baseline_path = args.baseline.resolve()
    snapshots[baseline_path] = sha256(baseline_path)
    if snapshots[baseline_path] != BASELINE_PDF_SHA256:
        raise ValueError("The source PDF is not the frozen unchanged eight-page study report")
    baseline_record_path = baseline_path.with_suffix(".json")
    baseline_record = snapshot_json(baseline_record_path, snapshots)
    if (baseline_record.get("pdf_sha256") != snapshots[baseline_path]
            or baseline_record.get("pages") != 8 or baseline_record.get("primary_phase") != "combined"):
        raise ValueError("Baseline PDF or its eight-page provenance record changed")
    return cfg, rows, stats, calibration, snapshots, artifacts_path, frozen_path, review


def page_content_hash(page):
    contents = page.get_contents()
    return hashlib.sha256(contents.get_data() if contents else b"").hexdigest()


def build_report(args):
    cfg, rows, stats, calibration, snapshots, artifacts_path, frozen_path, review = load_evidence(args)
    output = args.output.resolve()
    record_path = output.with_suffix(".json")
    if output.suffix.lower() != ".pdf" or output in snapshots or record_path in snapshots:
        raise ValueError("Choose a new PDF output that cannot overwrite an input")
    if (output.exists() or record_path.exists()) and not args.overwrite:
        raise FileExistsError("Output already exists; pass --overwrite to rebuild")
    for source in (args.run.resolve(), args.analysis.resolve(), args.frozen_protocol.resolve().parent):
        if output == source or source in output.parents:
            raise ValueError("The final PDF must be outside raw, analysis and calibration directories")

    from pypdf import PdfReader, PdfWriter
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular, bold = font_paths(args.font, args.font_bold)
    pdfmetrics.registerFont(TTFont("MovingReport", regular))
    pdfmetrics.registerFont(TTFont("MovingReportBold", bold))
    pdfmetrics.registerFontFamily("MovingReport", normal="MovingReport", bold="MovingReportBold", italic="MovingReport")
    width, height = A4
    body_width = width - 36 * mm
    styles = {
        "body": ParagraphStyle("moving-body", fontName="MovingReport", fontSize=9.1, leading=12.3,
                               textColor=colors.HexColor(INK), spaceAfter=6),
        "small": ParagraphStyle("moving-small", fontName="MovingReport", fontSize=7.9, leading=10.4,
                                textColor=colors.HexColor(MUTED), spaceAfter=5),
        "h1": ParagraphStyle("moving-h1", fontName="MovingReportBold", fontSize=18, leading=22,
                             textColor=colors.HexColor(BLUE), spaceAfter=10),
        "h2": ParagraphStyle("moving-h2", fontName="MovingReportBold", fontSize=11, leading=14,
                             textColor=colors.HexColor(TEAL), spaceBefore=7, spaceAfter=5),
        "kicker": ParagraphStyle("moving-kicker", fontName="MovingReportBold", fontSize=8.4, leading=11,
                                 textColor=colors.HexColor(TEAL), spaceAfter=8),
        "cell": ParagraphStyle("moving-cell", fontName="MovingReport", fontSize=7.9, leading=10.4,
                               textColor=colors.HexColor(INK)),
        "head": ParagraphStyle("moving-head", fontName="MovingReportBold", fontSize=7.9, leading=10.4,
                               textColor=colors.white),
        "code": ParagraphStyle("moving-code", fontName="MovingReport", fontSize=7.1, leading=9.5,
                               textColor=colors.HexColor(INK), wordWrap="CJK", spaceAfter=7,
                               backColor=colors.HexColor(PALE), borderPadding=5),
    }

    def p(value, style="body", *, markup=False):
        return Paragraph(str(value) if markup else clean(value), styles[style])

    def table(headers, data, fractions, *, padding=4):
        result = Table([[p(value, "head") for value in headers]] + [[p(value, "cell") for value in row] for row in data],
                       colWidths=[body_width * value for value in fractions], repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(PALE), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), padding), ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ]))
        return result

    def figure(name, aspect, caption, scale=1):
        return [Image(str(args.analysis / f"{name}.png"), width=body_width * scale,
                      height=body_width * scale * aspect), p(caption, "small")]

    def interval(item):
        return f"[{item['ci_low'] * 100:.1f}, {item['ci_high'] * 100:.1f}]"

    base, motion = cfg.base_config, cfg.motion
    first_after_onset = {k: (math.floor(motion.start_delay_s * base.control_hz / k + 1e-10) + 1)
                        * k / base.control_hz for k in (5, 10, 50)}
    phase_example = ""
    if (motion.ramp_s <= motion.periods_s["slow"] / 4
            and math.isclose(first_after_onset[50], motion.start_delay_s + motion.periods_s["fast"] / 2)
            and math.isclose(first_after_onset[50], motion.start_delay_s + motion.periods_s["slow"] / 4)):
        phase_example = (f"At {first_after_onset[50]:g} s, the fast reference has returned to its initial x "
                         "position while the slow reference is near its maximum offset. ")
    by_cell = {(row["condition"], row["k"]): row for row in stats["contact_and_motion"]}
    estimates = {(row["condition"], row["k"], row["metric"]): row for row in stats["estimates"]}
    differences = [row for row in stats["paired_vs_static"] if row["metric"] == "success"]
    interactions = stats["condition_by_k_interactions"]
    story = []

    # Page 9: fixed intervention and measured physical calibration.
    story.extend([
        p("SMOLVLA / LIBERO | MOVING-TARGET STRESS TEST", "kicker"),
        p("Replanning while the target moves", "h1"),
        p("The soup-can task is modified by a controlled external force before finger contact. "
          "The question is whether sensitivity to this intervention changes with K. These 135 episodes "
          "have their own matched static control and are not pooled with the standard-task results."),
        table(["Factor", "Fixed setting"], [
            ["Task and horizon", "Alphabet soup can into basket; 280 actions at 20 Hz"],
            ["Matched design", "States 10-14; policy seeds 11, 22, 33; K = 5, 10, 50"],
            ["Conditions and size", "Static, slow reference, fast reference; 15 episodes per cell; 135 total"],
            ["Policy and order", "Same pinned SmolVLA, H=50, FP32; condition/K order randomized within state/seed"],
        ], [.25, .75]),
        p("A bounded point force, released at first finger contact", "h2"),
        p(f"The x reference has amplitude {100 * motion.amplitude_m:g} cm and periods "
          f"{motion.periods_s['slow']:g}/{motion.periods_s['fast']:g} s. It starts at {motion.start_delay_s:g} simulated "
          f"seconds with a {motion.ramp_s:g} s smoothstep ramp. A PD controller uses kp={motion.kp_n_per_m:g} N/m, "
          f"kd={motion.kd_ns_per_m:g} N s/m and a {motion.max_force_n:g} N cap. Its application point is "
          f"{abs(motion.force_application_offset_z_m) * 100:g} cm below the center of mass; torque_y = offset_z * force_x. "
          "The can is never teleported and there is no separate orientation servo."),
        p("The first finger/pad contact permanently removes force and moment. Contact is not a verified grasp. "
          "Static uses the same instrumentation with zero intervention force. The intervention never triggers "
          "an extra model call; simulation pauses during inference.", "small"),
        p("Physical calibration with the robot stationary", "h2"),
    ])
    calibration_fields = [
        ("Actual x range (mm)", "precontact_x_range_m", 1000, 2),
        ("Actual y range (mm)", "precontact_y_range_m", 1000, 2),
        ("Actual z range (mm)", "precontact_z_range_m", 1000, 3),
        ("Maximum axis tilt from initial (degrees)", "max_precontact_tilt_deg", 1, 2),
        ("Active mean tracking error (mm)", "tracking_mae_m", 1000, 2),
        ("Peak applied force (N)", "peak_force_n", 1, 3),
        ("Enabled samples at force cap (%)", "force_saturation_fraction", 100, 1),
    ]
    calibration_table = []
    for label, field, scale, digits in calibration_fields:
        values = [calibration["results"][condition]["final"].get(field) for condition in CONDITIONS]
        calibration_table.append([label] + [num(None if value is None else value * scale, digits) for value in values])
    story.extend([
        table(["Physical diagnostic", "Static", "Slow", "Fast"], calibration_table, [.49, .17, .17, .17]),
        p("Calibration: excluded state 20, 240 actions (12 simulated seconds), no VLA inference, zero robot "
          "motion and open gripper. The selected controller still permits tilt and lateral drift. Slow/fast "
          "label reference periods, not perfectly isolated measured object speeds. Parameters were frozen "
          "from physical tracking and stability before collecting the main outcomes.", "small"),
        PageBreak(),
    ])

    # Page 10: all raw cell counts and comparable computational cost.
    story.extend([
        p("Completion and computational cost", "h1"),
        p("Every planned episode contributes, including no-contact episodes and unsuccessful transfers. "
          "Each condition/K cell contains five initial states with three policy-seed repeats."),
    ])
    story.extend(figure("success-and-cost", 3.8 / 12,
        "Vertical intervals resample initial states with their complete seed/treatment blocks. "
        "They are exploratory 95% percentile intervals, not independent-episode intervals."))
    cost_table = []
    for condition in CONDITIONS:
        for k in (5, 10, 50):
            cell = by_cell[condition, k]
            get = lambda metric: estimates[condition, k, metric]["mean"]
            cost_table.append([condition.capitalize(), k,
                f"{cell['successes']}/15 ({100 * get('success'):.1f}%)", num(get("model_calls")),
                num(get("compute_s"), 2), num(get("episode_wall_s"), 2)])
    seeds = sorted(base.policy_seeds)
    static_seed_counts = "; ".join(
        f"K={k}: " + "/".join(str(sum(row["success"] for row in rows
            if row["condition"] == "static" and row["k"] == k and row["policy_seed"] == seed))
            for seed in seeds) for k in (5, 10, 50)
    )
    story.extend([
        table(["Condition", "K", "Success", "Mean calls", "Compute (s)", "Wall time (s)"],
              cost_table, [.15, .07, .23, .15, .20, .20]),
        p(f"Static-control successes by seed {'/'.join(map(str, seeds))}: {static_seed_counts}. "
          "Each count is out of five initial states. These seed comparisons are descriptive; "
          "the uncertainty analysis remains conditional on the three selected seeds.", "small"),
        p("Timing definitions", "h2"),
        p("Compute time is synchronized inference plus simulator stepping. Episode wall time also includes "
          "reset, in-memory telemetry and video creation, recording and finalization; it excludes JSON "
          "log writing, checksums, model loading and warmup. It is not a real-time robot latency benchmark: the simulated target "
          "does not keep moving while the model is computing."),
        p("Interpret the cost and success columns together. Success can terminate an episode early, so a "
          "lower average cost can result from fewer calls per action, a shorter episode, or both. The fixed "
          "280-action time limit is shared across conditions."),
        p(f"Across the full stress test: {sum(row['success'] for row in rows)}/135 successes, "
          f"{sum(row['model_calls'] for row in rows):,} model calls, "
          f"{sum(row['steps'] for row in rows):,} executed actions and "
          f"{sum(row['episode_wall_s'] for row in rows) / 60:.1f} minutes of accumulated episode wall time. "
          "This total is descriptive and is not an aggregate benchmark score.", "small"),
        PageBreak(),
    ])

    # Page 11: paired contrasts answer the scientific question rather than raw rankings.
    story.extend([
        p("Does target motion change the effect of K?", "h1"),
        p("First compare each moving condition with its own static control at the same K. "
          "The difference is moving minus static success, expressed in percentage points (pp). "
          "A negative value indicates lower completion under the force intervention."),
        table(["Reference motion", "K", "Paired change (pp)", "Exploratory 95% interval"],
              [[row["condition"].capitalize(), row["k"], f"{100 * row['paired_difference']:+.1f}", interval(row)]
               for row in differences], [.24, .10, .27, .39]),
        p("Interaction relative to K=10", "h2"),
        p("Interaction = (moving - static at K) - (moving - static at K=10). "
          "A positive value means the motion-related change is more favorable at that K than at K=10. "
          "This contrast accounts for baseline performance differences while preserving paired starts."),
        table(["Reference motion", "K vs. 10", "Interaction (pp)", "Exploratory 95% interval"],
              [[row["condition"].capitalize(), row["k"], f"{100 * row['interaction']:+.1f}", interval(row)]
               for row in interactions], [.24, .13, .24, .39]),
        p("What the uncertainty calculation does", "h2"),
        p(f"The {stats['bootstrap']['draws']:,}-draw bootstrap resamples five initial states. Every sampled state "
          "retains all three policy seeds, conditions and K values. Seed repeats are averaged within a "
          "state; they are not treated as additional independent scenes. All contrasts share the same "
          f"state draws (analysis seed {stats['bootstrap']['seed']})."),
        p("With only five state units, uncertainty estimates are weak and can be degenerate for identical "
          "outcomes. Intervals are conditional on the chosen seed set, are not adjusted for multiple "
          "comparisons and do not establish equivalence when they include zero. The interaction also "
          "cannot remove success floor/ceiling constraints or the dependence of exposure duration on contact."),
        p("Fixed reference phase also matters", "h2"),
        p(f"With the {motion.start_delay_s:g} s onset, the first scheduled replan strictly after onset is at "
          f"{first_after_onset[5]:g}, {first_after_onset[10]:g} and {first_after_onset[50]:g} s "
          "for K=5, 10 and 50, if the episode continues. " + phase_example + "The real object may lag. "
          "An oscillating target can therefore return toward an older planned grasp position: a shorter "
          "reference period need not be harder. This is a possible timing mechanism, not a verified "
          "explanation of an individual outcome."),
        p("Raw cell counts appear on the preceding page. Per-episode state/seed outcomes and all four cell "
          "counts used in each interaction are retained in the CSV exports.", "small"),
        PageBreak(),
    ])

    # Page 12: observed motion and contact, including all no-contact denominators.
    story.extend([
        p("Observed motion and contact", "h1"),
        p("The commanded reference does not guarantee an identical object trajectory. These measurements "
          "check motion actually achieved before the first finger contact.", "small"),
    ])
    story.extend(figure("motion-and-contact", 7 / 11,
        "Ranges and tilt are measured at physics-substep resolution. Mean speed is a control-rate "
        "finite-difference diagnostic. Exposure ends at first contact or episode termination.", scale=.95))
    contact_table = []
    for condition in CONDITIONS:
        for k in (5, 10, 50):
            cell = by_cell[condition, k]
            contact_table.append([condition.capitalize(), k, f"{cell['contacts']}/15",
                                  cell["no_contact_episodes"], num(cell["mean_contact_time_s_among_contacts"], 2)])
    story.extend([
        table(["Condition", "K", "Finger contact", "No contact", "Mean contact time (s)*"],
              contact_table, [.19, .08, .22, .18, .33], padding=3),
        p("*Conditional on an observed contact; no-contact episodes are excluded from this time mean "
          "and counted explicitly. A single finger brush can stop the force without securing the can. "
          "Contact is not a verified grasp, recovery event or successful placement.", "small"),
        p("All completed episodes passed the force-cap, fixed-K schedule and permanent force/moment release "
          "checks. Tracking error, force saturation, x/y/z ranges and exposure are retained in the raw "
          "telemetry and analysis exports.", "small"),
        PageBreak(),
    ])

    # Page 13: numeric interpretation with explicit scope and reproducible evidence.
    covered = sum(row["ci_low"] <= 0 <= row["ci_high"] for row in interactions)
    summaries = []
    for condition in ("slow", "fast"):
        values = [row for row in differences if row["condition"] == condition]
        summaries.append(f"{condition.capitalize()} reference: " + "; ".join(
            f"{100 * row['paired_difference']:+.1f} pp at K={row['k']}" for row in values) + ".")
    reviewed_fast = {item["k"]: item for item in review["episodes"] if item["condition"] == "fast"}
    case5, case10, case50 = (reviewed_fast[k] for k in (5, 10, 50))
    if case50["success"] is not False or case10["success"] is not True or case5["success"] is not True:
        raise ValueError("The authored visual illustration does not match the reviewed fast-condition outcomes")
    case50_steps = read_jsonl(args.run / "episodes" / case50["episode_id"] / "steps.jsonl")
    closures = [row["step"] for index, row in enumerate(case50_steps)
                if row["action"][6] > 0 and (index == 0 or case50_steps[index - 1]["action"][6] <= 0)
                and row["step"] >= case50["first_contact_step"]]
    if not closures:
        raise ValueError("The reviewed K=50 contact was not followed by the described closing command")
    story.extend([
        p("Interpretation and research contribution", "h1"),
        p("Measured changes in completion", "h2"),
        *[p(text) for text in summaries],
        p(f"The exploratory interval contains zero for {covered} of {len(interactions)} condition-by-K "
          "interaction contrasts. These results describe sensitivity to this particular force rule; "
          "they do not establish a universal optimal K or an isolated effect of object speed."),
        p("Contribution in the context of action chunking", "h2"),
        p("ACT (Zhao et al., 2023) predicts action sequences and combines overlapping predictions through "
          "temporal ensembling. Its chunk-length ablation trained separate policies for each length. "
          "Here the checkpoint and H=50 remain fixed while the executed prefix K changes [1]."),
        p("RTC (Black et al., 2025) generates the next chunk during execution, retaining actions already "
          "committed. Our simulator pauses during inference; this is a synchronous replanning study, "
          "not an evaluation of asynchronous latency robustness [2]."),
        p("The empirical contribution is a controlled comparison of execution intervals, coupled with "
          "a calibrated force intervention, matched starts, nested policy-seed repeats and paired "
          "condition-by-K contrasts. It studies one pretrained policy rather than introducing a new "
          "action-chunking algorithm."),
        p("Preselected visual illustration", "h2"),
        p(f"Assistant review of state 10/seed 11 shows fast K=50 contacting the can at "
          f"{case50['first_contact_sim_time_s']:.3f} s and commanding closure at step {closures[0]}, then transporting an empty "
          f"gripper toward the basket without retrieval through {case50['steps']} actions despite later replans. "
          f"Fast K=10 returns after its first unsuccessful closure and succeeds in {case10['steps']} actions/"
          f"{case10['model_calls']} calls; fast K=5 succeeds in {case5['steps']} actions/{case5['model_calls']} calls. "
          "This single illustration gives neither error rates nor a causal explanation.", "small"),
        p("Validation and limits", "h2"),
        p("All 135 logs matched saved predictions and fixed-K schedules; all 135 videos were decoded. "
          "Across 45 state/seed/K triplets (90 moving-versus-static comparisons), pre-motion executed "
          "actions and full predictions were exactly identical: maximum differences were zero."),
        p("These are instrumentation checks, not independent proof of physical success. Contact is not "
          "a grasp label; sampled assistant review leaves grasp/drop counts unresolved. Contact release "
          "changes exposure, and native success does not establish stable placement.", "small"),
        p("References and reproducible evidence", "h2"),
        p(f'1. Zhao et al. (2023). <link href="https://www.roboticsproceedings.org/rss19/p016.pdf" color="{TEAL}">'
          'Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.</link> RSS.', "small", markup=True),
        p(f'2. Black, Galliker and Levine (2025). <link href="https://arxiv.org/abs/2506.07339" color="{TEAL}">'
          'Real-Time Execution of Action Chunking Flow Policies.</link> arXiv:2506.07339.', "small", markup=True),
        p("Protocol: docs/moving-target-protocol.md. Statistics, action/video verification, sampled visual "
          "review and full provenance accompany the report. Rebuild with analyze_moving_target.py, "
          "verify_moving_target.py --decode-video and build_moving_target_report.py.", "small"),
    ])

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D4E0E8"))
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont("MovingReport", 7.3)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, height - 11 * mm, "SMOLVLA / LIBERO | MOVING TARGET | 135 MATCHED EPISODES")
        canvas.drawString(18 * mm, 11 * mm, "5 initial states x 3 policy seeds x 3 K values x 3 conditions")
        canvas.drawRightString(width - 18 * mm, 11 * mm, str(document.page + 8))
        canvas.restoreState()

    temp_root = Path("tmp/pdfs").resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moving-report-", dir=temp_root) as temporary:
        supplement = Path(temporary) / "moving-target-pages.pdf"
        document = SimpleDocTemplate(str(supplement), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=21 * mm, bottomMargin=20 * mm, title="Moving-target replanning experiment",
            author="VLA replanning experiment", subject="135 matched moving-target stress-test episodes")
        document.build(story, onFirstPage=decorate, onLaterPages=decorate)
        supplement_reader = PdfReader(supplement)
        if len(supplement_reader.pages) != 5 or any(not page.extract_text().strip() for page in supplement_reader.pages):
            raise ValueError(f"Expected five nonempty new pages, got {len(supplement_reader.pages)}; revise layout")
        baseline_reader = PdfReader(args.baseline)
        if len(baseline_reader.pages) != 8:
            raise ValueError("The source study PDF must contain exactly eight pages")
        original_content = [page_content_hash(page) for page in baseline_reader.pages]
        writer = PdfWriter()
        writer.append(baseline_reader)
        writer.append(supplement_reader)
        writer.add_metadata({"/Title": "How often should a robot replan?",
            "/Author": "VLA replanning experiment", "/Subject": "Standard manipulation study and moving-target stress test"})
        staged = Path(temporary) / output.name
        with staged.open("wb") as stream:
            writer.write(stream)
        merged = PdfReader(staged)
        copied_content = [page_content_hash(page) for page in merged.pages[:8]]
        if len(merged.pages) != 13 or original_content != copied_content:
            raise ValueError("Merge changed the original page content or produced the wrong page count")
        if any(not page.extract_text().strip() for page in merged.pages):
            raise ValueError("Merged report contains an empty page")
        record = {
            "schema_version": 1, "artifact_kind": "complete_replanning_report", "pages": 13,
            "standard_study_pages": 8, "moving_target_pages": 5,
            "episode_counts": {"standard_manipulation": 150, "moving_target_stress_test": 135},
            "outcomes_pooled": False, "pdf_sha256": sha256(staged),
            "baseline_pdf_sha256": snapshots[args.baseline.resolve()],
            "baseline_record_sha256": snapshots[args.baseline.resolve().with_suffix('.json')],
            "baseline_page_content_sha256": original_content,
            "copied_baseline_page_content_sha256": copied_content,
            "statistics_sha256": snapshots[args.analysis.resolve() / "statistics.json"],
            "figures_record_sha256": snapshots[artifacts_path],
            "evidence_verification_sha256": snapshots[args.analysis.resolve() / "evidence-verification.json"],
            "demo_review_sha256": snapshots[args.analysis.resolve() / "demo-review.json"],
            "verifier_source_sha256": snapshots[Path(__file__).with_name("verify_moving_target.py").resolve()],
            "calibration_sha256": snapshots[args.calibration.resolve()],
            "frozen_protocol_sha256": snapshots[frozen_path],
            "protocol_sha256": snapshots[args.protocol.resolve()],
            "generator_sha256": snapshots[Path(__file__).resolve()],
            "input_files": {str(path): value for path, value in snapshots.items()},
            "visual_review_required": True,
        }
        verify_snapshots(snapshots)
        os.replace(staged, output)
        atomic_json(record_path, record)
    print(json.dumps({"output": str(output), "provenance": str(record_path), "pages": 13,
                      "original_eight_pages_unchanged": True, "visual_review_required": True}, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/moving-target"))
    parser.add_argument("--analysis", type=Path, default=Path("outputs/moving-target-report"))
    parser.add_argument("--baseline", type=Path, default=Path("output/pdf/vla-replanning-report.pdf"))
    parser.add_argument("--calibration", type=Path, default=Path("outputs/moving-calibration/physics-b/calibration.json"))
    parser.add_argument("--frozen-protocol", type=Path, default=Path("outputs/moving-calibration/frozen-protocol.json"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/moving-target-protocol.md"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/vla-replanning-complete-report.pdf"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-bold", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    build_report(parser.parse_args())


if __name__ == "__main__":
    main()
