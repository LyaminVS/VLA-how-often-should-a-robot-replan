"""Package verified sources, 150 standard episodes and 135 moving-target episodes.

Run on the Linux experiment host after analysis, rendering and visual review. The
two experimental designs remain distinct; their success outcomes are not pooled.
The archive excludes environments, model weights, locks, failed attempts and ZIPs.
"""

import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.package_expanded_submission import collect_inputs as collect_standard  # noqa: E402
from scripts.package_expanded_submission import validate_pdf  # noqa: E402
from scripts.package_submission import CHUNK_BYTES, archive_info, required_file, sha256_file  # noqa: E402

BASELINE_PDF_SHA256 = "694e08619bbd91583fdeafc1a7a157354801d4bd115df9b7f29709a124fcd0b8"
BASELINE_CODE_SHA256 = "d96cfeded0dc6e1ca630d2fbdc9f23e1cdf51806bfbe14211327200da412902e"
DEPENDENCY_LOCK_SHA256 = "8574b51f0fad0b6ec40c02b3c6192da4606e30a94fd8dc1e18068b461d81212a"
REPORT_SUFFIXES = {".png", ".pdf", ".csv", ".md", ".json", ".mp4"}
FORBIDDEN_PARTS = {".git", ".venv", "__pycache__", "attempts", "cache", ".cache", "models", "tmp"}
ANALYSIS_FILES = {
    "success-and-cost.png", "success-and-cost.pdf", "motion-and-contact.png",
    "motion-and-contact.pdf", "report.md", "episodes.csv", "summary.csv",
    "paired-vs-static.csv", "condition-by-k-interactions.csv", "contact-coverage.csv",
}


def read_json(path):
    result = json.loads(required_file(Path(path)).read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def relative_path(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"Invalid relative archive path: {name!r}")
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "." in path.parts
            or any(part in FORBIDDEN_PARTS or part.startswith(".venv") for part in path.parts)
            or any(part.startswith(".") for part in path.parts)
            or path.suffix.lower() == ".zip"):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def compare_fields(record, expected, label):
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{label} disagrees with verified evidence: {key}")


def baseline_code_hash():
    result = hashlib.sha256()
    for source in sorted((PROJECT / "vla_replan").glob("*.py")):
        result.update(source.name.encode())
        result.update(required_file(source).read_bytes())
    return result.hexdigest()


def verify_moving_statistics(config, rows, provenance, report):
    from scripts.analyze_moving_target import (
        DATA_KIND, cluster_statistics, contact_and_motion_summary, interaction_statistics,
    )

    statistics = read_json(report / "statistics.json")
    compare_fields(statistics, {
        "schema_version": 1, "data_kind": DATA_KIND, "complete": True, "episodes": 135,
        "config": canonical(config.asdict()), "intervals": [5, 10, 50],
        "bootstrap": {"draws": 5000, "seed": 20261024, "unit": "initial_state",
                      "policy_seeds_retained_in_cluster": True},
        "analysis_source_sha256": sha256_file(PROJECT / "scripts/analyze_moving_target.py"),
    }, "Moving statistics")
    compare_fields(statistics.get("provenance", {}), {
        key: provenance[key] for key in
        ("manifest_sha256", "config_fingerprint", "runtime", "hardware", "sources", "episodes")
    }, "Moving statistics provenance")
    estimates, comparisons = cluster_statistics(rows, [5, 10, 50], draws=5000, seed=20261024)
    compare_fields(statistics, {
        "estimates": estimates, "paired_vs_static": comparisons,
        "condition_by_k_interactions": interaction_statistics(rows, [5, 10, 50], draws=5000, seed=20261024),
        "contact_and_motion": contact_and_motion_summary(rows, [5, 10, 50]),
    }, "Recomputed moving statistics")
    return statistics


def verify_demo(report, rows, provenance, config):
    from scripts.build_moving_target_demo import FPS, HEIGHT, POLICY_SEED, STATE, WIDTH, inspect_video

    record = read_json(report / "demo.json")
    lookup = {(row["state"], row["policy_seed"], row["condition"], row["k"]): row for row in rows}
    sections = [
        [lookup[(STATE, POLICY_SEED, condition, 10)] for condition in ("static", "slow", "fast")],
        [lookup[(STATE, POLICY_SEED, "fast", k)] for k in (5, 10, 50)],
    ]
    identities = sorted({row["episode_id"] for section in sections for row in section})
    sources = {item["episode_id"]: item for item in provenance["episodes"]}
    frames = 8 * FPS + sum(max(row["steps"] for row in section) + FPS for section in sections)
    counts = []
    for condition in ("static", "slow", "fast"):
        for k in (5, 10, 50):
            group = [row for row in rows if row["condition"] == condition and row["k"] == k]
            counts.append({
                "condition": condition, "k": k, "successes": sum(row["success"] for row in group),
                "episodes": len(group), "mean_model_calls": sum(row["model_calls"] for row in group) / len(group),
            })
    compare_fields(record, {
        "schema_version": 1, "artifact_kind": "moving_target_demo",
        "data_kind": "moving_target_stress_test", "episodes_in_results": 135,
        "config_fingerprint": config.fingerprint, "manifest_sha256": provenance["manifest_sha256"],
        "statistics_sha256": sha256_file(report / "statistics.json"),
        "generator_sha256": sha256_file(PROJECT / "scripts/build_moving_target_demo.py"),
        "analysis_source_sha256": sha256_file(PROJECT / "scripts/analyze_moving_target.py"),
        "video_sha256": sha256_file(required_file(report / "demo.mp4")),
        "selected_episodes": [sources[identity] for identity in identities],
        "illustration_selection": {"state": STATE, "policy_seed": POLICY_SEED,
                                   "rule": "Fixed before collection; no selection by outcome"},
        "full_results_table": counts, "frames": frames, "fps": FPS,
        "size": [WIDTH, HEIGHT], "duration_s": frames / FPS, "audio": False,
    }, "Moving demo")
    timeline = record.get("timeline", [])
    if [item.get("kind") for item in timeline] != ["intro", "comparison", "divider", "comparison", "results"]:
        raise ValueError("Unexpected moving demo timeline")
    cursor = 0
    for item in timeline:
        duration = item.get("frames")
        if item.get("start_frame") != cursor or type(duration) is not int or duration <= 0:
            raise ValueError("Moving demo timeline has gaps or overlaps")
        cursor += duration
    comparisons = [item for item in timeline if item["kind"] == "comparison"]
    if cursor != frames or [item.get("episode_ids") for item in comparisons] != [
        [row["episode_id"] for row in section] for section in sections
    ]:
        raise ValueError("Moving demo timeline does not match its verified cases")
    inspect_video(report / "demo.mp4", (WIDTH, HEIGHT), frames)
    return record


def verify_saved_evidence(raw, report, calibration, config, provenance, manifest):
    """Bind the full-video-decode record to current raw data and recheck action logs.

    Decoding 135 videos is intentionally not repeated here. The saved successful
    decode record must cover the exact unchanged video checksums loaded above.
    Action execution and all 45 pre-motion triplets are recomputed from raw logs.
    """
    from scripts.analyze_moving_target import read_jsonl
    from scripts.verify_moving_target import (
        compare_prefixes, parsed_utc, pre_motion_prefix, verify_action_log,
        verify_force_timing, verify_frozen,
    )

    path = report / "evidence-verification.json"
    record = read_json(path)
    base = config.base_config
    compare_fields(record, {
        "schema_version": 1, "artifact_kind": "moving_target_evidence_verification",
        "data_kind": "moving_target_stress_test", "all_checks_passed": True,
        "config_fingerprint": config.fingerprint, "verified_episodes": 135,
        "prefix_pairs": 45, "condition_vs_static_prefix_comparisons": 90,
        "decoded_videos": 135, "video_verification_mode": "decoded_all_frames",
        "verifier_source_sha256": sha256_file(PROJECT / "scripts/verify_moving_target.py"),
        "analysis_source_sha256": sha256_file(PROJECT / "scripts/analyze_moving_target.py"),
        "prefix_block_definition": "Matched static/slow/fast triplet for one state, policy seed and K",
        "max_executed_action_error": 0.0, "max_action_prefix_error": 0.0,
        "max_prediction_prefix_error": 0.0,
        "checks": {
            "complete_plan_and_raw_checksums": True, "current_sources_match_frozen_run": True,
            "freeze_precedes_main_creation": True, "physics_b_bound_to_freeze": True,
            "full_50_by_7_chunks_finite": True, "executed_actions_equal_clipped_prefix": True,
            "sampling_seeds_and_fixed_k_schedule": True, "success_and_gripper_bookkeeping": True,
            "zero_force_before_onset_static_and_after_contact": True,
            "identical_pre_motion_actions_and_full_predictions": True, "all_video_frames_decoded": True,
        },
    }, "Independent evidence verification")
    if record["all_checks_passed"] is not True or any(value is not True for value in record["checks"].values()):
        raise ValueError("Independent verification check flags must be boolean true")
    compare_fields(record.get("provenance", {}), {
        key: provenance[key] for key in
        ("manifest_sha256", "config_fingerprint", "runtime", "hardware", "sources", "episodes")
    }, "Independent verification provenance")
    frozen_evidence = verify_frozen(config, manifest, calibration / "frozen-protocol.json")
    if record.get("frozen_evidence") != frozen_evidence:
        raise ValueError("Independent verification no longer matches the frozen/current inputs")
    if parsed_utc(record.get("verified_utc"), "verification") < parsed_utc(manifest.get("created_utc"), "main manifest"):
        raise ValueError("Verification predates main collection")

    totals = {"verified_episodes": 0, "steps": 0, "model_calls": 0,
              "predicted_action_values": 0, "executed_action_values": 0}
    prefixes = {}
    for item in provenance["episodes"]:
        directory = raw / "episodes" / item["episode_id"]
        summary = read_json(directory / "summary.json")
        rows, chunks = read_jsonl(directory / "steps.jsonl"), read_jsonl(directory / "chunks.jsonl")
        actions = verify_action_log(summary, rows, chunks, base.chunk_size, base.control_hz)
        verify_force_timing(summary, rows, config.motion.start_delay_s,
                            config.motion.force_application_offset_z_m, config.motion.max_force_n)
        key = summary["state"], summary["policy_seed"], summary["k"]
        treatments = prefixes.setdefault(key, {})
        if summary["condition"] in treatments:
            raise ValueError("Repeated treatment in independent prefix verification")
        treatments[summary["condition"]] = pre_motion_prefix(
            summary, rows, chunks, config.motion.start_delay_s, base.control_hz,
        )
        for field in ("steps", "model_calls", "predicted_action_values", "executed_action_values"):
            totals[field] += actions[field]
        totals["verified_episodes"] += 1
    prefix_checks = [
        {"state": state, "policy_seed": policy_seed, "k": k, **compare_prefixes(treatments)}
        for (state, policy_seed, k), treatments in sorted(prefixes.items())
    ]
    expected_keys = {(state, seed, k) for state in base.initial_states
                     for seed in base.policy_seeds for k in base.intervals}
    if set(prefixes) != expected_keys or len(prefix_checks) != 45:
        raise ValueError("Independent verification is missing a registered prefix triplet")
    # Each raw recording contains every pre-action frame plus a one-second
    # terminal hold; this is the decoder's exact expected total, not an estimate.
    totals["decoded_frames"] = totals["steps"] + totals["verified_episodes"] * base.control_hz
    compare_fields(record, {**totals, "prefix_checks": prefix_checks,
        "prefix_action_steps": list(range(10)),
    }, "Recomputed independent verification")
    return record


def verify_demo_review(report, rows, provenance):
    """Validate the prespecified five-case visual review against immutable records."""
    record = read_json(report / "demo-review.json")
    compare_fields(record, {
        "schema_version": 1, "role": "manual_assistant_visual_review", "reviewer": "assistant",
        "independent_human_review": False, "state": 10, "policy_seed": 11,
        "selection_fixed_before_selected_outcomes": True, "cameras": ["agentview", "eye_in_hand"],
    }, "Manual demonstration review")
    if (record["independent_human_review"] is not False
            or record["selection_fixed_before_selected_outcomes"] is not True):
        raise ValueError("Manual review must accurately label selection and reviewer independence")
    for field in ("selection_rule", "frame_indexing", "review_method", "overall_conclusion"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Manual demonstration review needs {field}")
    limits = record.get("limitations")
    if not isinstance(limits, list) or not limits or any(not isinstance(item, str) or not item.strip() for item in limits):
        raise ValueError("Manual review needs explicit limitations")
    expected_selection = {("static", 10), ("slow", 10), ("fast", 10), ("fast", 5), ("fast", 50)}
    by_key = {(row["condition"], row["k"]): row for row in rows
              if row["state"] == 10 and row["policy_seed"] == 11 and (row["condition"], row["k"]) in expected_selection}
    sources = {item["episode_id"]: item for item in provenance["episodes"]}
    episodes = record.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 5:
        raise ValueError("Manual review must cover exactly five preselected episodes")
    seen = set()
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("Manual review episode must be an object")
        key = item.get("condition"), item.get("k")
        if key in seen or key not in expected_selection or key not in by_key:
            raise ValueError("Manual review has a duplicate or unselected episode")
        seen.add(key)
        expected = by_key[key]
        source = sources[expected["episode_id"]]
        compare_fields(item, {
            **{field: expected[field] for field in (
                "episode_id", "condition", "k", "state", "policy_seed", "steps", "model_calls", "success",
            )},
            "raw_files": source["files"], "evidence_sha256": source["evidence_sha256"],
            "first_contact_step": expected["first_contact_step"],
            "first_contact_sim_time_s": expected["first_contact_sim_time_s"],
            "missed_grasps": None, "dropped_objects": None,
        }, f"Manual review episode {expected['episode_id']}")
        if type(item["success"]) is not bool:
            raise ValueError("Manual review outcome must be boolean")
        frames = item.get("inspected_frame_indices")
        if (not isinstance(frames, list) or not frames
                or any(type(frame) is not int or not 0 <= frame <= expected["steps"] for frame in frames)
                or frames != sorted(set(frames)) or frames[0] != 0 or frames[-1] != expected["steps"]):
            raise ValueError("Manual review frame indices must be unique, ordered and within the raw episode")
        windows = item.get("inspected_windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("Manual review must describe its inspected windows")
        covered, kinds = set(), set()
        for window in windows:
            if not isinstance(window, dict):
                raise ValueError("Manual review window must be an object")
            start, end, stride = (window.get(field) for field in ("start_frame", "end_frame", "stride"))
            if (any(type(value) is not int for value in (start, end, stride))
                    or not 0 <= start <= end <= expected["steps"] or stride < 1):
                raise ValueError("Invalid manual review inspection window")
            extra = window.get("additional_frames", [])
            if not isinstance(extra, list) or any(type(frame) is not int or not 0 <= frame <= expected["steps"] for frame in extra):
                raise ValueError("Invalid additional reviewed frame")
            covered.update(range(start, end + 1, stride))
            covered.update(extra)
            kinds.add(window.get("kind"))
        if set(frames) != covered or not {"whole_trajectory_coarse", "first_contact_neighborhood"}.issubset(kinds):
            raise ValueError("Manual review windows do not match the declared inspected frames")
        observations = item.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("Manual review needs observable behavior for every selected episode")
        for observation in observations:
            if not isinstance(observation, dict) or not isinstance(observation.get("text"), str) or not observation["text"].strip():
                raise ValueError("Manual review observation needs text")
            cited = observation.get("frames")
            if not isinstance(cited, list) or not cited or any(type(frame) is not int or frame not in covered for frame in cited):
                raise ValueError("Manual review observation cites an uninspected frame")
            times = observation.get("sim_time_s")
            if (not isinstance(times, list) or len(times) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in times)
                    or not math.isclose(times[0], min(cited) / 20, abs_tol=1e-9, rel_tol=0)
                    or not math.isclose(times[1], max(cited) / 20, abs_tol=1e-9, rel_tol=0)):
                raise ValueError("Manual review observation timestamps disagree with its frame indices")
        uncertainties = item.get("uncertainties")
        if not isinstance(uncertainties, list) or not uncertainties or any(not isinstance(text, str) or not text.strip() for text in uncertainties):
            raise ValueError("Manual review needs uncertainties for every selected episode")
    if seen != expected_selection:
        raise ValueError("Manual review omits a preselected demonstration condition")
    return record


def page_hashes(path):
    from pypdf import PdfReader

    result = []
    for page in PdfReader(str(path)).pages:
        contents = page.get_contents()
        result.append(hashlib.sha256(contents.get_data() if contents else b"").hexdigest())
    return result


def collect_inputs(args):
    from moving_target.runner import source_metadata
    from scripts.analyze_moving_target import load_run

    baseline = args.baseline_pdf.resolve()
    if sha256_file(required_file(baseline)) != BASELINE_PDF_SHA256:
        raise ValueError("The preserved eight-page baseline PDF changed")
    if baseline_code_hash() != BASELINE_CODE_SHA256:
        raise ValueError("The frozen baseline executor or adapter changed")
    if sha256_file(PROJECT / "requirements/linux.lock.txt") != DEPENDENCY_LOCK_SHA256:
        raise ValueError("The frozen rollout dependency lock changed")
    files, expected_hashes, standard = collect_standard(SimpleNamespace(
        run=args.standard, pilot=args.compact, validation=args.validation, pdf=baseline,
    ))
    compare_fields(standard, {
        "unique_episode_count": 150, "code_sha256": BASELINE_CODE_SHA256,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
    }, "Standard-study delivery")

    def add(source, destination, expected=None):
        source = required_file(Path(source).absolute())
        if source.resolve() != source or not source.is_relative_to(PROJECT):
            raise ValueError(f"Delivery input must be a regular project file: {source}")
        destination = str(relative_path(str(destination)))
        checksum = sha256_file(source)
        if expected is not None and checksum != expected:
            raise ValueError(f"Input changed since its evidence was recorded: {source}")
        if destination in files:
            if files[destination] != source or expected_hashes[destination] != checksum:
                raise ValueError(f"Conflicting duplicate archive destination: {destination}")
            return
        files[destination], expected_hashes[destination] = source, checksum

    raw, report, calibration = args.moving.resolve(), args.report.resolve(), args.calibration.resolve()
    for directory in (raw, report, calibration):
        if not directory.is_relative_to(PROJECT) or not directory.is_dir():
            raise ValueError(f"Expected a project directory: {directory}")
    config, rows, provenance = load_run(raw)
    base = config.base_config
    if (len(rows) != 135 or tuple(sorted(base.initial_states)) != (10, 11, 12, 13, 14)
            or tuple(sorted(base.policy_seeds)) != (11, 22, 33)
            or tuple(sorted(base.intervals)) != (5, 10, 50) or not base.record_video):
        raise ValueError("Delivery requires the complete registered 135-episode moving-target study")
    current_sources = source_metadata()
    if provenance["sources"] != current_sources:
        raise ValueError("Moving-target implementation changed after collection")
    compare_fields(provenance["runtime"], {
        "code_sha256": BASELINE_CODE_SHA256, "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
    }, "Moving runtime")
    for name, checksum in current_sources["files"].items():
        add(PROJECT / relative_path(name), name, checksum)

    frozen_path = calibration / "frozen-protocol.json"
    frozen = read_json(frozen_path)
    manifest = read_json(raw / "manifest.json")
    compare_fields(frozen, {
        "schema_version": 1, "main_config": canonical(config.asdict()),
        "config_fingerprint": config.fingerprint, "planned_episode_ids": manifest["planned_episode_ids"],
        "sources": current_sources, "baseline_runtime": provenance["runtime"],
        "protocol_sha256": sha256_file(PROJECT / "docs/moving-target-protocol.md"),
    }, "Frozen motion protocol")
    if datetime.fromisoformat(frozen["created_utc"]) > datetime.fromisoformat(manifest["created_utc"]):
        raise ValueError("The motion protocol was frozen after main collection began")
    compare_fields(frozen.get("model_checks", {}), {
        "episodes": 9, "contacts": 9, "all_intervention_and_schedule_checks_passed": True,
    }, "Pre-collection integrity checks")
    calibration_files = frozen.get("calibration_files")
    if not isinstance(calibration_files, dict) or not calibration_files:
        raise ValueError("Frozen protocol has no calibration evidence inventory")
    for name, checksum in calibration_files.items():
        relative = relative_path(name)
        if (relative.parts[0] not in {"physics-a", "physics-b", "model-check"}
                or relative.suffix not in {".json", ".jsonl", ".mp4"}):
            raise ValueError(f"Unexpected frozen calibration payload: {name}")
        add(calibration / relative, PurePosixPath("outputs/moving-calibration") / relative, checksum)
    for required in ("physics-a/calibration.json", "physics-b/calibration.json", "model-check/manifest.json"):
        if required not in calibration_files:
            raise ValueError(f"Missing frozen calibration record: {required}")
    observed_calibration = set()
    for phase in ("physics-a", "physics-b", "model-check"):
        for path in (calibration / phase).rglob("*"):
            relative = path.relative_to(calibration)
            if (path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".mp4"}
                    and not any(part in FORBIDDEN_PARTS or part.startswith(".") for part in relative.parts)):
                observed_calibration.add(relative.as_posix())
    if observed_calibration != set(calibration_files):
        raise ValueError("Calibration payload inventory changed after protocol freezing")
    add(frozen_path, "outputs/moving-calibration/frozen-protocol.json")
    calibrated = read_json(calibration / "physics-b/calibration.json")
    compare_fields(calibrated, {"state": 20, "steps": 240, "motion": canonical(config.motion.asdict())},
                   "Physical calibration")
    checked_config, checked_rows, checked_provenance = load_run(calibration / "model-check")
    if (len(checked_rows) != 9 or checked_config.base_config.initial_states != (20,)
            or checked_config.motion.asdict() != config.motion.asdict()
            or checked_provenance["sources"] != current_sources
            or checked_provenance["runtime"] != provenance["runtime"]
            or not all(row["contact"] for row in checked_rows)):
        raise ValueError("The preserved policy integrity check does not match the fixed controller")

    verify_moving_statistics(config, rows, provenance, report)
    add(raw / "manifest.json", "outputs/moving-target/manifest.json", provenance["manifest_sha256"])
    for item in provenance["episodes"]:
        directory = PurePosixPath("episodes") / relative_path(item["episode_id"])
        for name, checksum in item["files"].items():
            relative = directory / relative_path(name)
            add(raw / relative, PurePosixPath("outputs/moving-target") / relative, checksum)
        add(raw / directory / "evidence.json", PurePosixPath("outputs/moving-target") / directory / "evidence.json",
            item["evidence_sha256"])
    verification = verify_saved_evidence(raw, report, calibration, config, provenance, manifest)
    add(report / "evidence-verification.json", "outputs/moving-target-report/evidence-verification.json")

    artifacts = read_json(report / "artifacts.json")
    compare_fields(artifacts, {
        "schema_version": 1, "data_kind": "moving_target_stress_test",
        "statistics_sha256": sha256_file(report / "statistics.json"),
        "analysis_source_sha256": sha256_file(PROJECT / "scripts/analyze_moving_target.py"),
    }, "Moving report artifact record")
    if not isinstance(artifacts.get("files"), dict) or set(artifacts["files"]) != ANALYSIS_FILES:
        raise ValueError("Moving analysis requires the full report, plots and CSV inventory")
    for name, checksum in artifacts["files"].items():
        add(report / relative_path(name), PurePosixPath("outputs/moving-target-report") / name, checksum)
    for name in ("statistics.json", "artifacts.json"):
        add(report / name, PurePosixPath("outputs/moving-target-report") / name)
    demo = verify_demo(report, rows, provenance, config)
    for name in ("demo.mp4", "demo.json"):
        add(report / name, PurePosixPath("outputs/moving-target-report") / name)
    manual_review = verify_demo_review(report, rows, provenance)
    add(report / "demo-review.json", "outputs/moving-target-report/demo-review.json")

    complete = args.pdf.resolve()
    if complete == baseline or validate_pdf(complete) != 13:
        raise ValueError("Complete report must be a distinct thirteen-page PDF")
    complete_record_path = complete.with_suffix(".json")
    complete_record = read_json(complete_record_path)
    baseline_pages, complete_pages = page_hashes(baseline), page_hashes(complete)
    if complete_pages[:8] != baseline_pages:
        raise ValueError("The first eight pages differ from the preserved baseline report")
    compare_fields(complete_record, {
        "schema_version": 1, "artifact_kind": "complete_replanning_report", "pages": 13,
        "standard_study_pages": 8, "moving_target_pages": 5,
        "episode_counts": {"standard_manipulation": 150, "moving_target_stress_test": 135},
        "outcomes_pooled": False, "pdf_sha256": sha256_file(complete),
        "baseline_pdf_sha256": BASELINE_PDF_SHA256,
        "baseline_record_sha256": sha256_file(baseline.with_suffix(".json")),
        "baseline_page_content_sha256": baseline_pages, "copied_baseline_page_content_sha256": baseline_pages,
        "statistics_sha256": sha256_file(report / "statistics.json"),
        "figures_record_sha256": sha256_file(report / "artifacts.json"),
        "calibration_sha256": sha256_file(calibration / "physics-b/calibration.json"),
        "frozen_protocol_sha256": sha256_file(frozen_path), "protocol_sha256": frozen["protocol_sha256"],
        "generator_sha256": sha256_file(PROJECT / "scripts/build_moving_target_report.py"),
        "evidence_verification_sha256": sha256_file(report / "evidence-verification.json"),
        "verifier_source_sha256": sha256_file(PROJECT / "scripts/verify_moving_target.py"),
        "demo_review_sha256": sha256_file(report / "demo-review.json"),
    }, "Complete PDF provenance")
    input_files = complete_record.get("input_files")
    if not isinstance(input_files, dict) or not input_files:
        raise ValueError("Complete PDF has no input evidence inventory")
    # Absolute recorded paths may originate on the experiment host; match only
    # explicitly approved archive destinations rather than reading arbitrary paths.
    for recorded_path, checksum in input_files.items():
        matches = [name for name in files if recorded_path == name or recorded_path.endswith("/" + name)]
        if len(matches) != 1 or expected_hashes[matches[0]] != checksum:
            raise ValueError(f"Complete PDF input is missing or changed: {recorded_path}")
    add(complete, "output/pdf/vla-replanning-complete-report.pdf")
    add(complete_record_path, "output/pdf/vla-replanning-complete-report.json")

    qa_path = report / "visual-qa.json"
    qa = read_json(qa_path)
    compare_fields(qa, {"schema_version": 1, "approved": True}, "Final visual review")
    qa_files = qa.get("files")
    mandatory_review = {
        "output/pdf/vla-replanning-complete-report.pdf", "outputs/moving-target-report/demo.mp4",
        "outputs/moving-target-report/success-and-cost.png", "outputs/moving-target-report/motion-and-contact.png",
    }
    if not isinstance(qa_files, dict) or not mandatory_review.issubset(qa_files):
        raise ValueError("Visual review must cover the complete PDF, moving demo and both plots")
    for name, checksum in qa_files.items():
        relative_path(name)
        if expected_hashes.get(name) != checksum:
            raise ValueError(f"Visual review refers to a missing or changed file: {name}")
    add(qa_path, "outputs/moving-target-report/visual-qa.json")
    allowed_report = ANALYSIS_FILES | {
        "statistics.json", "artifacts.json", "demo.mp4", "demo.json", "visual-qa.json", "evidence-verification.json",
        "demo-review.json",
    }
    for path in report.rglob("*"):
        if path.is_file() and path.suffix.lower() in REPORT_SUFFIXES:
            if path.relative_to(report).as_posix() not in allowed_report:
                raise ValueError(f"Unverified presentation artifact in report directory: {path}")

    delivery = {
        "schema_version": 1, "kind": "complete_replanning_project_delivery",
        "research_episode_count": 285, "outcomes_pooled": False,
        "studies": {
            "standard_manipulation": {
                "episodes": 150, "tasks": 3, "initial_states_per_task": 10,
                "policy_seeds": [11], "intervals": [1, 5, 10, 25, 50],
                "statistics": "outputs/expanded/report/statistics.json",
                "statistics_sha256": standard["derived_report_sha256"],
                "source_manifest_sha256": standard["source_manifest_sha256"],
            },
            "moving_target_stress_test": {
                "episodes": 135, "tasks": 1, "initial_states": [10, 11, 12, 13, 14],
                "policy_seeds": [11, 22, 33], "intervals": [5, 10, 50],
                "conditions": ["static", "slow", "fast"],
                "statistics": "outputs/moving-target-report/statistics.json",
                "statistics_sha256": artifacts["statistics_sha256"],
                "manifest_sha256": provenance["manifest_sha256"],
                "config_fingerprint": config.fingerprint, "sources_sha256": current_sources["sha256"],
            },
        },
        "calibration": {"excluded_from_research_episode_count": True, "state": 20,
                        "policy_integrity_episodes": 9, "frozen_protocol_sha256": sha256_file(frozen_path)},
        "complete_report": "output/pdf/vla-replanning-complete-report.pdf",
        "complete_report_sha256": complete_record["pdf_sha256"],
        "baseline_report_sha256": BASELINE_PDF_SHA256,
        "moving_demo_sha256": demo["video_sha256"], "visual_qa_sha256": sha256_file(qa_path),
        "evidence_verification_sha256": expected_hashes["outputs/moving-target-report/evidence-verification.json"],
        "demo_review_sha256": expected_hashes["outputs/moving-target-report/demo-review.json"],
        "demo_review": {"episodes": len(manual_review["episodes"]), "reviewer": "assistant",
                        "independent_human_review": False, "selection_fixed_before_selected_outcomes": True},
        "independent_verification": {
            "verified_episodes": verification["verified_episodes"],
            "prefix_triplets": verification["prefix_pairs"],
            "condition_vs_static_prefix_comparisons": verification["condition_vs_static_prefix_comparisons"],
            "decoded_videos": verification["decoded_videos"],
            "decoded_frames": verification["decoded_frames"],
            "all_checks_passed": True,
        },
        "baseline_code_sha256": BASELINE_CODE_SHA256, "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "packager_sha256": sha256_file(Path(__file__)),
        "note": "The archive preserves original raw evidence and reports two distinct experimental designs. No pooled success rate is implied.",
    }
    return files, expected_hashes, delivery


def package_submission(args):
    output = args.output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must have the .zip extension")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    files, expected_hashes, delivery = collect_inputs(args)
    if output in files.values():
        raise ValueError("The archive must not replace a source artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".complete-submission-", dir=output.parent) as temporary:
        staged = Path(temporary) / "submission.zip"
        checksums = []
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for name, source in sorted(files.items()):
                checksum = hashlib.sha256()
                before = source.stat()
                with source.open("rb") as reader, archive.open(archive_info(name, source), "w", force_zip64=True) as writer:
                    while chunk := reader.read(CHUNK_BYTES):
                        checksum.update(chunk)
                        writer.write(chunk)
                after = source.stat()
                if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                        or checksum.hexdigest() != expected_hashes[name]):
                    raise RuntimeError(f"Input changed during packaging: {source}")
                checksums.append(f"{checksum.hexdigest()}  {name}\n")
            payload = (json.dumps(delivery, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
            archive.writestr(archive_info("DELIVERY.json"), payload)
            checksums.append(f"{hashlib.sha256(payload).hexdigest()}  DELIVERY.json\n")
            archive.writestr(archive_info("SHA256SUMS.txt"), "".join(checksums).encode())
        with zipfile.ZipFile(staged) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP CRC verification failed: {bad}")
            names = archive.namelist()
            if len(names) != len(files) + 2 or len(names) != len(set(names)):
                raise RuntimeError("Duplicate or unexpected archive members")
        checksum, size = sha256_file(staged), staged.stat().st_size
        if args.overwrite:
            staged.replace(output)
        else:
            output.hardlink_to(staged)
    result = {"output": str(output), "files": len(files) + 2, "bytes": size, "sha256": checksum,
              "standard_episodes": 150, "moving_target_episodes": 135,
              "research_episodes": 285, "outcomes_pooled": False, "crc_verified": True}
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--compact", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--baseline-pdf", type=Path, default=Path("output/pdf/vla-replanning-report.pdf"))
    parser.add_argument("--moving", type=Path, default=Path("outputs/moving-target"))
    parser.add_argument("--report", type=Path, default=Path("outputs/moving-target-report"))
    parser.add_argument("--calibration", type=Path, default=Path("outputs/moving-calibration"))
    parser.add_argument("--pdf", type=Path, default=Path("output/pdf/vla-replanning-complete-report.pdf"))
    parser.add_argument("--output", type=Path, default=Path("output/submission/vla-replanning-complete-project.zip"))
    parser.add_argument("--overwrite", action="store_true")
    package_submission(parser.parse_args(argv))


if __name__ == "__main__":
    main()
