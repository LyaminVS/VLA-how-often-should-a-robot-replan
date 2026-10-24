"""Package the complete 150-episode study and its verified presentation artifacts.

The archive retains original source manifests and raw evidence. Unified statistics
are derived artifacts, not a fabricated simulator manifest. Includes the final PDF,
demo, annotations, inspection evidence and hashes. Python environments, model weights,
Git history and old ZIP files are excluded.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.build_demo import FPS, HEIGHT, WIDTH, inspect_video  # noqa: E402
from scripts.build_report import validate_discussion  # noqa: E402
from scripts.package_submission import (  # noqa: E402
    CHUNK_BYTES,
    archive_info,
    required_file,
    sha256_file,
)
from vla_replan.analysis import summarize_annotations  # noqa: E402

REPORT_SUFFIXES = {".csv", ".json", ".png", ".pdf"}


def validate_pdf(path):
    path = required_file(path)
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(f"Expected a PDF document: {path}")
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if not reader.pages:
        raise ValueError(f"PDF has no pages: {path}")
    return len(reader.pages)


def collect_inputs(args):
    from scripts.build_expanded_figures import load_bundle, validate_expanded_discussion
    from scripts.build_expanded_report import _figure_paths

    expanded = args.run.resolve()
    bundle = load_bundle(expanded, args.pilot, args.validation)
    sources, stats = bundle["sources"], bundle["stats"]
    pilot, validation = sources["pilot"], sources["validation"]
    if (pilot["config"].name != "compact" or validation["config"].name != "validation"
            or stats["provenance"]["bootstrap_draws"] != 5000):
        raise ValueError("Delivery requires the original compact pilot and fixed validation protocol")
    if expanded.is_relative_to(pilot["root"]) or expanded.is_relative_to(validation["root"]):
        raise ValueError("Derived artifacts must remain outside the original source runs")
    if (expanded / "manifest.json").exists():
        raise ValueError("Expanded results must not contain a synthetic merged simulator manifest")
    if pilot["config"].order_seed != validation["config"].order_seed:
        raise ValueError("The fixed extension uses the same order seed as the pilot")
    validated_snapshots = {bundle["statistics_path"]: bundle["statistics_sha256"]}
    _figure_paths(bundle, validated_snapshots)  # Validate actual figures and retain their digests.
    state_record = json.loads(required_file(expanded / "report/state-outcomes.json").read_text())
    expected_state_record = {
        "schema_version": 1, "artifact_kind": "state_outcomes", "primary_phase": "combined",
        "combined_analysis_role": "main_descriptive",
        "statistics_sha256": sha256_file(bundle["statistics_path"]),
        "generator_sha256": sha256_file(PROJECT / "scripts/build_state_outcomes.py"),
        "episodes": {"pilot": 30, "validation": 120, "combined": 150},
        "states": list(range(10)), "intervals": [1, 5, 10, 25, 50], "metric": "native_success",
        "files": {
            name: sha256_file(required_file(expanded / "report" / name))
            for name in ("state-outcomes.png", "state-outcomes.pdf")
        },
    }
    if state_record != expected_state_record:
        raise ValueError("State-outcome figure provenance is stale; rerun scripts/build_state_outcomes.py")

    pilot_discussion = json.loads(required_file(pilot["root"] / "discussion.json").read_text())
    validation_discussion = json.loads(required_file(validation["root"] / "discussion.json").read_text())
    validate_discussion(pilot_discussion, pilot["rows"])
    validate_expanded_discussion(validation_discussion, validation["rows"])
    discussion_path = expanded / "discussion.json"
    discussion = json.loads(required_file(discussion_path).read_text())
    validate_expanded_discussion(discussion, bundle["rows"]["combined"])
    # All prespecified videos must have been inspected; ambiguous event counts remain valid.
    for phase in ("pilot", "validation"):
        selected = [label for label in bundle["labels"][phase] if label["selected_for_visual_review"]]
        if len(selected) != 30 or any(label.get("inspected") is not True for label in selected):
            raise ValueError(f"{phase}: delivery requires inspection of all 30 prespecified videos")
        source = sources[phase]
        stored = json.loads(required_file(source["root"] / "report/statistics.json").read_text())
        expected = stats["phases"][phase]
        for field in ("coverage", "estimates", "per_task"):
            if stored.get(field) != expected[field]:
                raise ValueError(f"{phase}: source-phase report {field} is stale")
        labels = summarize_annotations(source["root"] / "annotations.csv", source["rows"])
        if stored.get("annotations") != labels:
            raise ValueError(f"{phase}: source-phase annotation statistics are stale")

    pdf = args.pdf.resolve()
    pilot_pdf = pilot["root"] / "report/pilot-report.pdf"
    pdf_pages = validate_pdf(pdf)
    validate_pdf(pilot_pdf)
    if sha256_file(pdf) == sha256_file(pilot_pdf):
        raise ValueError("Final PDF is still identical to the preserved pilot PDF")
    pdf_evidence_path = pdf.with_suffix(".json")
    pdf_evidence = json.loads(required_file(pdf_evidence_path).read_text())
    expected_pdf = {
        "schema_version": 1, "primary_phase": "combined", "pdf_sha256": sha256_file(pdf),
        "statistics_sha256": sha256_file(bundle["statistics_path"]),
        "discussion_sha256": sha256_file(discussion_path),
        "figures_record_sha256": sha256_file(expanded / "report/expanded-figures.json"),
        "state_outcomes_record_sha256": sha256_file(expanded / "report/state-outcomes.json"),
        "generator_sha256": sha256_file(PROJECT / "scripts/build_expanded_report.py"), "pages": 8,
    }
    if pdf_pages != 8 or any(pdf_evidence.get(key) != value for key, value in expected_pdf.items()):
        raise ValueError("Final PDF provenance is stale; rebuild it from the final statistics and discussion")
    for video in (pilot["root"] / "demo.mp4", expanded / "demo.mp4"):
        required_file(video)
    demo_evidence = json.loads(required_file(expanded / "demo.json").read_text())
    selected_ids = [case["episode_ids"][:2] for case in discussion["cases"]]
    by_id = {row["episode_id"]: row for row in bundle["rows"]["combined"]}
    frames = 7 * FPS + sum(2 * FPS + max(by_id[identity]["steps"] for identity in pair) + FPS
                          for pair in selected_ids)
    expected_demo = {
        "schema_version": 1, "artifact_kind": "combined_demo", "primary_phase": "combined",
        "statistics_sha256": sha256_file(bundle["statistics_path"]),
        "discussion_sha256": sha256_file(discussion_path),
        "build_script_sha256": sha256_file(PROJECT / "scripts/build_expanded_demo.py"),
        "video_sha256": sha256_file(expanded / "demo.mp4"), "case_episode_ids": selected_ids,
        "source_video_sha256": {
            identity: sha256_file(sources[by_id[identity]["phase"]]["root"] / "episodes" / identity / "video.mp4")
            for pair in selected_ids for identity in pair
        },
        "frames": frames, "fps": FPS, "size": [WIDTH, HEIGHT],
    }
    if any(demo_evidence.get(key) != value for key, value in expected_demo.items()):
        raise ValueError("Expanded demo provenance is stale; rebuild it from the final statistics and cases")
    inspect_video(expanded / "demo.mp4", (WIDTH, HEIGHT), expected_frames=frames)

    files = {}
    expected_hashes = {}

    def add(source, destination, expected=None):
        source = required_file(Path(source))
        if source.resolve() != source.absolute():
            raise ValueError(f"Symlinks are not permitted in delivery paths: {source}")
        destination = PurePosixPath(destination)
        if destination.is_absolute() or ".." in destination.parts or str(destination) in files:
            raise ValueError(f"Unsafe or duplicate archive destination: {destination}")
        digest = sha256_file(source)
        if expected is None:
            expected = validated_snapshots.get(source.resolve())
        if expected is not None and expected != digest:
            raise ValueError(f"Source changed since expanded analysis: {source}")
        files[str(destination)] = source
        expected_hashes[str(destination)] = digest

    for name in ("README.md", "pyproject.toml", ".gitignore", ".gitattributes"):
        add(PROJECT / name, name)
    patterns = {
        "vla_replan": ("*.py",), "configs": ("*.json",), "requirements": ("*.in", "*.txt"),
        "docs": ("*.md",), "scripts": ("*.py", "*.sh"), "tests": ("*.py",),
    }
    for directory, globs in patterns.items():
        matching = sorted({path for pattern in globs for path in (PROJECT / directory).glob(pattern)})
        if not matching:
            raise FileNotFoundError(f"Missing source files: {directory}")
        for source in matching:
            add(source, source.relative_to(PROJECT).as_posix())

    recorded_sources = {item["phase"]: item for item in stats["provenance"]["sources"]}
    for phase, source in sources.items():
        root = source["root"]
        destination = PurePosixPath("outputs") / ("compact" if phase == "pilot" else "validation")
        recorded = recorded_sources[phase]
        add(root / "manifest.json", destination / "manifest.json", recorded["manifest_sha256"])
        for name, field in (("annotations.csv", "annotations"), ("inspection.json", "inspection")):
            evidence = recorded.get(field)
            if evidence is None:
                raise ValueError(f"{phase}: missing {field} provenance")
            add(root / name, destination / name, evidence["sha256"])
        add(root / "discussion.json", destination / "discussion.json")
        if phase == "pilot":
            add(root / "demo.mp4", destination / "demo.mp4")
        episodes = {item["episode_id"]: item for item in recorded["episodes"]}
        for row in source["rows"]:
            identity = row["episode_id"]
            for name in ("summary.json", "steps.jsonl", "chunks.jsonl", "video.mp4"):
                relative = PurePosixPath("episodes") / identity / name
                add(root.joinpath(*relative.parts), destination / relative,
                    episodes[identity]["files"][name]["sha256"])
        for directory, suffixes in (("report", REPORT_SUFFIXES), ("demos", {".mp4", ".png"})):
            for path in sorted((root / directory).rglob("*")):
                if path.suffix.lower() in suffixes:
                    add(path, destination / path.relative_to(root).as_posix())
        for path in sorted(root.glob("review-*.json")):
            add(path, destination / path.name)
        if (root / "protocol-record.json").exists():
            add(root / "protocol-record.json", destination / "protocol-record.json")

    for path in sorted((expanded / "report").rglob("*")):
        if path.suffix.lower() in REPORT_SUFFIXES:
            add(path, PurePosixPath("outputs/expanded") / path.relative_to(expanded).as_posix())
    add(expanded / "demo.mp4", "outputs/expanded/demo.mp4")
    add(expanded / "demo.json", "outputs/expanded/demo.json")
    add(discussion_path, "outputs/expanded/discussion.json")
    for path in sorted((expanded / "demos").glob("*")):
        if path.suffix.lower() in {".mp4", ".png"}:
            add(path, PurePosixPath("outputs/expanded/demos") / path.name)
    for path in sorted(expanded.glob("review-*.json")):
        add(path, PurePosixPath("outputs/expanded") / path.name)
    if (PROJECT / "outputs/validation-plan.json").exists():
        add(PROJECT / "outputs/validation-plan.json", "outputs/validation-plan.json")
    add(pdf, "output/pdf/vla-replanning-report.pdf")
    add(pdf_evidence_path, "output/pdf/vla-replanning-report.json")

    delivery = {
        "schema_version": 1, "kind": "combined_study_delivery", "primary_phase": "combined",
        "phases": {
            "pilot": {"episodes": 30, "states": [0, 1], "root": "outputs/compact"},
            "validation": {"episodes": 120, "states": list(range(2, 10)), "root": "outputs/validation"},
        },
        "unique_episode_count": 150, "combined_analysis_role": "main_descriptive",
        "derived_report": "outputs/expanded/report/statistics.json",
        "derived_report_sha256": expected_hashes["outputs/expanded/report/statistics.json"],
        "source_manifest_sha256": {phase: item["manifest_sha256"] for phase, item in recorded_sources.items()},
        "code_sha256": stats["provenance"]["runtime"]["code_sha256"],
        "dependency_lock_sha256": stats["provenance"]["runtime"]["dependency_lock_sha256"],
        "note": "This file describes the archive. Original run manifests are preserved; no merged simulator run is claimed.",
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
        raise ValueError("Output must not replace a source artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expanded-submission-", dir=output.parent) as temporary:
        staged = Path(temporary) / "submission.zip"
        checksums = []
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as archive:
            for name, source in sorted(files.items()):
                digest = hashlib.sha256()
                before = source.stat()
                with source.open("rb") as reader, archive.open(
                    archive_info(name, source), "w", force_zip64=True
                ) as writer:
                    while chunk := reader.read(CHUNK_BYTES):
                        digest.update(chunk)
                        writer.write(chunk)
                after = source.stat()
                if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                        or digest.hexdigest() != expected_hashes[name]):
                    raise RuntimeError(f"Input changed while packaging: {source}")
                checksums.append(f"{digest.hexdigest()}  {name}\n")
            payload = (json.dumps(delivery, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            archive.writestr(archive_info("DELIVERY.json"), payload)
            checksums.append(f"{hashlib.sha256(payload).hexdigest()}  DELIVERY.json\n")
            archive.writestr(archive_info("SHA256SUMS.txt"), "".join(checksums).encode("utf-8"))
        with zipfile.ZipFile(staged) as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                raise RuntimeError(f"ZIP CRC check failed: {bad_file}")
            if len(archive.infolist()) != len(files) + 2:
                raise RuntimeError("Unexpected archive member count")
        digest, size = sha256_file(staged), staged.stat().st_size
        if args.overwrite:
            staged.replace(output)
        else:
            output.hardlink_to(staged)
    print(json.dumps({
        "output": str(output), "files": len(files) + 2, "bytes": size, "sha256": digest,
        "pilot_episodes": 30, "validation_episodes": 120, "unique_episodes": 150,
        "primary_phase": "combined", "crc_verified": True,
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--pilot", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--pdf", type=Path, default=Path("output/pdf/vla-replanning-report.pdf"))
    parser.add_argument("--output", type=Path, default=Path("output/submission/vla-replanning-project.zip"))
    parser.add_argument("--overwrite", action="store_true")
    package_submission(parser.parse_args(argv))


if __name__ == "__main__":
    main()
