"""Package source code and completed compact results, without Git history or model weights.

Example:
    .venv/bin/python scripts/package_submission.py --run outputs/compact \
        --pdf output/pdf/vla-replanning-report.pdf \
        --output output/submission/vla-replanning-project.zip

The ZIP has one top-level directory, vla-replan. Run data is stored under outputs/compact,
the final PDF under output/pdf/vla-replanning-report.pdf. SHA256SUMS.txt covers every other
archive member and uses paths relative to vla-replan, suitable for `sha256sum -c` there.
"""

import argparse
import hashlib
import json
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.build_report import validate_discussion, validate_statistics  # noqa: E402
from vla_replan.analysis import load_results, summarize_annotations  # noqa: E402

ARCHIVE_ROOT = PurePosixPath("vla-replan")
DATA_DESTINATION = PurePosixPath("outputs/compact")
PDF_DESTINATION = PurePosixPath("output/pdf/vla-replanning-report.pdf")
CHUNK_BYTES = 1024 * 1024


def required_file(path):
    """Do not accidentally follow symlinks outside the intended delivery inputs."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required nonempty regular file is missing or is a symlink: {path}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def collect_inputs(root, pdf):
    cfg, rows, coverage = load_results(root, allow_incomplete=False)
    if (
        cfg.name != "compact"
        or coverage["completed_episodes"] != 30
        or len(cfg.tasks) != 3
        or len(cfg.initial_states) != 2
        or len(cfg.policy_seeds) != 1
        or tuple(sorted(cfg.intervals)) != (1, 5, 10, 25, 50)
    ):
        raise ValueError("Submission requires the completed 30-episode compact experiment")
    for name in ("demo.mp4", "annotations.csv", "discussion.json", "report/statistics.json", "manifest.json"):
        required_file(root / name)
    required_file(pdf)
    with pdf.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError(f"Not a PDF document: {pdf}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "report/statistics.json").read_text(encoding="utf-8"))
    discussion = json.loads((root / "discussion.json").read_text(encoding="utf-8"))
    validate_discussion(discussion, rows)
    validate_statistics(stats, rows, coverage, cfg.intervals)
    labels = summarize_annotations(root / "annotations.csv", rows)
    if stats.get("annotations") != labels:
        raise ValueError("Annotation statistics are stale; rerun analyze with --annotations")

    # Use an allowlist, never a recursive archive of the project or run directory.
    files = {}

    def add(source, destination):
        source = required_file(source)
        if source.resolve() != source.absolute():
            raise ValueError(f"A parent directory is a symlink: {source}")
        destination = PurePosixPath(destination)
        if destination.is_absolute() or ".." in destination.parts or str(destination) in files:
            raise ValueError(f"Unsafe or duplicate archive destination: {destination}")
        files[str(destination)] = source

    for name in ("README.md", "pyproject.toml", ".gitignore"):
        add(PROJECT / name, name)
    patterns = {
        "vla_replan": ("*.py",),
        "configs": ("*.json",),
        "requirements": ("*.in", "*.txt"),
        "docs": ("*.md",),
        "scripts": ("*.py", "*.sh"),
        "tests": ("*.py",),
    }
    for directory, globs in patterns.items():
        matching = sorted({path for pattern in globs for path in (PROJECT / directory).glob(pattern)})
        if not matching:
            raise FileNotFoundError(f"Missing source files: {directory}")
        for source in matching:
            add(source, source.relative_to(PROJECT).as_posix())

    code_hash = hashlib.sha256()
    for source in sorted((PROJECT / "vla_replan").glob("*.py")):
        code_hash.update(source.name.encode())
        code_hash.update(source.read_bytes())
    if code_hash.hexdigest() != manifest["runtime"]["code_sha256"]:
        raise ValueError("Current package sources differ from the code used for the measured run")
    if sha256_file(PROJECT / "requirements/linux.lock.txt") != manifest["runtime"]["dependency_lock_sha256"]:
        raise ValueError("Current dependency lock differs from the measured run")

    for name in ("manifest.json", "annotations.csv", "discussion.json", "demo.mp4"):
        add(root / name, DATA_DESTINATION / name)
    for row in rows:
        for name in ("summary.json", "steps.jsonl", "chunks.jsonl", "video.mp4"):
            relative = PurePosixPath("episodes") / row["episode_id"] / name
            add(root.joinpath(*relative.parts), DATA_DESTINATION / relative)
    for directory, suffixes in (("report", {".csv", ".json", ".png", ".pdf"}), ("demos", {".mp4", ".png"})):
        for source in sorted((root / directory).glob("*")):
            if source.suffix.lower() in suffixes:
                add(source, DATA_DESTINATION / directory / source.name)
    for source in sorted(root.glob("review-*.json")):
        add(source, DATA_DESTINATION / source.name)
    add(pdf, PDF_DESTINATION)
    return files, coverage, labels


def archive_info(name, source=None):
    # Stable metadata avoids host-specific uid/gid and path leakage.
    info = zipfile.ZipInfo(str(ARCHIVE_ROOT / name), date_time=(2026, 1, 1, 0, 0, 0))
    info.create_system = 3
    mode = 0o755 if source is not None and source.suffix == ".sh" else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = (
        zipfile.ZIP_STORED
        if PurePosixPath(name).suffix.lower() in {".mp4", ".png", ".pdf"}
        else zipfile.ZIP_DEFLATED
    )
    return info


def package_submission(args):
    root = args.run.resolve()
    pdf = args.pdf.resolve()
    output = args.output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must have the .zip extension")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    files, coverage, labels = collect_inputs(root, pdf)
    if output in files.values():
        raise ValueError("Output must not replace a source artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".submission-", dir=output.parent) as temporary:
        staged = Path(temporary) / "submission.zip"
        checksums = []
        with zipfile.ZipFile(
            staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as archive:
            for name, source in sorted(files.items()):
                digest = hashlib.sha256()
                before = source.stat()
                with (
                    source.open("rb") as reader,
                    archive.open(archive_info(name, source), "w", force_zip64=True) as writer,
                ):
                    while chunk := reader.read(CHUNK_BYTES):
                        digest.update(chunk)
                        writer.write(chunk)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Input changed while packaging: {source}")
                checksums.append(f"{digest.hexdigest()}  {name}\n")
            archive.writestr(archive_info("SHA256SUMS.txt"), "".join(checksums).encode("utf-8"))
        with zipfile.ZipFile(staged) as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                raise RuntimeError(f"ZIP CRC check failed: {bad_file}")
            if len(archive.infolist()) != len(files) + 1:
                raise RuntimeError("Archive member count does not match the delivery manifest")
        archive_hash = sha256_file(staged)
        archive_bytes = staged.stat().st_size
        if args.overwrite:
            staged.replace(output)
        else:
            # Atomic no-clobber publication also handles a destination created concurrently.
            output.hardlink_to(staged)
    print(
        json.dumps(
            {
                "output": str(output),
                "files": len(files) + 1,
                "bytes": archive_bytes,
                "sha256": archive_hash,
                "episodes": coverage["analyzed_episodes"],
                "reviewed": sum(label["reviewed"] for label in labels),
                "crc_verified": True,
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
    parser.add_argument("--pdf", type=Path, default=Path("output/pdf/vla-replanning-report.pdf"))
    parser.add_argument("--output", type=Path, default=Path("output/submission/vla-replanning-project.zip"))
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace an existing archive")
    package_submission(parser.parse_args(argv))


if __name__ == "__main__":
    main()
