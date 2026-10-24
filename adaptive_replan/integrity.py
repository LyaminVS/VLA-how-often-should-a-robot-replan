"""Hashes of collection sources and artifacts for reproducible adaptive runs."""

import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "adaptive_replan").glob("*.py"))
    paths += [root / name for name in (
        "scripts/run_dvac.py", "scripts/check_dvac.py", "scripts/collect_dvac.sh",
        "docs/dvac-protocol.md",
    )]
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(paths) if path.is_file()}
