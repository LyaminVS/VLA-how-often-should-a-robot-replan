"""Source identity of the cap extension and all frozen collection dependencies."""

from pathlib import Path

from adaptive_replan.integrity import file_sha256, source_manifest as adaptive_source_manifest


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    result = dict(adaptive_source_manifest())
    paths = list((root / "capped_replan").glob("*.py"))
    paths += list((root / "vla_replan").glob("*.py"))
    paths += [root / name for name in (
        "scripts/run_capped.py", "scripts/check_capped.py", "scripts/check_capped_reference.py",
        "scripts/collect_capped.sh", "docs/capped-protocol.md", "requirements/linux.lock.txt",
    )]
    for path in sorted(paths):
        if path.is_file():
            result[str(path.relative_to(root))] = file_sha256(path)
    return dict(sorted(result.items()))
