"""Pin the attention adaptation alongside unchanged rollout dependencies."""

from pathlib import Path

from capped_replan.integrity import file_sha256, source_manifest as capped_sources


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    result = capped_sources()
    paths = list((root / "autohorizon_replan").glob("*.py"))
    paths += [root / name for name in (
        "scripts/run_autohorizon.py", "scripts/collect_autohorizon.sh",
        "docs/autohorizon-protocol.md", "configs/autohorizon.json",
    )]
    for path in paths:
        if path.is_file():
            result[str(path.relative_to(root))] = file_sha256(path)
    return dict(sorted(result.items()))
