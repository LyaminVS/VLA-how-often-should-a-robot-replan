"""Collection source fingerprint, including unchanged historical dependencies."""

from pathlib import Path
from capped_replan.integrity import file_sha256, source_manifest as capped_sources


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    result = dict(capped_sources())
    paths = list((root / "action_dropout").glob("*.py"))
    paths += [root / name for name in (
        "scripts/run_action_dropout.py", "scripts/collect_action_dropout.sh", "scripts/run_action_dropout_batch.sh",
        "configs/action-dropout.json",
    )]
    for path in paths:
        if path.is_file():
            result[str(path.relative_to(root))] = file_sha256(path)
    return dict(sorted(result.items()))
