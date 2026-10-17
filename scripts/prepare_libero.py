"""Configure LIBERO without interactive prompts; prefetch its simulator assets.

Use the installed package paths, never guesses about another Python environment.
Downloads public simulator assets, not demonstration trajectories.
"""

import json
import os
from importlib.metadata import distribution
from pathlib import Path

from huggingface_hub import snapshot_download

ASSET_REPO = "lerobot/libero-assets"
ASSET_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"

project = Path(__file__).resolve().parents[1]
root = Path(distribution("hf-libero").locate_file("libero/libero")).resolve()
if not (root / "bddl_files").is_dir() or not (root / "init_files").is_dir():
    raise RuntimeError(f"Installed LIBERO resources not found at {root}")
config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", project / ".cache/libero"))
config_dir.mkdir(parents=True, exist_ok=True)
asset_record = config_dir / "asset_revision.json"
if asset_record.exists():
    recorded = json.loads(asset_record.read_text())
    if recorded != {"repo_id": ASSET_REPO, "revision": ASSET_REVISION}:
        raise RuntimeError(f"Simulator asset revision differs from the pinned experiment: {asset_record}")
else:
    asset_record.write_text(json.dumps({"repo_id": ASSET_REPO, "revision": ASSET_REVISION}, indent=2))
revision = ASSET_REVISION
assets = Path(snapshot_download(ASSET_REPO, repo_type="dataset", revision=revision))
for name in ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects"):
    if not (assets / name).is_dir():
        raise RuntimeError(f"Incomplete simulator assets: {name}")
# hf-libero's asset resolver checks this package path before attempting its own downloader.
# Its historical downloader uses a removed Hub argument; a symlink avoids that code path.
package_assets = root / "assets"
if not package_assets.exists() and not package_assets.is_symlink():
    package_assets.symlink_to(assets, target_is_directory=True)
if package_assets.resolve() != assets.resolve():
    raise RuntimeError(f"Installed asset path {package_assets} does not match the recorded snapshot {assets}")
paths = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(config_dir / "datasets"),
    "assets": str(package_assets),
}
config = config_dir / "config.yaml"
if config.exists():
    import yaml

    if yaml.safe_load(config.read_text()) != paths:
        raise RuntimeError(f"Existing LIBERO config differs: {config}; inspect it before replacing")
else:
    config.write_text(json.dumps(paths, indent=2) + "\n")  # JSON is valid YAML.
print(json.dumps({"config": str(config), "asset_revision": revision, "assets": str(assets)}, indent=2))
