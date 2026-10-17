import contextlib
import fcntl
import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_jsonl(path, records):
    with Path(path).open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@contextlib.contextmanager
def exclusive_run(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with (Path(directory) / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another experiment is writing to this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def runtime_metadata():
    packages = {}
    for name in (
        "lerobot",
        "hf-libero",
        "mujoco",
        "robosuite",
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
        "huggingface-hub",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    project = Path(__file__).resolve().parents[1]
    lock = project / "requirements/linux.lock.txt"
    asset_record = (
        Path(os.environ.get("LIBERO_CONFIG_PATH", project / ".cache/libero")) / "asset_revision.json"
    )
    assets = json.loads(asset_record.read_text()) if asset_record.exists() else None
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "code_sha256": source.hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "simulator_assets": assets,
        "render_backend": os.environ.get("MUJOCO_GL"),
        "cpu_count": os.cpu_count(),
    }


def git_metadata():
    root = Path(__file__).resolve().parent.parent

    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}
