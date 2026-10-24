"""Bundle audited research sources, report and raw evidence without environments/weights."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package(report, output):
    report = Path(report).resolve()
    output = Path(output).resolve()
    if not report.is_relative_to(ROOT):
        raise ValueError("Keep the portable report inside the repository")
    result = json.loads((report / "analysis.json").read_text())
    if result["audit"]["provisional"] or not result["audit"]["all_videos_validated"]:
        raise ValueError("Package requires complete results and the video audit")
    source = hashlib.sha256()
    for path in sorted((ROOT / "vla_replan").glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    if source.hexdigest() != result["primary_manifest"]["runtime"]["code_sha256"]:
        raise ValueError("Runner sources differ from the measured study")
    if hashlib.sha256((ROOT / "scripts/analyze_research.py").read_bytes()).hexdigest() != result["analysis_script_sha256"]:
        raise ValueError("Analysis script changed after report generation")
    if result.get("release_manifest") and hashlib.sha256((ROOT / "scripts/check_release_stability.py").read_bytes()).hexdigest() != result["release_manifest"]["script_sha256"]:
        raise ValueError("Persistence script differs from the measured diagnostic")
    paths = set()
    for name, expected in result["artifact_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Evidence paths must stay inside the repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact changed since analysis: {name}")
        paths.add(path)
    for directory in ("vla_replan", "scripts", "configs", "requirements", "docs", "tests", "results"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and not any(p in ("__pycache__", ".pytest_cache", ".ruff_cache") for p in path.parts):
                paths.add(path)
    for name in ("README.md", "pyproject.toml"):
        paths.add(ROOT / name)
    for name in result["input_runs"]:
        root = ROOT / name
        if root.exists():
            for filename in ("manifest.json", "status.json"):
                paths.add(root / filename)
            paths.update(root.glob("episodes/*/skipped.json"))
    if result.get("release_summary"):
        release = ROOT / result["release_directory"]
        paths.update((release / name for name in ("manifest.json", "status.json")))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    index = {"source_commit": revision, "trials": result["audit"]["trials"],
             "heldout_pairs": result["audit"]["heldout_pairs"],
             "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=3) as archive:
        for path in sorted(paths):
            archive.add(path, arcname="vla-replanning-study/"+str(path.relative_to(ROOT)), recursive=False)
        payload = (json.dumps(index, indent=2)+"\n").encode()
        info = tarfile.TarInfo("vla-replanning-study/artifact_index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(output) as archive:
        assert len(archive.getmembers()) == len(paths)+1
        assert all(m.name.startswith("vla-replanning-study/") and m.isfile() for m in archive.getmembers())
        for name, expected in index["files"].items():
            assert hashlib.sha256(archive.extractfile("vla-replanning-study/"+name).read()).hexdigest() == expected
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+".sha256").write_text(f"{sha}  {output.name}\n")
    print(json.dumps({"archive": str(output), "files": len(paths), "bytes": output.stat().st_size,
                      "sha256": sha, "source_commit": revision}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/research_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.report, args.output)
