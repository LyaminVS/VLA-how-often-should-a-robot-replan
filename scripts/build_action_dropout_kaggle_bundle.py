#!/usr/bin/env python3
"""Build a single-upload notebook with a verified compact source/reference bundle."""

import argparse
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cell(kind, text):
    record = {"cell_type": kind, "id": hashlib.sha256((kind + text).encode()).hexdigest()[:12],
              "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        record.update(execution_count=None, outputs=[])
    return record


def notebook(payload):
    return {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "cells": [
            cell("markdown", """# Translation-command dropout: Kaggle K=1 cohort

**60 episodes:** three LIBERO tasks × ten initial states × two probabilities (`p=0.1, 0.3`).
Every control step replans (`K=1`). With probability `p`, only the clipped XYZ command is replaced by zero; rotation and gripper commands are preserved. The same deterministic disturbance stream is used on the home machine.

In Kaggle **Settings**, enable a GPU accelerator and Internet, then run all cells. One GPU is used. This notebook embeds all experiment code and compact reference metadata; no repository clone, extra dataset, SSH connection or credentials are needed. Public model weights and simulator assets are downloaded at their frozen revisions. The isolated environment and model cache live under `/tmp`; only results are written under `/kaggle/working`.

The main home collection runs the other eight methods. Existing `p=0` results are reused; **this notebook does not add 30 baseline episodes**. Three short excluded checks establish that the dropout wrapper is exactly transparent at `p=0` on the Kaggle GPU. Initial-state bytes must match home. GPU-dependent first observations/actions are logged, and the two probabilities must have exactly matching local initial observations and first predictions. Cross-GPU success/call comparisons carry a hardware limitation; wall-clock timings must remain separate.

This notebook was generated from the frozen project sources. GPU execution must pass its checks on Kaggle before its results are usable. Installation/download time depends on Kaggle networking; no fixed completion time is promised.
"""),
            cell("code", f'''from pathlib import Path
import base64, hashlib, json, os, subprocess, sys, zipfile, io

EMBEDDED_BUNDLE = {payload!r}
BUNDLE_BYTES = base64.b64decode(EMBEDDED_BUNDLE)
# A revised bundle gets a fresh setup directory; previous results stay intact.
BUNDLE_ID = hashlib.sha256(BUNDLE_BYTES).hexdigest()[:12]
ROOT = Path(f"/tmp/vla-dropout-kaggle-{{BUNDLE_ID}}")
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(BUNDLE_BYTES)) as archive:
    for entry in archive.infolist():
        target = (ROOT / entry.filename).resolve()
        if not target.is_relative_to(ROOT.resolve()) or entry.filename.startswith("/"):
            raise RuntimeError("Unsafe embedded archive path")
        if target.exists() and target.is_file() and target.read_bytes() != archive.read(entry):
            raise RuntimeError(f"Existing source differs: {{target}}; choose a fresh ROOT")
    archive.extractall(ROOT)
manifest = json.loads((ROOT / "bundle-manifest.json").read_text())
for name, expected in manifest["files"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
os.environ["HF_HOME"] = "/tmp/vla-dropout-hf-cache"
os.environ["PIP_CACHE_DIR"] = "/tmp/vla-dropout-pip-cache"
os.environ["UV_CACHE_DIR"] = "/tmp/vla-dropout-uv-cache"
print("Verified embedded bundle:", len(manifest["files"]), "files")
print("Results will be saved under /kaggle/working")
'''),
            cell("markdown", """## Install the frozen environment and download public assets

This installs the pinned standalone uv binary, then uses an isolated managed Python 3.12.3 and the exact project dependency lock. It does not require the notebook kernel's pip or ensurepip. Keep Internet enabled for this cell. Subsequent collection uses offline caches. If Kaggle reports an unavailable package or incompatible CUDA driver, stop and retain the error; do not silently replace package versions. See the [official uv installer documentation](https://docs.astral.sh/uv/reference/installer/).
"""),
            cell("code", '''subprocess.run(["bash", str(ROOT / "scripts/setup_action_dropout_kaggle.sh")], cwd=ROOT, check=True)
PYTHON = ROOT / ".venv/bin/python"
print("Environment ready:", PYTHON)
'''),
            cell("markdown", """## Optional resume

For a fresh run leave `RESUME_ZIP` empty. To resume a saved run, attach the previously downloaded `action-dropout-kaggle-results.zip` as a Kaggle input and set its path below. The collector verifies every completed episode and requires the same GPU type, package versions, configuration and source bundle. A different GPU/runtime starts a separate cohort and cannot silently resume these results. Completed episodes in the current session are reused automatically.
"""),
            cell("code", '''RESUME_ZIP = ""  # e.g. /kaggle/input/my-dropout-checkpoint/action-dropout-kaggle-results.zip
OUTPUT = Path("/kaggle/working/action-dropout-kaggle")
ARCHIVE = Path("/kaggle/working/action-dropout-kaggle-results.zip")
if RESUME_ZIP:
    if OUTPUT.exists():
        raise RuntimeError("Output already exists; do not overwrite a current run with a resume archive")
    with zipfile.ZipFile(RESUME_ZIP) as archive:
        for entry in archive.infolist():
            target = (Path("/kaggle/working") / entry.filename).resolve()
            if not target.is_relative_to(OUTPUT.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError("Unsafe resume archive path or symbolic link")
        archive.extractall("/kaggle/working")
    print("Resume files extracted; all evidence will be verified before reuse.")
'''),
            cell("markdown", """## Run the 60 episodes

`MAX_HOURS` is a soft time limit checked between episodes, excluding setup and the short invariance checks. Set it to `0` to disable the limit. Leave `EPISODE_LIMIT=None` for all 60; a small positive value can check the pipeline and is resumed by rerunning this cell. The progress table includes only initial states completed at **both** probabilities. Each completed episode is committed atomically, and a downloadable ZIP is refreshed every five episodes and on normal completion/interruption. A hard Kaggle session kill can interrupt ZIP creation; the last completed ZIP and committed episode directory remain the checkpoints.

Kaggle **Save Version → Save & Run All** runs a separate session; it is not a live checkpoint of the current Python process. Download the ZIP from the running session or save the completed notebook outputs.
"""),
            cell("code", '''MAX_HOURS = 10.0
EPISODE_LIMIT = None
command = [str(PYTHON), "-u", str(ROOT / "scripts/run_action_dropout_kaggle.py"),
           "--output", str(OUTPUT), "--archive", str(ARCHIVE), "--max-hours", str(MAX_HOURS)]
if EPISODE_LIMIT is not None:
    command += ["--limit", str(EPISODE_LIMIT)]
process = subprocess.Popen(command, cwd=ROOT)
try:
    code = process.wait()
except KeyboardInterrupt:
    import signal
    process.send_signal(signal.SIGINT)
    process.wait()  # allow atomic export of completed episodes
    raise
if code:
    raise RuntimeError(f"Collector stopped with exit code {code}; inspect the error above and attempts/error.json")
'''),
            cell("markdown", """## Verify and download results

Send the resulting ZIP back for joint analysis with the home collection. It contains raw commanded/applied actions, masks, model chunks, videos, reset/model references, the p=0 check and hardware provenance. Incomplete runs are explicitly identified; missing episodes are never counted as failures. The small table below is descriptive and does not establish cross-hardware causal equivalence.
"""),
            cell("code", '''subprocess.run([str(PYTHON), str(ROOT / "scripts/run_action_dropout_kaggle.py"),
                "--output", str(OUTPUT), "--archive", str(ARCHIVE), "--verify-only"], cwd=ROOT, check=True)
from IPython.display import display, FileLink
progress = json.loads((OUTPUT / "progress.json").read_text())
print(f"Completed: {progress['completed_episodes']}/60; paired starts: {progress['complete_matched_starts']}/30")
for row in progress["conditions"]:
    print(f"p={row['p']:.1f}: {row['successes']}/{row['episodes']} success; "
          f"mean calls={row['mean_model_calls']:.1f}; realized dropout={row['drop_fraction']:.3f}")
display(FileLink(str(ARCHIVE)))
'''),
        ]}


def notebook_p02(payload):
    """Dedicated second-session notebook; preserve the original K=1 notebook."""
    result = notebook(payload)
    cells = result["cells"]
    cells[0] = cell("markdown", """# Translation-command dropout: p=0.2

**270 episodes:** three LIBERO tasks × ten initial states × nine methods, all at `p=0.2`.

Methods: fixed `K=1, 5, 10, 25, 50` and DVAC with execution caps `C=5, 10, 25, 50`. At each control step, the clipped XYZ command is replaced by zero with probability 0.2; rotation and gripper commands are preserved. The checkpoint, initial states, policy seed and step-indexed disturbance stream match the other experiment cohorts.

Enable **GPU** and **Internet** in Kaggle, import this notebook, and select **Run All**. No individual-cell editing, repository upload or extra dataset is required. The full code is embedded, including the fixes for system Python without ensurepip and inherited uv install directories. Dependencies, model weights and simulator assets use the frozen project versions.

The other eight methods run first; the expensive fixed `K=1` runs last. Completed episodes are saved individually. A method enters the comparison table only after all of its 30 episodes finish. Missing or unfinished episodes are never counted as failures. The historical `p=0` baseline is reused, with no additional full baseline episodes. Three short excluded checks verify the p=0 wrapper before collection.

Hardware and differences from the historical first prediction are recorded. All nine methods in this notebook share the same Kaggle GPU; elapsed times across different machines must be analyzed separately. Outputs use the separate `action-dropout-kaggle-p020` directory.
""")
    for index in (4, 5):
        original = "".join(cells[index]["source"])
        cells[index] = cell(cells[index]["cell_type"], original.replace(
            "action-dropout-kaggle", "action-dropout-kaggle-p020"))
    cells[6] = cell("markdown", """## Run all 270 episodes

The default runs the entire registered plan. `MAX_HOURS=10` is a soft limit checked between episodes, excluding setup and the short invariance checks. Each finished episode is saved atomically. A downloadable ZIP is refreshed every five episodes and when the collector exits normally or handles an interruption. A hard session termination may leave only the previous ZIP checkpoint.

The eight methods other than fixed K=1 run first. Progress prints the current method, task, initial state, success, model calls and elapsed time. Only complete 30-episode method panels enter the final table. Already completed episodes are verified and reused if the collection is resumed in the same environment.

Kaggle **Save Version → Save & Run All** starts a separate execution session. Download the ZIP from the active session or save the finished notebook outputs.
""")
    cells[8] = cell("markdown", """## Verify and download the p=0.2 results

The ZIP includes raw action chunks, commanded and applied actions, disturbance masks, videos, provenance and the short invariance checks. Send `action-dropout-kaggle-p020-results.zip` back for comparison with the p=0, 0.1 and 0.3 results. Partial collection is explicitly reported.
""")
    cells[9] = cell("code", '''subprocess.run([str(PYTHON), str(ROOT / "scripts/run_action_dropout_kaggle.py"),
                "--output", str(OUTPUT), "--archive", str(ARCHIVE), "--verify-only"], cwd=ROOT, check=True)
from IPython.display import display, FileLink
progress = json.loads((OUTPUT / "progress.json").read_text())
print(f"Completed: {progress['completed_episodes']}/{progress['planned_episodes']} episodes")
print("Complete methods:", progress.get("complete_methods", []))
for row in progress["conditions"]:
    print(f"{row['method']} | p={row['p']:.1f}: {row['successes']}/{row['episodes']} success; "
          f"mean calls={row['mean_model_calls']:.1f}; realized dropout={row['drop_fraction']:.3f}")
display(FileLink(str(ARCHIVE)))
''')
    return result


def main():
    from action_dropout.config import DropoutExperiment
    from action_dropout.integrity import source_manifest
    from action_dropout.reference import load_references
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/action-dropout.json")
    parser.add_argument("--baseline-root", default="outputs/capped")
    parser.add_argument("--cohort", choices=("k1", "p02"), default="k1")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = DropoutExperiment.load(args.config)
    references, evidence, historical_manifest = load_references(args.baseline_root, cfg)
    portable = cfg.asdict()
    from action_dropout.config import ALL_METHODS
    is_p02 = args.cohort == "p02"
    portable["methods"] = list(ALL_METHODS) if is_p02 else ["fixed_1"]
    portable["probabilities"] = [0.2] if is_p02 else [0.1, 0.3]
    portable["base"]["name"] = "translation-command-dropout-kaggle-" + ("p020" if is_p02 else "fixed-1")
    portable_cfg = DropoutExperiment.fromdict(portable)
    count = 270 if is_p02 else 60
    if len(portable_cfg.episodes()) != count or len(references) != 30:
        raise ValueError(f"Expected {count} portable episodes and 30 historical matched starts")
    files = {name: (ROOT / name).read_bytes() for name in source_manifest()}
    for name in ("scripts/run_action_dropout_kaggle.py", "scripts/setup_action_dropout_kaggle.sh", "scripts/prepare_libero.py",
                 "scripts/verify_capped.py", "scripts/verify_dvac.py", "scripts/verify_evidence.py"):
        files[name] = (ROOT / name).read_bytes()
    files["configs/action-dropout-kaggle.json"] = (json.dumps(portable, indent=2) + "\n").encode()
    files["references/home-first-predictions.json"] = (json.dumps({"references": references,
        "manifest": historical_manifest, "verified_home_evidence": evidence}, indent=2) + "\n").encode()
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files["bundle-manifest.json"] = (json.dumps({"schema_version": 1, "files": inventory,
        "description": "Frozen source and compact historical references; no raw home videos, model weights or SSH secrets."}, indent=2) + "\n").encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    output = Path(args.output or ("notebooks/action_dropout_kaggle_p02.ipynb" if is_p02
                                  else "notebooks/action_dropout_kaggle.ipynb"))
    output.parent.mkdir(parents=True, exist_ok=True)
    make_notebook = notebook_p02 if is_p02 else notebook
    output.write_text(json.dumps(make_notebook(base64.b64encode(stream.getvalue()).decode()), indent=1) + "\n")
    output.with_suffix(".bundle.zip").write_bytes(stream.getvalue())
    print(json.dumps({"notebook": str(output), "notebook_bytes": output.stat().st_size,
                      "bundle_bytes": len(stream.getvalue()), "source_files": len(inventory),
                      "episodes": count, "probabilities": portable["probabilities"]}, indent=2))


if __name__ == "__main__":
    main()
