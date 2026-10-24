# Reproduction and artifact assembly

The study contains 150 unique episodes: three tasks, ten initial states per task,
and five replanning intervals. Source evidence is stored in `outputs/compact` and
`outputs/validation`; the unified analysis and presentation are in `outputs/expanded`.
The method and metric definitions are in `protocol.md`.

## Linux execution environment

Use Python 3.12, a supported NVIDIA driver and working EGL/OpenGL. The measured
configuration uses an RTX 2060 (6 GB), FP32, batch size 1, and synchronous
simulation. Dependencies and checkpoint revisions are pinned. No training or
paid inference service is required.

```bash
bash scripts/remote_setup.sh
.venv/bin/python -m vla_replan doctor
```

`doctor` should report Linux, CUDA=true, and the intended GPU. The measured
software includes LeRobot 0.6.1, hf-libero 0.1.4, MuJoCo 3.3.2, PyTorch 2.7.1
with CUDA 12.6, and Python 3.12.3. The driver was 580.178.04. The complete recorded
package list, GPU settings, asset revision, code hash and dependency-lock hash
are retained in each original `manifest.json`.

Check `nvidia-smi` for competing GPU processes and available memory. Initial
installation and cached model downloads need additional disk space. Transfer
sources and results between machines; create an environment on the target
machine instead of copying a virtual environment from another operating system.

## Reproduce the study

Keep the supplied evidence unchanged and use new output directories for a fresh
reproduction. A technical smoke run checks weights, camera orientation, actions,
rendering and recording before the full series.

```bash
bash scripts/run_remote.sh configs/smoke.json outputs/new-smoke
.venv/bin/python -m vla_replan estimate configs/validation.json --from-run outputs/new-smoke
bash scripts/run_remote.sh configs/compact.json outputs/new-compact
bash scripts/run_remote.sh configs/validation.json outputs/new-validation
```

The two configuration files cover the disjoint state indices 0–1 and 2–9. Together
they evaluate the same three tasks, seed 11, and K=1/5/10/25/50 over ten states.
The collection order is shuffled
with a fixed seed. Only one experiment should use the GPU at a time. Run in a
persistent terminal session or redirect the process with `nohup` for a long job.

Repeating a run command resumes completed episode directories and continues
unfinished work. A changed configuration, runtime, source hash or hardware is
rejected. Interrupted attempts are retained in `attempts/`; a complete episode
is atomically published in `episodes/`. Adding `--limit 10` after the output path
limits new episodes for that invocation without changing the experiment plan.
Never edit manifests or episode fingerprints to bypass these checks.

The convenience command `bash scripts/collect_validation.sh` uses the fixed
`outputs/validation` destination, writes completion status to
`logs/validation.exit`, and performs quantitative analysis only after all 120
episodes finish. `logs/validation.log` contains progress when redirected there.
The script creates an empty annotation worksheet only if no worksheet exists.
A blank row or `reviewed=no` is not evidence of zero errors.

## Visual review

The balanced event-review subset contains states 0–3 for all tasks and all K
values: 60 videos in total. Numerical success and timing results use all 150
episodes. Review both cameras and retain
ambiguous event counts as unknown. `inspection.json` records selected and
actually inspected episode IDs; it is distinct from whether both event counts
are fully resolved in `annotations.csv`.

```bash
.venv/bin/python scripts/review_frames.py PATH_TO_EPISODE --output PATH_TO_SHEETS --stride 5
```

Contact sheets include every fifth action frame, the last action, neighborhoods
of gripper sign changes, and the final outcome frame. Inspect uncertain contact
in denser windows where needed. Record exact steps, observations and uncertainty
in English notes. The supplied annotations were made by the assistant; there
was no independent human annotation. `outputs/expanded/discussion.json` stores
illustrative paired cases and separates observations from hypotheses.

## Build analysis, report, demo and archive

GPU execution is not needed to rebuild artifacts from saved results. Build on the
Linux execution host, in a separate Python 3.12 environment, and transfer finished
artifacts afterwards. This keeps presentation dependencies separate from the frozen
rollout environment. The source package must remain identical to the rollout code.

```bash
python3.12 -m venv .venv-artifacts
.venv-artifacts/bin/python -m pip install -e '.[analysis,report,dev]'
```

Before final assembly, complete the selected visual review and save each source
directory's `annotations.csv` and `inspection.json`, plus the study discussion in
`outputs/expanded/discussion.json`. Keep the original manifests, raw logs and videos
unchanged. The analysis reads both source directories and validates their combined
150-episode grid.

```bash
.venv-artifacts/bin/python scripts/verify_evidence.py outputs/compact --decode-video --output outputs/compact/report/evidence-verification.json
.venv-artifacts/bin/python scripts/verify_evidence.py outputs/validation --decode-video --output outputs/validation/report/evidence-verification.json
.venv-artifacts/bin/python -m vla_replan analyze outputs/compact --output outputs/compact/report --annotations outputs/compact/annotations.csv
.venv-artifacts/bin/python -m vla_replan analyze outputs/validation --output outputs/validation/report --annotations outputs/validation/annotations.csv
.venv-artifacts/bin/python scripts/analyze_extension.py --pilot outputs/compact --validation outputs/validation --output outputs/expanded/report
.venv-artifacts/bin/python scripts/build_expanded_figures.py --run outputs/expanded --overwrite
.venv-artifacts/bin/python scripts/build_state_outcomes.py --run outputs/expanded --overwrite
.venv-artifacts/bin/python scripts/build_expanded_report.py --run outputs/expanded --discussion outputs/expanded/discussion.json --overwrite
.venv-artifacts/bin/python scripts/build_expanded_demo.py --run outputs/expanded --discussion outputs/expanded/discussion.json --output outputs/expanded/demo.mp4 --overwrite
.venv-artifacts/bin/python scripts/package_expanded_submission.py --run outputs/expanded --pdf output/pdf/vla-replanning-report.pdf --output output/submission/vla-replanning-project.zip --overwrite
```

Run these stages sequentially. On a Linux host with systemd user services and CPU
indices 0 and 1, an individual build command can be bounded as follows:

```bash
systemd-run --user --scope -p CPUQuota=200% -p MemoryMax=4G -p MemorySwapMax=0 \
  -p TasksMax=96 taskset -c 0,1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv-artifacts/bin/python \
  scripts/build_expanded_demo.py --output outputs/expanded/demo.mp4 --overwrite
```

The process tree, including ffmpeg, inherits the limits. CPU affinity also limits
automatic worker sizing. Use the same wrapper for analysis, rendering and packaging;
adjust CPU indices to the host. Inspect small previews sequentially rather than
loading every contact sheet into an application at once. Full source-video decoding
is needed when validating new evidence, not on every report-only rebuild.

The PDF is eight pages. Open every page and check plots, text and tables; inspect
the demo's title cards, instruction, paired motion, REPLAN markers and outcome.
The PDF and demo have JSON sidecars binding them to the exact statistics and
case descriptions. Regenerate downstream artifacts after changing annotations
or discussion. The archive rejects stale artifacts, checks raw hashes, retains
all 150 episodes and includes `SHA256SUMS.txt` plus `DELIVERY.json`. It excludes virtual
environments, cached model weights, Git history and intermediate review sheets.

The compact-only assembly scripts operate on a subset and should not be used
for the full-study deliverables. The unused
`study.json` and `full.json` grids are alternative designs; do not add their
outputs to the supplied cohorts without checking overlap and provenance.

## Verification and diagnostics

```bash
.venv-artifacts/bin/python -m pytest -q
.venv-artifacts/bin/python -m ruff check vla_replan moving_target scripts tests
```

Evidence verification checks all H=50 predictions, exact executed K-prefixes
with the documented clipping, simulation-step sampling seeds, replanning and
call counters, outcomes, timing sums, command metrics, and video/log alignment.
The analysis rejects incomplete runs, mismatched initial observations, overlapping
source cohorts and changed execution conditions. Its bootstrap retains task/state
pairing across K. The top-level statistics use all 150 episodes (`primary_phase`
is `combined`); source-specific diagnostics are not independent replications.

For failures, retain `doctor`, `nvidia-smi`, the process log, and the traceback in
`outputs/.../attempts/.../error.json`. Do not disable strict model loading or
ignore state-hash mismatches to make a run finish. Use a new configuration and
output directory for changes such as precision, horizons, camera processing or
sampling settings, beginning with a separate smoke check.

## Moving-target robustness experiment

The moving-target protocol is specified in `moving-target-protocol.md`. It uses a
separate package and output directory while reusing the unchanged VLA and K-step
executor. The main grid is one task, five new initial states, three policy seeds,
three execution intervals, and three motion conditions: 135 episodes. State 20 is
reserved for physics and integration checks and is excluded from the main results.

Run only one GPU experiment at a time. The following commands assume the existing
Linux environment and cached model assets. Use a new output directory for a fresh
reproduction; the runner verifies checksums before resuming a completed episode.

```bash
export MUJOCO_GL=egl LIBERO_CONFIG_PATH="$PWD/.cache/libero"
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/run_moving_target.py plan configs/moving-target.json
.venv/bin/python scripts/calibrate_moving_target.py \
  --output outputs/new-moving-calibration/physics-b --state 20 --steps 240 --video
```

Run the nine integration checks first, using a bounded service from the project
directory. These exercise contact-triggered force release and the fixed K schedule:

```bash
systemd-run --user --wait --pipe --unit=vla-moving-check-reproduction \
  -p "WorkingDirectory=$PWD" -p CPUQuota=200% -p AllowedCPUs=0-1 \
  -p MemoryMax=8G -p MemorySwapMax=0 -p TasksMax=128 \
  -p Nice=10 -p IOWeight=10 -p RuntimeMaxSec=2h \
  /bin/bash scripts/collect_moving_target.sh \
  configs/moving-target-check.json outputs/new-moving-calibration/model-check
```

Inspect the physical calibration. Once the controller is fixed and the integration
checks pass, register the selected evidence before creating the main output directory:

```bash
.venv/bin/python scripts/freeze_moving_target_protocol.py \
  --physics outputs/new-moving-calibration/physics-b \
  --model-checks outputs/new-moving-calibration/model-check \
  --main-output outputs/new-moving-target \
  --output outputs/new-moving-calibration/frozen-protocol.json
```

The freeze utility validates all nine checks, compares their pre-motion action
prefixes, and hashes their raw evidence together with the physical calibration,
configuration, source code and protocol. It never overwrites an existing record or
creates a retrospective record after the main output directory exists. Use the
rollout environment for this command; its runtime must match the integration checks.

Then collect the main series:

```bash
systemd-run --user --wait --pipe --unit=vla-moving-reproduction \
  -p "WorkingDirectory=$PWD" -p CPUQuota=200% -p AllowedCPUs=0-1 \
  -p MemoryMax=8G -p MemorySwapMax=0 -p TasksMax=128 \
  -p Nice=10 -p IOWeight=10 -p RuntimeMaxSec=2h \
  /bin/bash scripts/collect_moving_target.sh \
  configs/moving-target.json outputs/new-moving-target
```

Adjust CPU indices to the host. Keep the SSH session connected when using `--wait`;
a host without user-service lingering may
stop a user unit after logout. An interrupted run can be resumed with the same config,
source, environment, hardware and output path. Do not modify frozen files to bypass
identity checks. `collect_moving_target.sh` limits numerical-library thread counts,
uses cached assets offline and writes progress under `logs/`.

After all 135 main episodes complete, build derived artifacts sequentially on the
same Linux host, with the CPU and memory limits described above:

```bash
.venv-artifacts/bin/python scripts/verify_moving_target.py \
  --input outputs/moving-target \
  --frozen-protocol outputs/moving-calibration/frozen-protocol.json \
  --output outputs/moving-target-report/evidence-verification.json --decode-video
.venv-artifacts/bin/python scripts/analyze_moving_target.py \
  --input outputs/moving-target --output outputs/moving-target-report
.venv-artifacts/bin/python scripts/build_moving_target_demo.py \
  --input outputs/moving-target --report outputs/moving-target-report \
  --output outputs/moving-target-report/demo.mp4 --overwrite
.venv-artifacts/bin/python scripts/build_moving_target_report.py --overwrite
```

These artifact commands rebuild the supplied experiment. For a fresh reproduction,
pass its new run, calibration, frozen-protocol and analysis paths to the corresponding
commands. The supplied frozen record binds the supplied evidence; never reuse or
rewrite it for a new collection.

Inspect the final PDF pages, plots and demo, then write a `visual-qa.json` record with
their SHA-256 hashes in `outputs/moving-target-report/`. The complete archive builder
requires that record and independently verifies both experiments:

```bash
.venv-artifacts/bin/python scripts/package_complete_submission.py --overwrite
```

The complete report contains the eight-page fixed-interval study and five pages on
the moving target. The complete archive includes both experiments, calibration,
raw logs/videos, source code and provenance. Their outcomes remain separate: 150
episodes address the original three tasks, and 135 address the moving-target stress
test. The combined count is not a pooled success-rate denominator.
