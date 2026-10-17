# Reproduction and artifact assembly on Linux / RTX 2060

The main experiment is complete: 30/30 episodes in `outputs/compact`, with 15 LIBERO
successes. Rebuilding the report, video, or archive does not require another GPU run.
The checklist below describes reproduction on a new machine; existing results should
be retained separately.

1. Transfer the sources and configurations, not the macOS `.venv`. Commits and pushes
   are left to the user; the program neither publishes data nor changes Git.
2. Inspect `nvidia-smi` for the GPU, VRAM, driver, and competing processes. Check free
   disk space: CUDA dependencies require substantially more storage than the VLA weights.
3. Create a Python 3.12 environment with `bash scripts/remote_setup.sh`, which installs
   the pinned dependencies from `requirements/linux.lock.txt`. Check EGL/OpenGL and FFmpeg.
4. `doctor` should report Linux, CUDA=true, the expected GPU, and lerobot=0.6.1.
5. Run smoke: one task, one state, and five K values. Exclude downloads and warmup from episode timing.
6. Review all five videos for camera orientation, motion toward the goal, and grasping.
   If every episode fails, check integration and the K=10 baseline before attributing
   the result to replanning frequency.
7. Do not disable strict loading to bypass incompatible weights. Correct the versions
   or architecture and use a new output directory. If initial hashes differ, inspect
   reset behavior and versions rather than ignoring the check.
8. Check memory, latency, action clipping, and video quality. Frequent clipping warrants
   checking normalization and action parameterization.
9. Run `estimate configs/compact.json --from-run outputs/smoke`. The selected main design
   uses 30 episodes, FP32, an RTX 2060, and EGL. Repeat the run command to resume without
   mixing configurations.
10. After completion, transfer the outputs to the Mac for figures, annotation, and paired videos.

## Verified on home, 17 October 2026

- Ubuntu 24.04, Python 3.12.3, RTX 2060, driver 580.178.04, and PyTorch CUDA 12.6.
- Checkpoint loading with strict=True, EGL, and the complete inference → env.step → video pipeline.
- Five technical episodes completed; the K=10 baseline succeeded.
- Mean call latency, including observation and action processing, in smoke: 0.805 s.
- Peak allocated PyTorch memory in smoke: 2,535,382,016 bytes (2.36 GiB).
- Hashes verified identical initial states and camera images across all five K values.
- Small action-limit violations mainly involved the gripper command near ±1; predicted
  action arrays are retained, and clipping is applied identically for every K.
- Main experiment complete: 30/30 episodes, 1775 calls, 6409 steps; mean call latency 0.770 s.
- Sum of measured episode durations, `episode_wall_s`: 27.35 minutes. Installation,
  model loading, warmup, and JSON log writing are excluded.
- Initial states and observations matched across all K values in each of the six paired blocks.
- Successes for K=1/5/10/25/50: 2/6, 5/6, 4/6, 2/6, and 2/6.

Main configuration: `configs/compact.json`; results:
`/home/kaban/vla-replan/outputs/compact`. The experiment ran in the `vla-compact` tmux
session, with log `/home/kaban/vla-replan/logs/compact.log`. The session may no longer
exist after completion. Following a successful run, `scripts/finish_run.py` is called
to produce the analysis, six paired K=1/K=50 videos, and twelve motion plots. The exit
code for the complete sequence is saved in `logs/compact.exit` (0 indicates success).

```bash
ssh home 'tail -20 /home/kaban/vla-replan/logs/compact.log'
ssh home 'cat /home/kaban/vla-replan/logs/compact.exit'
```

For a future run in tmux, the SSH connection can be closed without stopping the process.
When using `tmux attach`, detach with Ctrl+B followed by D. Do not run a second experiment
concurrently on the GPU.

## Assemble final artifacts from saved results

On macOS or Linux, the `[analysis,report]` dependencies are sufficient; CUDA is not needed.
Validated `outputs/compact/annotations.csv` and `outputs/compact/discussion.json` must
be available before assembly. Ambiguous annotated events remain unknown; the LIBERO
success signal alone does not confirm release of the can into the basket.

For analysis without the simulator on a new machine, create a standard Python 3.12 environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[analysis,report,dev]'
```

In an existing Linux environment prepared by `remote_setup.sh`, install additional PDF
dependencies with `.tools/bin/uv pip install --python .venv/bin/python -e '.[analysis,report]'`.
The rendering scripts can use Arial on macOS or DejaVu Sans on Linux; custom TTF paths
can be supplied through `--font` and `--font-bold` when building the PDF and demo.

From the project root:

```bash
.venv/bin/python scripts/finish_run.py outputs/compact
.venv/bin/python -m vla_replan analyze outputs/compact --output outputs/compact/report --annotations outputs/compact/annotations.csv
.venv/bin/python scripts/build_figures.py --run outputs/compact
.venv/bin/python scripts/build_report.py --run outputs/compact --discussion outputs/compact/discussion.json --output output/pdf/vla-replanning-report.pdf
.venv/bin/python scripts/build_demo.py --run outputs/compact --discussion outputs/compact/discussion.json --output outputs/compact/demo.mp4
.venv/bin/python scripts/package_submission.py --run outputs/compact --pdf output/pdf/vla-replanning-report.pdf --output output/submission/vla-replanning-project.zip
```

After changing annotations, rerun `analyze` before rebuilding the PDF, demo, and archive.
To intentionally replace an existing demo or ZIP, add `--overwrite` to the respective
command. The PDF is rebuilt at the specified path. Before submission, open the PDF and
demo and check text rendering, captions, REPLAN events, and outcomes. The archive includes
`SHA256SUMS.txt` for integrity checks. It contains sources and results, but excludes
Python environments and model weights. The packaging script compares source code and
the dependency lock against the manifest of the measured experiment.

For diagnostics, retain the outputs of `doctor` and `nvidia-smi`, together with any
traceback in `outputs/.../attempts/.../error.json`.

## Commands for repeating the experiment

Linux execution requires Python 3.12, working EGL/OpenGL, and a supported NVIDIA driver.
The lock uses PyTorch 2.7.1 / CUDA 12.6; video recording uses FFmpeg from imageio-ffmpeg.
The first run downloads public model weights and simulator assets; no paid inference API is used.

```bash
bash scripts/remote_setup.sh
.venv/bin/python -m vla_replan doctor
bash scripts/run_remote.sh configs/smoke.json outputs/new-smoke
.venv/bin/python -m vla_replan estimate configs/compact.json --from-run outputs/new-smoke
bash scripts/run_remote.sh configs/compact.json outputs/new-compact
.venv/bin/python scripts/finish_run.py outputs/new-compact
```

New output directories separate the repeat from the retained results. Repeating the same
command resumes an unfinished run; changes to code, configuration, software environment,
or GPU prevent resumption. Adding `--limit 10` after the output path limits the number of
new episodes in that invocation. Interrupted attempts remain in `attempts/`; only completed
episodes are published in `episodes/`.

When changing precision or another setting, use a new configuration and a separate output
directory, beginning with another smoke run. The measured experiment used FP32; BF16 was
not selected for the RTX 2060.

Implementation checks: `.venv/bin/pytest -q` and `.venv/bin/ruff check vla_replan scripts tests`.
Contact sheets: `.venv/bin/python scripts/review_frames.py PATH_TO_EPISODE --output PATH_TO_SHEETS --stride 5`.
