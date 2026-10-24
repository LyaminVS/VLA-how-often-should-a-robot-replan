# Reproduce the DVAC comparison

The experiment specification is in [dvac-protocol.md](dvac-protocol.md). This
page contains execution commands; scientific results belong in the report.
Use the Linux/NVIDIA environment prepared in [linux-checklist.md](linux-checklist.md).
Run these commands from the repository root on that machine. The pinned model
and simulator assets must already be cached. No model training is required.

## Check the implementation and freeze the schedule

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-artifacts/bin/python -m pytest -q tests/test_dvac*.py
.venv/bin/python scripts/run_dvac.py --plan > outputs/dvac-plan.json
```

## Prove instrumentation preserves the generated actions

```bash
systemd-run --user --wait --pipe --unit=dvac-invariance \
  -p WorkingDirectory="$PWD" -p CPUQuota=200% -p AllowedCPUs=0-1 \
  -p MemoryMax=8G -p MemorySwapMax=0 -p TasksMax=128 \
  -p Nice=10 -p IOWeight=10 -p RuntimeMaxSec=10min \
  /usr/bin/taskset -c 0,1 /usr/bin/bash scripts/collect_dvac.sh check \
  configs/dvac.json --output outputs/dvac-check/invariance.json
```

This compares unmodified and instrumented predictions on state 21 for each
task, using the same observation and initial noise. It requires exact equality
of every final action. The output records collection-source hashes, the actual
installed FM sampler hashes, configuration, hardware, and dependency versions.
An existing check is never overwritten. For a changed implementation, choose
new check and collection output paths and retain the old evidence.

## Collect all 48 matched episodes

```bash
systemd-run --user --wait --pipe --unit=dvac-comparison \
  -p WorkingDirectory="$PWD" -p CPUQuota=200% -p AllowedCPUs=0-1 \
  -p MemoryMax=8G -p MemorySwapMax=0 -p TasksMax=128 \
  -p Nice=10 -p IOWeight=10 -p RuntimeMaxSec=2h \
  /usr/bin/taskset -c 0,1 /usr/bin/bash scripts/collect_dvac.sh \
  --config configs/dvac.json --output outputs/dvac \
  --integrity-check outputs/dvac-check/invariance.json
```

Keep the SSH connection alive while `--wait --pipe` is running. The wrapper
sets offline model loading, EGL rendering, deterministic CUDA settings, and
one thread per numerical library. There is one policy and one simulator,
and episodes are collected sequentially. Video encoding uses one thread.

The runner refuses a changed configuration, source tree, sampler, hardware,
or integrity check. On an interrupted run, reuse the same output with an
available service name: complete episodes are checked and skipped, while
incomplete attempts remain under `attempts/`. The next completed episode
uses the original frozen order. Runtime limits are protection, not a promised
duration. Inference, resets, encoding, and completed episode counts are
logged in `logs/dvac.log`.

## Verify evidence and build the presentation

After collection has finished, run the following sequentially under a bounded
CPU service (two cores and 4 GB RAM are sufficient):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv-artifacts/bin/python scripts/analyze_dvac.py \
  outputs/dvac outputs/dvac-report --decode-videos
.venv-artifacts/bin/python scripts/build_dvac_demo.py \
  --input outputs/dvac --output outputs/dvac-report
.venv-artifacts/bin/python scripts/build_dvac_review.py \
  --input outputs/dvac --output outputs/dvac-report/review --include-discordant-failure
.venv-artifacts/bin/python scripts/build_dvac_report.py
```

The analysis independently replays the population variances, past-only
thresholds, selected prefixes, executed actions, seeds, and paired initial
conditions. It also decodes every episode video. `artifacts.json` binds the
analysis, tables, figures, report, and verification to their checksums.
The separate verifier can be invoked with
`scripts/verify_dvac.py outputs/dvac --decode-videos`.

The demonstration always includes state 15 / seed 11 for all three tasks,
fixed K=10 beside DVAC, regardless of outcome. Videos use simulation time;
they do not portray the real inference delay. Visual contact sheets support
qualitative inspection and do not themselves label grasp failures.
The optional failure inspection adds the first pair, in the frozen collection
order, where DVAC fails and fixed K=10 succeeds. It is explicitly selected after
observing outcomes and cannot estimate a representative error rate.

Preserve `outputs/dvac/` with the raw evidence, `outputs/dvac-check/`, the
frozen plan, `outputs/dvac-report/`, and the final PDF. The existing fixed-K
evidence is retained separately and is not pooled into the 48-episode analysis.
