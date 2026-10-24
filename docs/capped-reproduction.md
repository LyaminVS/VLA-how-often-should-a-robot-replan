# Reproduce the capped-DVAC comparison

The scientific design is in [capped-protocol.md](capped-protocol.md).
The comparison reuses the 150 fixed-interval episodes already present in
`outputs/compact/` and `outputs/validation/`. It collects 120 adaptive episodes.
Run the following from the repository on the Linux/NVIDIA host after preparing
the pinned environment described in [linux-checklist.md](linux-checklist.md).

## Tests and schedule

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-artifacts/bin/python -m pytest -q tests/test_capped*.py
.venv/bin/python scripts/run_capped.py --config configs/capped.json --plan \
  > outputs/capped-plan.json
```

Use a bounded CPU service for tests and analysis. One or two CPU cores and
4 GiB RAM are sufficient for the artifact tools. No training is required.

## Compatibility checks

The first command compares plain and instrumented final action chunks on
excluded state 21. The second resets each of the 30 recorded starting
situations and compares one new complete prediction with the historical
first prediction. Neither check executes a robot rollout.

```bash
systemd-run --user --wait --pipe --unit=capped-invariance \
  -p WorkingDirectory="$PWD" -p CPUQuota=200% -p AllowedCPUs=0-1 \
  -p MemoryMax=8G -p MemorySwapMax=0 -p TasksMax=128 \
  -p Nice=10 -p IOWeight=10 -p RuntimeMaxSec=10min \
  /usr/bin/taskset -c 0,1 /usr/bin/bash scripts/collect_capped.sh check \
  configs/capped.json --output outputs/capped-check/invariance.json

systemd-run --user --wait --pipe --unit=capped-reference \
  -p WorkingDirectory="$PWD" -p CPUQuota=200% -p AllowedCPUs=0-1 \
  -p MemoryMax=8G -p MemorySwapMax=0 -p TasksMax=128 \
  -p Nice=10 -p IOWeight=10 -p RuntimeMaxSec=10min \
  /usr/bin/taskset -c 0,1 /usr/bin/bash scripts/collect_capped.sh reference-check \
  configs/capped.json --reference-roots outputs/compact outputs/validation \
  --output outputs/capped-check/historical-reference.json
```

Existing check files are not overwritten. If code or settings change, use
new check and collection paths and preserve the earlier evidence. A failed
compatibility check must be investigated before reusing the historical controls.

## Collect the missing adaptive methods

```bash
systemd-run --user --wait --pipe --unit=capped-comparison \
  -p WorkingDirectory="$PWD" -p CPUQuota=200% -p AllowedCPUs=0-1 \
  -p MemoryMax=8G -p MemorySwapMax=0 -p TasksMax=128 \
  -p Nice=10 -p IOWeight=10 -p RuntimeMaxSec=4h \
  /usr/bin/taskset -c 0,1 /usr/bin/bash scripts/collect_capped.sh \
  --config configs/capped.json --output outputs/capped \
  --integrity-check outputs/capped-check/invariance.json \
  --reference-roots outputs/compact outputs/validation
```

Keep the SSH connection alive while using `--wait --pipe`. The wrapper sets
offline model loading, EGL rendering, deterministic CUDA settings, and one
thread per numerical library. Video encoding also uses one thread. There is
one policy, one simulator, and one sequential collection process.

The same historical reset and first-prediction gates run before the first
action of every collected episode. The manifest binds the historical references
to their hashes. Source/configuration/reference changes are rejected on resume.
If interrupted, rerun the command with the same paths and an available unit
name; verified completed episodes are skipped and incomplete attempts retained.
The four-hour limit is a resource bound, not an estimated completion time.

## Verify and assemble the comparison

After all 120 adaptive episodes finish, run these sequentially under a bounded
CPU service with two cores and 4 GiB RAM:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv-artifacts/bin/python scripts/analyze_capped.py \
  outputs/capped outputs/capped-report --decode-videos
.venv-artifacts/bin/python scripts/build_capped_demo.py
.venv-artifacts/bin/python scripts/build_capped_review.py
.venv-artifacts/bin/python scripts/build_capped_report.py \
  --discussion outputs/capped-report/discussion.json
```

Analysis refuses an incomplete 270-episode comparison. It verifies historical
evidence hashes and traces, replays all adaptive variance and cap decisions,
checks the matched starts, and decodes the source videos. Derived statistics,
tables and figures are bound to hashes in `outputs/capped-report/artifacts.json`.

The demo always includes state 0 / seed 11 for all three tasks. The review
builder also selects the first disagreement in each direction between caps
10 and 50, when present. Review those frames before writing a behavioral
interpretation. The delivered full report includes the reviewed cases from
`outputs/capped-report/discussion.json` through `--discussion`. That file must
distinguish observations from hypotheses about failure causes and identify the
inspected evidence. Omitting `--discussion` builds the quantitative report without
its additional qualitative page.

Render every page of the PDF and inspect it before delivery. To rebuild the
existing full PDF, retain the discussion argument and add `--overwrite`:

```bash
.venv-artifacts/bin/python scripts/build_capped_report.py \
  --discussion outputs/capped-report/discussion.json --overwrite
```

The demo and review builders also require `--overwrite` when replacing their
existing derived outputs. These flags do not overwrite the raw episode records.

## Interpretation of timing

The historical fixed runs used six Torch CPU threads without FM diagnostic
capture, while the new adaptive runs use one thread with diagnostics. Compare
fixed and adaptive methods primarily by success and model calls. New adaptive
caps share measurement conditions and can be compared on inference, simulation,
selection, and wall time. Do not report an old-versus-new wall-time speedup.

Preserve both historical data directories, the new `outputs/capped/`, both
checks, the frozen schedule, and derived analysis in `outputs/capped-report/`.
The collection scripts do not commit or push repository changes.
