#!/usr/bin/env bash
# Collection stops between episodes before the defense; then verify and summarize.
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
# The 60 fixed_1 dropout episodes run in the separate Kaggle notebook.
bash scripts/collect_action_dropout.sh --methods fixed_5 fixed_10 fixed_25 fixed_50 dvac_cap_5 dvac_cap_10 dvac_cap_25 dvac_cap_50 --deadline "2026-10-25T08:15:00+03:00"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH="$VLA_ROOT" PYTHONUNBUFFERED=1
"$VLA_ROOT/.venv-artifacts/bin/python" -u scripts/analyze_action_dropout.py --draws 2000 2>&1 | tee logs/action-dropout-analysis.log
