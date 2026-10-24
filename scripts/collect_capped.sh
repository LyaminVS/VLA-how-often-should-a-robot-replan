#!/usr/bin/env bash
# Run only inside a resource-limited systemd service on the Linux/CUDA host.
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$VLA_ROOT/.cache/libero"
export HF_HOME="$VLA_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p logs
if [[ "${1:-}" == "check" ]]; then
    shift
    "$VLA_ROOT/.venv/bin/python" -u scripts/check_capped.py "$@" 2>&1 | tee -a logs/capped-check.log
elif [[ "${1:-}" == "reference-check" ]]; then
    shift
    "$VLA_ROOT/.venv/bin/python" -u scripts/check_capped_reference.py "$@" 2>&1 | tee -a logs/capped-reference-check.log
else
    "$VLA_ROOT/.venv/bin/python" -u scripts/run_capped.py "$@" 2>&1 | tee -a logs/capped.log
fi
