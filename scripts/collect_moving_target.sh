#!/usr/bin/env bash
# Invoke inside a resource-limited systemd unit on the Linux/CUDA host.
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
VLA_CONFIG="${1:?configuration required}"
VLA_OUTPUT="${2:?output directory required}"
shift 2
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$VLA_ROOT/.cache/libero"
export HF_HOME="$VLA_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p logs
"$VLA_ROOT/.venv/bin/python" -u scripts/run_moving_target.py run "$VLA_CONFIG" \
  --output "$VLA_OUTPUT" "$@" 2>&1 | tee -a "logs/$(basename "$VLA_OUTPUT").log"
