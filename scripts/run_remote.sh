#!/usr/bin/env bash
# Usage: bash scripts/run_remote.sh configs/smoke.json outputs/smoke
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$VLA_ROOT/.cache/libero"
export HF_HOME="$VLA_ROOT/.cache/huggingface"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -u scripts/prepare_libero.py
.venv/bin/python -u -m vla_replan run "${1:?configuration required}" --output "${2:?output directory required}" "${@:3}"
