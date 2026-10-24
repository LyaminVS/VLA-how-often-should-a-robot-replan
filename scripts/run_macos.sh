#!/usr/bin/env bash
# Run from Terminal/Codex in a logged-in macOS desktop session (GLFW needs OpenGL).
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_ROOT/.cache/libero}"
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/prepare_libero.py
exec .venv/bin/python -m vla_replan run "${1:-configs/smoke.json}" \
    --output "${2:-outputs/mac-smoke}" --device mps "${@:3}"
