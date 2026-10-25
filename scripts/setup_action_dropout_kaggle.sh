#!/usr/bin/env bash
# Isolate the frozen Linux environment from the notebook kernel.
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--bootstrap-only" ) ]]; then
    echo "Usage: $0 [--bootstrap-only]" >&2
    exit 2
fi
# Kaggle's system Python may have no working ensurepip. Bootstrap a native uv
# binary first; uv creates environments and installs packages without pip.
VLA_UV="$VLA_ROOT/.tools/bin/uv"
if [[ ! -x "$VLA_UV" ]]; then
    mkdir -p "$VLA_ROOT/.tools/bin"
    curl --fail --location --silent --show-error --retry 3 \
        https://astral.sh/uv/0.13.1/install.sh -o "$VLA_ROOT/.tools/install-uv.sh"
    # UV_INSTALL_DIR (often inherited from a container) takes precedence over
    # UV_UNMANAGED_INSTALL. Set both so the pinned binary lands at VLA_UV.
    UV_INSTALL_DIR="$VLA_ROOT/.tools/bin" UV_UNMANAGED_INSTALL="$VLA_ROOT/.tools/bin" \
        sh "$VLA_ROOT/.tools/install-uv.sh"
fi
case "$("$VLA_UV" --version)" in
    "uv 0.13.1"|"uv 0.13.1 "*) ;;
    *) echo "Expected the pinned uv 0.13.1 binary" >&2; exit 1 ;;
esac
export UV_PYTHON_INSTALL_DIR="$VLA_ROOT/.tools/python"
if [[ ! -x .build-tools/bin/python ]]; then
    "$VLA_UV" venv --python 3.12.3 --python-preference only-managed .build-tools
fi
"$VLA_UV" pip install --python .build-tools/bin/python cmake==3.31.6
export PATH="$VLA_ROOT/.build-tools/bin:$VLA_ROOT/.tools/bin:$PATH"
if [[ ! -x .venv/bin/python ]]; then
    "$VLA_UV" venv --python 3.12.3 --python-preference only-managed .venv
fi
if [[ "${1:-}" == "--bootstrap-only" ]]; then
    .venv/bin/python --version
    cmake --version
    exit 0
fi
"$VLA_UV" pip sync --python .venv/bin/python requirements/linux.lock.txt
export LIBERO_CONFIG_PATH="$VLA_ROOT/.cache/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/prepare_libero.py
.venv/bin/python - <<'PY'
import json
import torch
from huggingface_hub import snapshot_download
from pathlib import Path
cfg = json.loads(Path('configs/action-dropout-kaggle.json').read_text())['base']
if not torch.cuda.is_available():
    raise RuntimeError('Enable a Kaggle NVIDIA GPU accelerator before running this notebook')
print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)
snapshot_download(cfg['model_id'], revision=cfg['model_revision'])
snapshot_download(cfg['backbone_id'], revision=cfg['backbone_revision'],
                  allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
print('Pinned simulator assets and policy files are available.')
PY
