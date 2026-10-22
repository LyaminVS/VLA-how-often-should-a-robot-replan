#!/usr/bin/env bash
# Native Apple Silicon setup. No sudo, CUDA, model downloads or experiment execution.
set -euo pipefail
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Use a native arm64 shell on Apple Silicon macOS; Linux uses remote_setup.sh." >&2
    exit 1
fi
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
if [[ ! -x .tools/bin/uv ]]; then
    python3 -m venv .tools
    .tools/bin/python -m pip install --disable-pip-version-check uv==0.13.1
fi
if [[ ! -x .venv/bin/python ]]; then
    .tools/bin/uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info[:2]==(3,12), "Expected native arm64 Python 3.12"'
# pip sync installs the explicit lock without resolving the omitted Linux EGL helpers.
.tools/bin/uv pip sync --python .venv/bin/python requirements/macos.lock.txt
.tools/bin/uv pip install --python .venv/bin/python --no-deps .
.venv/bin/python -m vla_replan doctor
