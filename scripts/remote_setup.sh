#!/usr/bin/env bash
# Run from a dedicated project checkout on Linux. No sudo, commits or uploads.
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
if [[ ! -x .tools/bin/uv ]]; then
    python3 -m venv .tools
    .tools/bin/python -m pip install --disable-pip-version-check uv==0.13.1
fi
.tools/bin/uv pip install --python .tools/bin/python cmake==3.31.6
export PATH="$VLA_ROOT/.tools/bin:$PATH"
if [[ ! -x .venv/bin/python ]]; then
    .tools/bin/uv venv --python /usr/bin/python3 .venv
fi
.tools/bin/uv pip sync --python .venv/bin/python requirements/linux.lock.txt
.tools/bin/uv pip install --python .venv/bin/python --no-deps .
.venv/bin/python -m vla_replan doctor
