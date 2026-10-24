#!/usr/bin/env bash
# Collect the fixed prospective extension and analyze only after completion.
set -euo pipefail
VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLA_ROOT"
mkdir -p logs
trap 'printf "%s\n" "$?" > logs/validation.exit' EXIT
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vla-matplotlib"
bash scripts/run_remote.sh configs/validation.json outputs/validation
.venv/bin/python -m vla_replan analyze outputs/validation --output outputs/validation/report
if [[ ! -f outputs/validation/annotations.csv ]]; then
  .venv/bin/python -m vla_replan annotations outputs/validation --output outputs/validation/annotations.csv
fi
