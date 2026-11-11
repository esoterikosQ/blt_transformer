#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "$(hostname -s)" != "itcerdo" ]]; then
  echo "This script is for itcerdo only; do not execute it on neuron." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/home/itcmaster/miniconda3/envs/phdq_blt_hf/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export TMPDIR="$PROJECT_ROOT/artifacts/tmp"
export HF_HOME="$PROJECT_ROOT/artifacts/hf_home"
export HF_HUB_CACHE="$PROJECT_ROOT/artifacts/hub"
mkdir -p "$TMPDIR" "$HF_HOME"

"$PYTHON_BIN" blt_hf_checks/check_env.py --output "blt_hf_checks/results/env_itcerdo_${RUN_ID}.json"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_hf_*.py' -v
"$PYTHON_BIN" blt_hf_checks/analyze_data_lengths.py --output "blt_hf_checks/results/data_itcerdo_${RUN_ID}.json"
# No model weight download, optimizer, backward, SLURM, or environment installation.
