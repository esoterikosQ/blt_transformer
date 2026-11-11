#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(hostname -s)" == itcerdo ]] || exit 2
export TMPDIR="$PWD/artifacts/tmp" HF_HOME="$PWD/artifacts/hf_home" HF_HUB_CACHE="$PWD/artifacts/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PYTHON=/home/itcmaster/miniconda3/envs/phdq_blt_hf/bin/python
RUN_NAME=${RUN_NAME:-p1_neuron_code_20260916}
[[ ! -e "artifacts/logs/$RUN_NAME.status" ]] || exit 2
exec >>"artifacts/logs/$RUN_NAME.log" 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_at=%s\n" "$rc" "$(date -Iseconds)" > "artifacts/logs/$RUN_NAME.status"' EXIT
"$PYTHON" -m unittest discover -s tests -p 'test_hf_*.py' -v
"$PYTHON" blt_hf_checks/check_generation.py --output "blt_hf_checks/results/$RUN_NAME.json"
