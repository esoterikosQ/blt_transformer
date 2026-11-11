#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(hostname -s)" == itcerdo ]] || exit 2
export TMPDIR="$PWD/artifacts/tmp" HF_HOME="$PWD/artifacts/hf_home" HF_HUB_CACHE="$PWD/artifacts/hub"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PYTHON=/home/itcmaster/miniconda3/envs/phdq_blt_hf/bin/python
RUN_NAME="${RUN_NAME:-p1_osc_20260915}"
exec >> "artifacts/logs/${RUN_NAME}.log" 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_at=%s\n" "$rc" "$(date -Iseconds)" > "artifacts/logs/${RUN_NAME}.status"' EXIT
printf 'running_since=%s\n' "$(date -Iseconds)" > "artifacts/logs/${RUN_NAME}.status"
"$PYTHON" blt_hf_checks/check_env.py --output "blt_hf_checks/results/env_${RUN_NAME}.json"
"$PYTHON" -m unittest discover -s tests -p 'test_hf_*.py' -v
"$PYTHON" blt_hf_checks/check_osc_smoke.py --output "blt_hf_checks/results/${RUN_NAME}.json"
"$PYTHON" blt_hf_checks/check_patch_parity.py --output "blt_hf_checks/results/patches_${RUN_NAME}.json"
