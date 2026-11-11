#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(hostname -s)" == itcerdo ]] || exit 2
export TMPDIR="$PWD/artifacts/tmp" HF_HOME="$PWD/artifacts/hf_home" HF_HUB_CACHE="$PWD/artifacts/hub"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PYTHON=/home/itcmaster/miniconda3/envs/phdq_blt_hf/bin/python
RUN_NAME="${RUN_NAME:-p1_validation_20260915}"
exec >> "artifacts/logs/${RUN_NAME}.log" 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_at=%s\n" "$rc" "$(date -Iseconds)" > "artifacts/logs/${RUN_NAME}.status"' EXIT
printf 'running_since=%s\n' "$(date -Iseconds)" > "artifacts/logs/${RUN_NAME}.status"
"$PYTHON" blt_hf_checks/check_env.py --output "blt_hf_checks/results/env_${RUN_NAME}.json"
"$PYTHON" -m unittest discover -s tests -p 'test_hf_*.py' -v
"$PYTHON" blt_hf_checks/check_weight_conversion.py \
  --model-path artifacts/converted/blt-1b-hf-own \
  --conversion-report blt_hf_checks/manifests/conversion_B_20260915.json \
  --mapping-output "blt_hf_checks/manifests/mapping_${RUN_NAME}.json" \
  --output "blt_hf_checks/results/weights_${RUN_NAME}.json"
for backend in eager sdpa; do
  "$PYTHON" blt_hf_checks/check_load_generate.py \
    --model-path artifacts/converted/blt-1b-hf-own \
    --conversion-report blt_hf_checks/manifests/conversion_B_20260915.json \
    --backend "$backend" --output "blt_hf_checks/results/load_${backend}_${RUN_NAME}.json"
done
