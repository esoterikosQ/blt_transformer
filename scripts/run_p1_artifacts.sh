#!/usr/bin/env bash
# itcerdo only: downloads, CPU checks/conversion. No training or neuron access.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
[[ "$(hostname -s)" == itcerdo ]] || { echo 'itcerdo only'; exit 2; }
export TMPDIR="$PROJECT_ROOT/artifacts/tmp"
export HF_HOME="$PROJECT_ROOT/artifacts/hf_home"
export HF_HUB_CACHE="$PROJECT_ROOT/artifacts/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
PYTHON=/home/itcmaster/miniconda3/envs/phdq_blt_hf/bin/python
mkdir -p "$TMPDIR" artifacts/logs
RUN_NAME=p1_artifacts_20260915
exec >> "artifacts/logs/${RUN_NAME}.log" 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_at=%s\n" "$rc" "$(date -Iseconds)" > "artifacts/logs/${RUN_NAME}.status"' EXIT
printf 'running_since=%s\n' "$(date -Iseconds)" > "artifacts/logs/${RUN_NAME}.status"
echo "=== START $(date -Iseconds) ==="
"$PYTHON" -m unittest discover -s tests -p 'test_hf_*.py' -v
if [[ ! -f blt_hf_checks/manifests/original_downloads_2026-09-15.json ]]; then
  "$PYTHON" blt_hf_checks/download_originals.py
fi
if [[ ! -f blt_hf_checks/results/entropy_sources_20260915.json ]]; then
  "$PYTHON" blt_hf_checks/compare_entropy_sources.py
fi
"$PYTHON" -c 'import psutil; m=psutil.virtual_memory(); print({"ram_total":m.total,"ram_available":m.available}); assert m.available > 28*1024**3, "Need at least 28 GiB available host RAM for CPU conversion"'
"$PYTHON" blt_hf_checks/vendor/convert_blt_weights_to_hf.py \
  --model_id facebook/blt-1b \
  --model-revision 8134b32f0b1d25d1248c30e8c7bdfd442d3bb380 \
  --entropy-model-id facebook/blt-entropy \
  --entropy-revision f2aae511e44e2086b1204bc4ddec6ac6c9651332 \
  --local-files-only \
  --output_dir artifacts/converted/blt-1b-hf-own \
  --report blt_hf_checks/manifests/conversion_B_20260915.json
echo "=== CONVERSION COMPLETE; identity validation pending $(date -Iseconds) ==="
