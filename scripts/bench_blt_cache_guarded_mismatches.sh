#!/bin/bash
#SBATCH --job-name=blt-cache-guarded
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail
export NUM_GPUS=1
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${CKPT_PATH:?Set CKPT_PATH to the immutable native checkpoint directory}"
: "${COMPARE_REPORT:?Set COMPARE_REPORT to full v1 comparison JSON}"
: "${CONTROL_REPORT:?Set CONTROL_REPORT to 12-sample full generation JSON}"
: "${BENCH_OUTPUT:?Set BENCH_OUTPUT to a new report path}"
run_job srun --ntasks=1 python -m blt_hf_checks.bench_cache_v2_mismatches \
  --comparison "$COMPARE_REPORT" --control-report "$CONTROL_REPORT" \
  --checkpoint "$CKPT_PATH" --output "$BENCH_OUTPUT" \
  --backend global-prefix-guarded-greedy-v1
