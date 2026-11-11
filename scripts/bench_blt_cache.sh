#!/bin/bash
#SBATCH --job-name=blt-cache-bench
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
set -euo pipefail
export NUM_GPUS=1
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${CKPT_PATH:?Set CKPT_PATH to an immutable checkpoint directory or best_gleu.json}"
: "${BENCH_OUTPUT:?Set BENCH_OUTPUT to a new JSON report path}"
args=(--checkpoint "$CKPT_PATH" --output "$BENCH_OUTPUT"
      --dataset "${DATASET_TYPE:-native}" --split "${SPLIT:-val}"
      --samples "${SAMPLES:-12}" --steps "${STEPS:-32}")
if [[ "${REUSE_DECODER:-0}" == 1 ]]; then
  args+=(--reuse-decoder)
fi
run_job srun --ntasks=1 python -m blt_hf_checks.bench_global_reuse "${args[@]}"
