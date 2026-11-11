#!/bin/bash
#SBATCH --job-name=blt-cache-complete
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
set -euo pipefail
export NUM_GPUS=1
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${CKPT_PATH:?Set CKPT_PATH to the immutable native checkpoint directory}"
: "${BENCH_OUTPUT:?Set BENCH_OUTPUT to a new JSON report path}"
run_job srun --ntasks=1 python -m blt_hf_checks.bench_complete_generation \
  --checkpoint "$CKPT_PATH" --output "$BENCH_OUTPUT" \
  --samples "${SAMPLES:-12}" --max-new-bytes "${MAX_NEW_BYTES:-768}"
