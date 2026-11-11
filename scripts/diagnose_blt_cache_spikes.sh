#!/bin/bash
#SBATCH --job-name=blt-cache-diag
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
: "${CKPT_PATH:?Set CKPT_PATH to the immutable native checkpoint directory}"
: "${DIAG_OUTPUT:?Set DIAG_OUTPUT to a new JSON report path}"
args=(--checkpoint "$CKPT_PATH" --output "$DIAG_OUTPUT"
      --reference-report blt_hf_checks/results/p3a_native_cache_probe_03_decoder.json
      --repeats "${REPEATS:-3}")
run_job srun --ntasks=1 python -m blt_hf_checks.diagnose_cache_spikes "${args[@]}"
