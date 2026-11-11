#!/bin/bash
#SBATCH --job-name=blt-gen-bench
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
export NUM_GPUS=1
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${CKPT_PATH:?Set CKPT_PATH to an immutable checkpoint directory or best_gleu.json}"
: "${BENCH_OUTPUT:?Set BENCH_OUTPUT to a new JSON report path}"
DATASET_TYPE=${DATASET_TYPE:-native}
if [[ "$DATASET_TYPE" == lang8 ]]; then
  python -m blt_hf.derive_lang8
fi
run_job srun --ntasks=1 python -m blt_hf_checks.bench_generation \
  --checkpoint "$CKPT_PATH" --output "$BENCH_OUTPUT" \
  --dataset "$DATASET_TYPE" --split "${SPLIT:-val}" \
  --num-beams "${BLT_NUM_BEAMS:-1}" --batch-size "${BATCH_SIZE:-4}" \
  --groups "${GROUPS:-16}" --max-new-bytes "${MAX_NEW_BYTES:-768}"
