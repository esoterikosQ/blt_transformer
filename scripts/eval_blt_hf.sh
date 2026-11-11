#!/bin/bash
#SBATCH --job-name=blt-hf-eval
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:55:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
export NUM_GPUS=1
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${CKPT_PATH:?Set CKPT_PATH to an immutable checkpoint directory or best.json}"
: "${EVAL_DIR:?Set EVAL_DIR to a separate directory for each checkpoint/beam/batch condition}"
DATASET_TYPE=${DATASET_TYPE:-native}
[[ "$DATASET_TYPE" != learner ]] || DATASET_TYPE=korean_learner
if [[ "$DATASET_TYPE" == lang8 ]]; then
  python -m blt_hf.derive_lang8
fi
run_job srun --ntasks=1 python -m blt_hf.eval --dataset "$DATASET_TYPE" --split "${SPLIT:-test}" \
  --checkpoint "$CKPT_PATH" --output-dir "$EVAL_DIR" \
  --shard-id "${SHARD_ID:-0}" --shard-count "${SHARD_COUNT:-1}" \
  --num-beams "${BLT_NUM_BEAMS:-1}" --batch-size "${BATCH_SIZE:-1}" \
  --generation-backend "${GENERATION_BACKEND:-hf-generate-exact-length-unpadded-v1}" \
  --max-new-bytes "${MAX_NEW_BYTES:-768}" --length-penalty "${LENGTH_PENALTY:-1}" \
  --max-seconds "${MAX_SECONDS:-6300}"
