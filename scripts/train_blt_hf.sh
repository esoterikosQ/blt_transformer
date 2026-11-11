#!/bin/bash
#SBATCH --job-name=blt-hf-train
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
export NUM_GPUS=${NUM_GPUS:-1}
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${RUN_ID:?Set a fresh RUN_ID (or the original run when resuming)}"
DATASET_TYPE=${DATASET_TYPE:-native}
[[ "$DATASET_TYPE" != learner ]] || DATASET_TYPE=korean_learner
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
if [[ "$DATASET_TYPE" == lang8 ]]; then
  python -m blt_hf.derive_lang8
fi
args=(--dataset "$DATASET_TYPE" --run-dir "outputs/blt_hf/$DATASET_TYPE/$RUN_ID"
      --mode "${TRAIN_MODE:-train}" --epochs "${EPOCHS:-10}" --effective-batch "${EFFECTIVE_BATCH:-32}"
      --lr "${LR:-0.00001}" --warmup-ratio "${WARMUP_RATIO:-0.05}"
      --selection-metric "${SELECTION_METRIC:-val_loss}"
      --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-0}"
      --validation-beams "${VALIDATION_BEAMS:-1}"
      --validation-batch-size "${VALIDATION_BATCH_SIZE:-1}"
      --validation-max-new-bytes "${VALIDATION_MAX_NEW_BYTES:-768}"
      --validation-generation-backend "${VALIDATION_GENERATION_BACKEND:-hf-generate-exact-length-unpadded-v1}"
      --save-every "${SAVE_EVERY:-500}" --seed "${SEED:-0}" --max-seconds "${MAX_SECONDS:-21000}"
      --max-steps "${MAX_STEPS:-0}" --overfit-steps "${OVERFIT_STEPS:-200}")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
if [[ "${TRAIN_MODE:-train}" == smoke ]]; then
  run_job srun --ntasks=1 python blt_hf_checks/check_train_forward.py --output "${report}_tiny_training.json"
fi
export LAUNCHER_IS_TORCHRUN=1
run_job srun --ntasks=1 torchrun --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" -m blt_hf.train "${args[@]}"
