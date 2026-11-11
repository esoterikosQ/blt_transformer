#!/bin/bash
#SBATCH --job-name=blt-hf-m2-local
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
export JOB_KIND=cpu NUM_GPUS=1 SKIP_GLOBAL_DATA_AUDIT=1
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${EVAL_DIR:?Set EVAL_DIR to a score with existing GLEU and M2 progress}"
DATASET_TYPE=${DATASET_TYPE:-korean_learner}
[[ "$DATASET_TYPE" != learner ]] || DATASET_TYPE=korean_learner
if [[ "$DATASET_TYPE" == lang8 ]]; then
  python -m blt_hf.derive_lang8
fi
run_job srun --ntasks=1 python -m blt_hf.score_local --dataset "$DATASET_TYPE" \
  --split "${SPLIT:-test}" --output-dir "$EVAL_DIR" --m2-workers "${M2_WORKERS:-8}" \
  --m2-timeout "${M2_TIMEOUT:-30}" --m2-passes "${M2_PASSES:-4}" \
  --max-seconds "${MAX_SECONDS:-6300}"
