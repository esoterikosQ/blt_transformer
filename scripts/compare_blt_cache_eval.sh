#!/bin/bash
#SBATCH --job-name=blt-cache-compare
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail
export JOB_KIND=cpu
source /scratch/r984a02/phdq3/scripts/neuron_blt_hf_common.sh
: "${REFERENCE_DIR:?Set REFERENCE_DIR to complete HF native validation output}"
: "${CANDIDATE_DIR:?Set CANDIDATE_DIR to complete global-prefix native validation output}"
: "${COMPARE_OUTPUT:?Set COMPARE_OUTPUT to a new JSON report path}"
run_job srun --ntasks=1 python -m blt_hf_checks.compare_cache_eval \
  --reference-dir "$REFERENCE_DIR" --candidate-dir "$CANDIDATE_DIR" \
  --output "$COMPARE_OUTPUT"
