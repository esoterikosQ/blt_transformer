#!/usr/bin/env bash
# Sourced by batch scripts. No remote connections or package installation.
set -euo pipefail
PROJECT_ROOT=/scratch/r984a02/phdq3
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'SLURM allocation required; do not run on login nodes' >&2; exit 2; }
JOB_STARTED_AT_EPOCH=$(date +%s)
log_job_end() {
  local rc=$?
  local ended_at_epoch
  ended_at_epoch=$(date +%s)
  printf 'End Time: %s\nElapsed Seconds: %s\nExit Code: %s\n' \
    "$(date -Iseconds)" "$(( ended_at_epoch - JOB_STARTED_AT_EPOCH ))" "$rc"
}
trap log_job_end EXIT
[[ "${SLURM_SUBMIT_DIR:-}" == "$PROJECT_ROOT" ]] || {
  echo "Submit this job from $PROJECT_ROOT (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2; exit 2;
}
cd "$PROJECT_ROOT"
[[ "$(pwd -P)" == "$(cd "$PROJECT_ROOT" && pwd -P)" ]] || exit 2
[[ "${SLURM_NNODES:-1}" == 1 && "${SLURM_NTASKS:-1}" == 1 ]] || { echo 'One node/task required' >&2; exit 2; }
job_details=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$job_details" =~ Comment=field=[^\;\ ]+\;appl=pytorch([\ ]|$) ]] || {
  echo 'Required --comment="field=<showappl value>;appl=pytorch" not found' >&2; exit 2;
}
printf 'Job ID: %s\nNode: %s\nStart Time: %s\nSubmit Dir: %s\nPartition: %s\n' \
  "$SLURM_JOB_ID" "${SLURMD_NODENAME:-unknown}" "$(date -Iseconds)" "$SLURM_SUBMIT_DIR" "${SLURM_JOB_PARTITION:-unknown}"
printf 'CUDA_VISIBLE_DEVICES: %s\nCPUs per task: %s\n' \
  "${CUDA_VISIBLE_DEVICES:-none}" "${SLURM_CPUS_PER_TASK:-unknown}"
if [[ "${JOB_KIND:-gpu}" == cpu ]]; then
  [[ "${SLURM_JOB_PARTITION:-}" == cpu ]] || { echo 'CPU scoring requires cpu partition' >&2; exit 2; }
else
  case "${SLURM_JOB_PARTITION:-}" in
    amd_a100nv_8) cpu_per_gpu=8; max_gpus=8;;
    amd_a100_4) cpu_per_gpu=16; max_gpus=4;;
    amd_h200nv_8) cpu_per_gpu=8; max_gpus=2;;
    *) echo 'Only reviewed A100/H200 partitions are supported' >&2; exit 2;;
  esac
  NUM_GPUS=${NUM_GPUS:-1}
  [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ && "$NUM_GPUS" -le "$max_gpus" ]] || exit 2
  cpus=${SLURM_CPUS_PER_TASK:-1}
  (( cpus <= cpu_per_gpu * NUM_GPUS )) || { echo 'CPU/GPU policy exceeded' >&2; exit 2; }
fi
# Activate an existing environment. Do not load a system CUDA module.
if [[ -n "${CONDA_SH:-}" ]]; then
  source "$CONDA_SH"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  conda_base=$("$CONDA_EXE" info --base)
  source "$conda_base/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  conda_base=$(conda info --base)
  source "$conda_base/etc/profile.d/conda.sh"
elif [[ -f /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh ]]; then
  source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
else
  echo 'Set CONDA_SH to the existing conda.sh path' >&2; exit 2
fi
conda activate "${CONDA_ENV:-phdq_blt_hf}"
export HF_HOME="$PROJECT_ROOT/artifacts/hf_home" HF_HUB_CACHE="$PROJECT_ROOT/artifacts/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$PROJECT_ROOT/artifacts/tmp" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-1} / ${NUM_GPUS:-1} ))
(( OMP_NUM_THREADS > 0 )) || export OMP_NUM_THREADS=1
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$TMPDIR" artifacts/logs
report="blt_hf_checks/results/neuron_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
if [[ "${JOB_KIND:-gpu}" == cpu ]]; then
  python blt_hf_checks/check_env.py --cpu-only --output "${report}_env.json"
else
  python blt_hf_checks/check_env.py --output "${report}_env.json"
  python -c 'import os, torch; expected=(9,0) if os.environ["SLURM_JOB_PARTITION"]=="amd_h200nv_8" else (8,0); assert torch.cuda.device_count()==int(os.environ["NUM_GPUS"]), "Allocated GPU count differs from NUM_GPUS"; assert all(torch.cuda.get_device_capability(i)==expected for i in range(torch.cuda.device_count())), f"GPU capability must match partition: {expected}"; assert all((torch.cuda.set_device(i) is None and torch.cuda.is_bf16_supported()) for i in range(torch.cuda.device_count())), "Native BF16 required"'
fi
if [[ "${SKIP_GLOBAL_DATA_AUDIT:-0}" == 1 ]]; then
  [[ "${JOB_KIND:-gpu}" == cpu && -n "${EVAL_DIR:-}" \
    && -f "$EVAL_DIR/scored/gleu.json" \
    && -f "$EVAL_DIR/scored/m2/run_config.json" ]] || {
    echo 'Global data audit can be skipped only for a previously started CPU score' >&2; exit 2;
  }
  echo 'Global data audit already recorded for this scoring run; selected split is verified by eval.py'
else
  python blt_hf_checks/analyze_data_lengths.py --output "${report}_data.json"
fi

# Forward warning/termination to Python workers, not the torchrun supervisor.
launcher=''
forward_stop() {
  [[ -n "$launcher" ]] || return 0
  if [[ "${LAUNCHER_IS_TORCHRUN:-0}" == 1 ]]; then
    pkill -USR1 -P "$launcher" || true
  else
    kill -USR1 "$launcher" 2>/dev/null || true
  fi
}
trap forward_stop USR1 TERM
run_job() {
  "$@" & launcher=$!
  local rc=0
  while true; do
    wait "$launcher" && rc=0 || rc=$?
    kill -0 "$launcher" 2>/dev/null || break
  done
  launcher=''
  if [[ "$rc" == 75 ]]; then
    echo 'Paused with persisted progress. Resume explicitly with the same run configuration.' >&2
  fi
  return "$rc"
}
