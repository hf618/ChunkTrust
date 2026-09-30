#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
CONFIG_NAME="pi05_base_aloha_robotwin_full_qha_heldout6"

MODE=""
EXP_NAME=""
STUDENT_GPUS="0,1"
TEACHER_GPUS="2,3,4,5,6,7"
STEPS=""
SAVE_INTERVAL=""
BATCH_SIZE="384"
NUM_WORKERS="2"
DATALOADER_PREFETCH_FACTOR="1"
HOST_PREFETCH="2"
TEACHER_PREFETCH="2"
MEM_FRACTION="0.90"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/train_qha_heldout_8gpu.sh --mode smoke|formal --exp-name NAME [options] [-- EXTRA_TRAIN_ARGS...]

This launcher reserves two GPUs for FSDP QHA-head training and one different
GPU for each of the six task-routed frozen AHS teacher workers. It refuses
overlap and consumes all eight allocated H100s.

Options:
  --mode smoke|formal              smoke defaults to 20 steps; formal to 10000
  --exp-name NAME                  Required unique checkpoint directory name
  --student-gpus 0,1               GPU ids visible to the student process
  --teacher-gpus 2,3,4,5,6,7       Physical GPU ids pinned by teacher workers
  --steps N                        Override the registered mode's step count
  --save-interval N                Override checkpoint interval; formal defaults to 2500
  --batch-size N                   Global student batch size; default 384
  --num-workers N                  Data-loader workers; default 2
  --dataloader-prefetch-factor N   Per-worker batch prefetch; default 1
  --host-prefetch N                Host batch prefetch depth; default 2
  --teacher-prefetch N             Online teacher prefetch depth; default 2
  --mem-fraction F                 XLA memory fraction per process; default 0.90
  --dry-run                        Print the exact command without starting it
  -h, --help                       Show this help

The formal command uses the immutable six-train/two-held-out protocol. Do not
pass overrides that change split, candidate mode, cache compatibility, or the
teacher manifest.
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --student-gpus) STUDENT_GPUS="$2"; shift 2 ;;
    --teacher-gpus) TEACHER_GPUS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --dataloader-prefetch-factor) DATALOADER_PREFETCH_FACTOR="$2"; shift 2 ;;
    --host-prefetch) HOST_PREFETCH="$2"; shift 2 ;;
    --teacher-prefetch) TEACHER_PREFETCH="$2"; shift 2 ;;
    --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ "${MODE}" == "smoke" || "${MODE}" == "formal" ]] || die "--mode must be smoke or formal"
[[ -n "${EXP_NAME}" ]] || die "--exp-name is required"
[[ "${BATCH_SIZE}" =~ ^[0-9]+$ && "${BATCH_SIZE}" -gt 0 ]] || die "--batch-size must be positive"
[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "--num-workers must be non-negative"
[[ "${DATALOADER_PREFETCH_FACTOR}" =~ ^[1-9][0-9]*$ ]] || die "--dataloader-prefetch-factor must be positive"
[[ "${HOST_PREFETCH}" =~ ^[1-9][0-9]*$ ]] || die "--host-prefetch must be positive"
[[ "${TEACHER_PREFETCH}" =~ ^[0-9]+$ ]] || die "--teacher-prefetch must be non-negative"

IFS=',' read -r -a STUDENT_GPU_ARRAY <<< "${STUDENT_GPUS}"
IFS=',' read -r -a TEACHER_GPU_ARRAY <<< "${TEACHER_GPUS}"
[[ ${#STUDENT_GPU_ARRAY[@]} -eq 2 ]] || die "The registered recipe requires exactly two student GPUs"
[[ ${#TEACHER_GPU_ARRAY[@]} -eq 6 ]] || die "The registered recipe requires one teacher GPU for each of six tasks"
for student_gpu in "${STUDENT_GPU_ARRAY[@]}"; do
  for teacher_gpu in "${TEACHER_GPU_ARRAY[@]}"; do
    [[ "${student_gpu}" != "${teacher_gpu}" ]] || die "Student and teacher GPUs overlap at ${student_gpu}"
  done
done

if [[ -z "${STEPS}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    STEPS="20"
  else
    STEPS="10000"
  fi
fi
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer"
if [[ -z "${SAVE_INTERVAL}" ]]; then
  if [[ "${MODE}" == "formal" ]]; then
    # QHA-only heads are compact, so retain auditable recovery checkpoints
    # without making a partial checkpoint eligible for held-out reporting.
    SAVE_INTERVAL="2500"
  else
    SAVE_INTERVAL="${STEPS}"
  fi
fi
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "--save-interval must be a positive integer"

if [[ -e "${ROOT}/checkpoints/pi05_base_aloha_robotwin_full_qha_heldout/${EXP_NAME}" ]]; then
  die "Refusing to reuse an existing run directory: ${ROOT}/checkpoints/pi05_base_aloha_robotwin_full_qha_heldout/${EXP_NAME}"
fi

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate RoboTwin
fi

export CUDA_VISIBLE_DEVICES="${STUDENT_GPUS}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/chunktrust/jax}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/chunktrust/openpi}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${ROOT}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

CMD=(
  python "${TRAIN_SCRIPT}" "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --num-train-steps "${STEPS}"
  --save-interval "${SAVE_INTERVAL}"
  --keep-period "${SAVE_INTERVAL}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}"
  --dataloader-pin-memory
  --host-prefetch "${HOST_PREFETCH}"
  --use-preprocessed-cache
  --teacher-worker-devices "${TEACHER_GPUS}"
  --teacher-prefetch "${TEACHER_PREFETCH}"
  --fsdp-devices "${#STUDENT_GPU_ARRAY[@]}"
  --log-interval 20
  --horizon-metrics-interval 100
  --param-norm-interval 100
  --no-wandb-enabled
)
CMD+=("${EXTRA_ARGS[@]}")

printf 'mode=%s\nstudent_gpus=%s\nteacher_gpus=%s\nsteps=%s\nbatch_size=%s\n' \
  "${MODE}" "${STUDENT_GPUS}" "${TEACHER_GPUS}" "${STEPS}" "${BATCH_SIZE}"
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

cd "${ROOT}"
exec "${CMD[@]}"
