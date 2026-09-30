#!/usr/bin/env bash
# Eight-task pi0.5 recipe using the released online-teacher training runtime.
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to a checkout prepared with --heldout}"
POLICY_ROOT="${ROBOTWIN_ROOT}/policy/pi05_horizon"
EXP_NAME="qha-eight-task"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Usage: bash scripts/train_qha_eight_task.sh [--exp-name NAME] [--dry-run]" >&2; exit 2 ;;
  esac
done
[[ -f "${POLICY_ROOT}/src/openpi/training/online_teachers.py" ]] || { echo 'Prepare the backend with --heldout to include the online-teacher runtime.' >&2; exit 2; }
[[ ! -e "${POLICY_ROOT}/checkpoints/pi05_base_aloha_robotwin_full_qha_joint/${EXP_NAME}" ]] || { echo 'Choose a new experiment name; the output directory already exists.' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="${POLICY_ROOT}:${POLICY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
CMD=(python scripts/train.py pi05_base_aloha_robotwin_full_qha_joint
  --exp-name "${EXP_NAME}"
  --batch-size 256 --num-train-steps 10000
  --lr-schedule.warmup-steps 1000 --lr-schedule.peak-lr 5e-5
  --lr-schedule.decay-steps 10000 --lr-schedule.decay-lr 5e-6
  --ema-decay 0.99 --fsdp-devices 2 --no-eager-compile-step-fns
  --multi-task-batch-mode task_balanced_coverage
  --teacher-runtime-mode online_per_task
  --teacher-execution-backend gpu_workers --teacher-worker-devices 2,3,4,5,6,7
  --teacher-checkpoint-manifest "${RELEASE_ROOT}/configs/qha_8task/pi05_teachers.json"
  --teacher-prefetch 2 --host-prefetch 2 --num-workers 2
  --use-preprocessed-cache
  --preprocessed-cache-root "${POLICY_ROOT}/assets/preprocessed/pi05_base_aloha_robotwin_full"
  --preprocessed-cache-compat-config-name pi05_base_aloha_robotwin_full
  --preprocessed-cache-compat-data-factory-type LeRobotAlohaDataConfig
  --save-interval 2500 --keep-period 2500 --no-wandb-enabled)
printf 'Working directory: %s\n' "${POLICY_ROOT}"
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then exit 0; fi
cd "${POLICY_ROOT}"
exec "${CMD[@]}"
