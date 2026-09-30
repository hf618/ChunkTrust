#!/bin/bash
set -euo pipefail

# Respect externally supplied H100 memory tuning; keep the old workstation default otherwise.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

policy_name=pi05_horizon
if [[ $# -lt 6 ]]; then
  echo "Usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id qha_eval_variant eval_tag ...] [-- eval overrides]"
  exit 2
fi

task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
shift 6

checkpoint_id=20000
qha_eval_variant=native
eval_tag=official
checkpoint_root_tag=
qha_hybrid_prior_gamma_override=
base_model_name=
base_checkpoint_id=
base_checkpoint_root_tag=
horizon_ts_forget_rho_override=
horizon_ts_update_eta_override=
horizon_ts_kernel_bandwidth_override=
passthrough_overrides=()

# Backward compatible positional mode used by QHA launchers:
#   eval.sh ... checkpoint_id qha_eval_variant eval_tag checkpoint_root_tag ...
if [[ $# -gt 0 && "${1}" != --* ]]; then
  checkpoint_id=${1:-${checkpoint_id}}; shift || true
  if [[ $# -gt 0 && "${1}" != --* ]]; then qha_eval_variant=${1:-${qha_eval_variant}}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then eval_tag=${1:-${eval_tag}}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then checkpoint_root_tag=${1:-}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then qha_hybrid_prior_gamma_override=${1:-}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then base_model_name=${1:-}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then base_checkpoint_id=${1:-}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then base_checkpoint_root_tag=${1:-}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then horizon_ts_forget_rho_override=${1:-}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then horizon_ts_update_eta_override=${1:-}; shift || true; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then horizon_ts_kernel_bandwidth_override=${1:-}; shift || true; fi
fi

# Flag mode used by eval_all.sh TT launches. Recognized flags are mapped to
# eval.sh's structured arguments; the rest are forwarded to eval_policy.py as
# config overrides.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_id)
      checkpoint_id="$2"; shift 2 ;;
    --qha_eval_variant)
      qha_eval_variant="$2"; shift 2 ;;
    --eval_tag)
      eval_tag="$2"; shift 2 ;;
    --checkpoint_root_tag)
      checkpoint_root_tag="$2"; shift 2 ;;
    --qha_hybrid_prior_gamma|--qha_hybrid_prior_gamma_override)
      qha_hybrid_prior_gamma_override="$2"; shift 2 ;;
    --base_model_name)
      base_model_name="$2"; shift 2 ;;
    --base_checkpoint_id)
      base_checkpoint_id="$2"; shift 2 ;;
    --base_checkpoint_root_tag)
      base_checkpoint_root_tag="$2"; shift 2 ;;
    --horizon_ts_forget_rho)
      horizon_ts_forget_rho_override="$2"; shift 2 ;;
    --horizon_ts_update_eta)
      horizon_ts_update_eta_override="$2"; shift 2 ;;
    --horizon_ts_kernel_bandwidth)
      horizon_ts_kernel_bandwidth_override="$2"; shift 2 ;;
    --)
      shift
      passthrough_overrides+=("$@")
      break ;;
    *)
      if [[ $# -ge 2 ]]; then
        passthrough_overrides+=("$1" "$2")
        shift 2
      else
        passthrough_overrides+=("$1")
        shift
      fi
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

checkpoint_root_args=()
if [[ -n "${checkpoint_root_tag}" ]]; then
  checkpoint_root_args=(--checkpoint_root_tag "${checkpoint_root_tag}")
fi

gamma_tag=""
if [[ -n "${qha_hybrid_prior_gamma_override}" ]]; then
  gamma_tag="${qha_hybrid_prior_gamma_override//+/}"
  gamma_tag="${gamma_tag//./p}"
  gamma_tag="${gamma_tag//-/m}"
fi

ckpt_setting=$([[ "${qha_eval_variant}" == "native" ]] && echo "${model_name}" || echo "${model_name}__${qha_eval_variant}")
if [[ -n "${gamma_tag}" ]]; then
  ckpt_setting="${ckpt_setting}__gamma_${gamma_tag}"
fi

gamma_override_args=()
if [[ -n "${qha_hybrid_prior_gamma_override}" ]]; then
  gamma_override_args=(--qha_hybrid_prior_gamma_override "${qha_hybrid_prior_gamma_override}")
fi

base_model_name_args=()
if [[ -n "${base_model_name}" ]]; then
  base_model_name_args=(--base_model_name "${base_model_name}")
fi

base_checkpoint_id_args=()
if [[ -n "${base_checkpoint_id}" ]]; then
  base_checkpoint_id_args=(--base_checkpoint_id "${base_checkpoint_id}")
fi

base_checkpoint_root_tag_args=()
if [[ -n "${base_checkpoint_root_tag}" ]]; then
  base_checkpoint_root_tag_args=(--base_checkpoint_root_tag "${base_checkpoint_root_tag}")
fi

horizon_ts_forget_rho_args=()
if [[ -n "${horizon_ts_forget_rho_override}" ]]; then
  horizon_ts_forget_rho_args=(--horizon_ts_forget_rho "${horizon_ts_forget_rho_override}")
fi

horizon_ts_update_eta_args=()
if [[ -n "${horizon_ts_update_eta_override}" ]]; then
  horizon_ts_update_eta_args=(--horizon_ts_update_eta "${horizon_ts_update_eta_override}")
fi

horizon_ts_kernel_bandwidth_args=()
if [[ -n "${horizon_ts_kernel_bandwidth_override}" ]]; then
  horizon_ts_kernel_bandwidth_args=(--horizon_ts_kernel_bandwidth "${horizon_ts_kernel_bandwidth_override}")
fi

cd "${REPO_ROOT}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config "policy/${policy_name}/deploy_policy.yml" \
    --eval_tag "${eval_tag}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --base_model_type "${policy_name}" \
    --ckpt_type "${train_config_name}" \
    --ckpt_setting "${ckpt_setting}" \
    --checkpoint_id "${checkpoint_id}" \
    --qha_eval_variant "${qha_eval_variant}" \
    "${checkpoint_root_args[@]}" \
    "${gamma_override_args[@]}" \
    "${base_model_name_args[@]}" \
    "${base_checkpoint_id_args[@]}" \
    "${base_checkpoint_root_tag_args[@]}" \
    "${horizon_ts_forget_rho_args[@]}" \
    "${horizon_ts_update_eta_args[@]}" \
    "${horizon_ts_kernel_bandwidth_args[@]}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    "${passthrough_overrides[@]}"
