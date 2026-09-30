#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QHA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${QHA_ROOT}/../.." && pwd)"

METHOD=""
TASK=""
SETTING=""
MANIFEST=""
QHA_MODEL=""
QHA_STEP="10000"
QHA_ROOT_TAG="pi05_base_aloha_robotwin_full_qha_heldout"
BASE_ROOT_TAG="pi05_base_aloha_robotwin_full"
BASE_MODEL=""
BASE_STEP="20000"
RESULT_ROOT="${QHA_ROOT}/eval_result/heldout_manifest"
GPU="0"
SAVE_VIDEO="0"
EVAL_TAG=""
DRY_RUN=0

dense_candidates() {
  local values=()
  local i
  for i in $(seq 1 50); do values+=("${i}"); done
  (IFS=,; echo "${values[*]}")
}

die() { echo "ERROR: $*" >&2; exit 2; }

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/eval_qha_heldout_manifest.sh \
    --method ahs_only|qha_only|ahs_qha --task TASK --setting demo_clean|demo_randomized \
    --manifest PATH [method/checkpoint options]

This launcher is only for the registered two-task held-out protocol. It pins
all methods to candidates 1..50, expected_round, temperature 1.0, and the
literal seed+instruction manifest. It never uses the legacy random-seed path.

Required:
  --method METHOD                 ahs_only, qha_only, or ahs_qha
  --task TASK                     blocks_ranking_rgb or place_bread_basket
  --setting SETTING               demo_clean or demo_randomized
  --manifest PATH                 Immutable manifest for exactly TASK/SETTING

QHA options (required for qha_only and ahs_qha):
  --qha-model NAME                Formal QHA exp name under QHA root tag
  --qha-step STEP                 QHA checkpoint step (default: 10000)
  --qha-root-tag TAG              QHA checkpoint root tag

Common options:
  --base-root-tag TAG             Base checkpoint root tag
  --base-model NAME               Defaults to pi05_TASK_clean50_qnorm
  --base-step STEP                Base checkpoint step (default: 20000)
  --result-root PATH              Output root (default: pi05_horizon/eval_result/heldout_manifest)
  --gpu ID                        One GPU for this serial evaluator (default: 0)
  --save-video 0|1                Default 0; replan JSONL is always written
  --eval-tag TAG                  Stable result tag; default includes method and manifest hash
  --dry-run                       Print the exact command without evaluating
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --setting) SETTING="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --qha-model) QHA_MODEL="$2"; shift 2 ;;
    --qha-step) QHA_STEP="$2"; shift 2 ;;
    --qha-root-tag) QHA_ROOT_TAG="$2"; shift 2 ;;
    --base-root-tag) BASE_ROOT_TAG="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --base-step) BASE_STEP="$2"; shift 2 ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --save-video) SAVE_VIDEO="$2"; shift 2 ;;
    --eval-tag) EVAL_TAG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ "${METHOD}" == "ahs_only" || "${METHOD}" == "qha_only" || "${METHOD}" == "ahs_qha" ]] || die "invalid --method"
[[ "${TASK}" == "blocks_ranking_rgb" || "${TASK}" == "place_bread_basket" ]] || die "invalid held-out --task"
[[ "${SETTING}" == "demo_clean" || "${SETTING}" == "demo_randomized" ]] || die "invalid --setting"
[[ -f "${MANIFEST}" ]] || die "manifest does not exist: ${MANIFEST}"
[[ "${SAVE_VIDEO}" == "0" || "${SAVE_VIDEO}" == "1" ]] || die "--save-video must be 0 or 1"
[[ "${QHA_STEP}" =~ ^[1-9][0-9]*$ ]] || die "--qha-step must be a positive integer"
[[ "${BASE_STEP}" =~ ^[1-9][0-9]*$ ]] || die "--base-step must be a positive integer"
if [[ "${METHOD}" != "ahs_only" ]]; then
  [[ -n "${QHA_MODEL}" ]] || die "--qha-model is required for ${METHOD}"
fi
if [[ -z "${BASE_MODEL}" ]]; then
  BASE_MODEL="pi05_${TASK}_clean50_qnorm"
fi

MANIFEST="$(realpath "${MANIFEST}")"
MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
if [[ -z "${EVAL_TAG}" ]]; then
  EVAL_TAG="heldout_${METHOD}_${MANIFEST_SHA:0:12}"
fi
DENSE_CANDIDATES="$(dense_candidates)"

COMMON=(
  python script/eval_policy.py
  --config policy/pi05_horizon/deploy_policy.yml
  --eval_tag "${EVAL_TAG}"
  --overrides
  --task_name "${TASK}"
  --task_config "${SETTING}"
  --policy_name pi05_horizon
  --base_model_type pi05_horizon
  --ckpt_type "qha_heldout_${METHOD}"
  --ckpt_setting "${METHOD}__${MANIFEST_SHA:0:12}"
  --episode_manifest_path "${MANIFEST}"
  --eval_root "${RESULT_ROOT}"
  --eval_video_log "${SAVE_VIDEO}"
  --rollouts_parallel 1
  --seed 0
  --horizon_candidates "${DENSE_CANDIDATES}"
  --horizon_exec_mode expected_round
  --horizon_expected_temp 1.0
    --qha_candidate_mode_override dense_full
    --qha_selector_exec_mode_override expected_round
    --qha_selector_expected_temp_override 1.0
    --qha_hybrid_prior_gamma_override 1.0
)

if [[ "${METHOD}" == "ahs_only" ]]; then
  COMMON+=(
    --train_config_name pi05_base_aloha_robotwin_full
    --checkpoint_root_tag "${BASE_ROOT_TAG}"
    --model_name "${BASE_MODEL}"
    --checkpoint_id "${BASE_STEP}"
    --pi0_step 50
    --use_qha False
    --eval_type horizon
  )
elif [[ "${METHOD}" == "qha_only" ]]; then
  COMMON+=(
    # Policy loading needs the per-task base data config for its norm stats;
    # the QHA architecture and weights come from the overlaid head metadata.
    --train_config_name pi05_base_aloha_robotwin_full
    --checkpoint_root_tag "${QHA_ROOT_TAG}"
    --model_name "${QHA_MODEL}"
    --checkpoint_id "${QHA_STEP}"
    --base_checkpoint_root_tag "${BASE_ROOT_TAG}"
    --base_model_name "${BASE_MODEL}"
    --base_checkpoint_id "${BASE_STEP}"
    --pi0_step 50
    --use_qha True
    --qha_infer_mode_override posterior_only
    --eval_type default
  )
else
  COMMON+=(
    --train_config_name pi05_base_aloha_robotwin_full
    --checkpoint_root_tag "${QHA_ROOT_TAG}"
    --model_name "${QHA_MODEL}"
    --checkpoint_id "${QHA_STEP}"
    --base_checkpoint_root_tag "${BASE_ROOT_TAG}"
    --base_model_name "${BASE_MODEL}"
    --base_checkpoint_id "${BASE_STEP}"
    --pi0_step 50
    --use_qha True
    --qha_infer_mode_override hybrid
    --eval_type default
  )
fi

printf 'method=%s\ntask=%s\nsetting=%s\nmanifest_sha256=%s\nbase=%s/%s/%s\n' \
  "${METHOD}" "${TASK}" "${SETTING}" "${MANIFEST_SHA}" "${BASE_ROOT_TAG}" "${BASE_MODEL}" "${BASE_STEP}"
if [[ "${METHOD}" != "ahs_only" ]]; then
  printf 'qha=%s/%s/%s\n' "${QHA_ROOT_TAG}" "${QHA_MODEL}" "${QHA_STEP}"
fi
printf 'command:'; printf ' %q' "${COMMON[@]}"; printf '\n'
if [[ "${DRY_RUN}" -eq 1 ]]; then exit 0; fi

if [[ -f "${QHA_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${QHA_ROOT}/.venv/bin/activate"
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${QHA_ROOT}:${QHA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${COMMON[@]}"
