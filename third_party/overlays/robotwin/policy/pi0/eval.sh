#!/bin/bash

# Respect an externally provided memory fraction; otherwise keep the old
# default that was chosen to stay within a 24G budget.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
checkpoint_id=${7:-30000}
qha_eval_variant=${8:-native}
eval_tag=${9:-official}
checkpoint_root_tag=${10:-}
qha_hybrid_prior_gamma_override=${11:-}
base_model_name=${12:-}
base_checkpoint_id=${13:-}
base_checkpoint_root_tag=${14:-}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate

cd ../.. # move to root

checkpoint_root_args=()
if [[ -n "${checkpoint_root_tag}" ]]; then
  checkpoint_root_args=(--checkpoint_root_tag "${checkpoint_root_tag}")
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

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --base_model_type pi0 \
    --eval_tag ${eval_tag} \
    --ckpt_type ${train_config_name} \
    --ckpt_setting ${ckpt_setting} \
    --checkpoint_id ${checkpoint_id} \
    --qha_eval_variant ${qha_eval_variant} \
    "${checkpoint_root_args[@]}" \
    "${gamma_override_args[@]}" \
    "${base_model_name_args[@]}" \
    "${base_checkpoint_id_args[@]}" \
    "${base_checkpoint_root_tag_args[@]}" \
    --seed ${seed} \
    --policy_name ${policy_name}
