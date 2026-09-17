#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GENERATED_DIR="${GENERATED_DIR:-generated_data}"
TRAIN_FILE="${TRAIN_FILE:-${GENERATED_DIR}/train_v3.parquet}"
VAL_FILE="${VAL_FILE:-${GENERATED_DIR}/val_v3.parquet}"
if [[ -z "${DATA_ROOT_REL:-}" ]]; then
  if [[ -d "${PROJECT_DIR}/train_databases" ]]; then
    DATA_ROOT_REL="."
  else
    DATA_ROOT_REL=".."
  fi
fi
TRAIN_DB_ROOT="${TRAIN_DB_ROOT:-${DATA_ROOT_REL}/train_databases}"
DEV_DB_ROOT="${DEV_DB_ROOT:-${DATA_ROOT_REL}/dev_databases}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="${DATA_ROOT_REL}/models"
  MODEL_SNAPSHOT="${DATA_ROOT_REL}/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
  if [[ -f "${MODEL_SNAPSHOT}/config.json" ]]; then
    MODEL_PATH="${MODEL_SNAPSHOT}"
  fi
fi

[[ -f "${TRAIN_FILE}" ]] || { echo "Missing prepared training parquet: ${TRAIN_FILE}; run ./prepare_data.sh first" >&2; exit 1; }
[[ -f "${VAL_FILE}" ]] || { echo "Missing prepared validation parquet: ${VAL_FILE}; run ./prepare_data.sh first" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" train_rein_sql.py \
  --config-name rein_sql_v3_h100_2gpu \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  reward.custom_reward_function.reward_kwargs.db_root="${TRAIN_DB_ROOT}" \
  reward.custom_reward_function.reward_kwargs.dev_db_root="${DEV_DB_ROOT}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32000 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32000 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
  trainer.experiment_name=rein_sql_v3_2gpu_48gb \
  'trainer.logger=[console,tensorboard,wandb]' \
  "$@"
