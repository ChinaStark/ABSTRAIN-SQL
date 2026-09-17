#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/zhiliang/conda_env/verl_dynamic/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PARENT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d "${PARENT_DIR}/data" ]]; then
    DATA_ROOT="${PARENT_DIR}/data"
  else
    DATA_ROOT="${PARENT_DIR}"
  fi
fi
if [[ -z "${TRAIN_FILE:-}" ]]; then
  if [[ -f "${DATA_ROOT}/train_v3.parquet" ]]; then
    TRAIN_FILE="${DATA_ROOT}/train_v3.parquet"
  else
    TRAIN_FILE="${DATA_ROOT}/train_sql_lt60s_with_schema.parquet"
  fi
fi
if [[ -z "${VAL_FILE:-}" ]]; then
  if [[ -f "${DATA_ROOT}/val_v3.parquet" ]]; then
    VAL_FILE="${DATA_ROOT}/val_v3.parquet"
  else
    VAL_FILE="${DATA_ROOT}/dev_with_schema.parquet"
  fi
fi
TRAIN_DB_ROOT="${TRAIN_DB_ROOT:-${DATA_ROOT}/train_databases}"
DEV_DB_ROOT="${DEV_DB_ROOT:-${DATA_ROOT}/dev_databases}"
MODEL_PATH="${MODEL_PATH:-/zhiliang/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_DIR}"
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
