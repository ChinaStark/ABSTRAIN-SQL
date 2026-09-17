#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/zhiliang/conda_env/verl_dynamic/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" train_rein_sql.py \
  --config-name rein_sql_v3_h100_2gpu \
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
