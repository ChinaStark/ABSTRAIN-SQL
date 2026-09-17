#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_NAME="${RUN_NAME:-rein_sql_v3_h100_temp10_fresh}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints_${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_DIR}/rollout_data_${RUN_NAME}}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" train_rein_sql.py \
  --config-name rein_sql_v3_h100_1gpu \
  actor_rollout_ref.rollout.temperature=1.0 \
  trainer.resume_mode=disable \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  "$@"
