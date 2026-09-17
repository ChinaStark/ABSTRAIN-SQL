#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT_REL="${DATA_ROOT_REL:-.}"
GENERATED_DIR="${GENERATED_DIR:-${PROJECT_DIR}/generated_data}"
RAW_TRAIN_FILE="${RAW_TRAIN_FILE:-${DATA_ROOT_REL}/train_sql_lt60s_with_schema.parquet}"
RAW_VAL_FILE="${RAW_VAL_FILE:-${DATA_ROOT_REL}/dev_with_schema.parquet}"

cd "${PROJECT_DIR}"
mkdir -p "${GENERATED_DIR}"

[[ -f "${RAW_TRAIN_FILE}" ]] || { echo "Missing training parquet: ${RAW_TRAIN_FILE}" >&2; exit 1; }
[[ -f "${RAW_VAL_FILE}" ]] || { echo "Missing validation parquet: ${RAW_VAL_FILE}" >&2; exit 1; }

"${PYTHON_BIN}" build_v3_train_data.py \
  "${RAW_TRAIN_FILE}" \
  "${GENERATED_DIR}/train_v3.parquet"

"${PYTHON_BIN}" build_v3_train_data.py \
  "${RAW_VAL_FILE}" \
  "${GENERATED_DIR}/val_v3.parquet"

echo "Prepared: ${GENERATED_DIR}/train_v3.parquet"
echo "Prepared: ${GENERATED_DIR}/val_v3.parquet"
