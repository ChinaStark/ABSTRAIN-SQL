#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  "${PYTHON_BIN}" -m wandb login "${WANDB_API_KEY}"
else
  echo "W&B will prompt for an API key. Create one at https://wandb.ai/authorize"
  "${PYTHON_BIN}" -m wandb login
fi
