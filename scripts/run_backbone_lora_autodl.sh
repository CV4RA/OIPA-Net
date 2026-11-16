#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/artifacts/perception_models"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OIPA_PYTHON:-/root/stet-r1-venv/bin/python}"
exec "$python_bin" -u -m place_retrieval.study.backbone_lora --config configs/backbone_lora_autodl.yaml "$@"
