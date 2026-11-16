#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src:$PWD/diagnostics:$PWD/artifacts/perception_models"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OIPA_PYTHON:-/root/stet-r1-venv/bin/python}"
run_dir=/root/autodl-tmp/OIPA-Net
"$python_bin" -u -m place_retrieval.study.oipa_loss --config configs/oipa_loss_autodl.yaml
status=$?
printf '%s\n' "$status" > "$run_dir/study_loss_v2.exit"
exit "$status"
