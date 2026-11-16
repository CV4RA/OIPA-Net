#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src:$PWD/artifacts/perception_models"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OIPA_PYTHON:-/root/stet-r1-venv/bin/python}"
if (($#)); then
    exec "$python_bin" -u -m place_retrieval.study.pe_lora --config configs/pe_lora_autodl.yaml "$@"
fi
"$python_bin" -u -m place_retrieval.study.pe_lora --config configs/pe_lora_autodl.yaml
status=$?
printf '%s\n' "$status" > /root/autodl-tmp/OIPA-Net/study_pe_lora_v1.exit
exit "$status"
