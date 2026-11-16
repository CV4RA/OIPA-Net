#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src:$PWD/diagnostics:$PWD/artifacts/perception_models"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir=/root/autodl-tmp/OIPA-Net
python_bin="${OIPA_PYTHON:-/root/stet-r1-venv/bin/python}"
if [ "$#" -gt 0 ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 20; done
    if [ ! -f "$run_dir/study_loss_v2/completion.json" ]; then
        printf '%s\n' 'The preceding PE loss study did not complete.'
        printf '%s\n' 1 > "$run_dir/study_backbones_v1.exit"
        exit 1
    fi
fi
"$python_bin" -u -m place_retrieval.study.oipa_backbones --config configs/oipa_backbones_autodl.yaml
status=$?
printf '%s\n' "$status" > "$run_dir/study_backbones_v1.exit"
exit "$status"
