#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src:$PWD/diagnostics:$PWD/artifacts/perception_models"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir=/root/autodl-tmp/OIPA-Net
python_bin="${OIPA_PYTHON:-/root/stet-r1-venv/bin/python}"
if [ "$#" -gt 0 ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 20; done
fi
"$python_bin" -u -m place_retrieval.study.oipa_itm --config configs/oipa_itm_autodl.yaml
status=$?
printf '%s\n' "$status" > "$run_dir/study_itm_v1.exit"
exit "$status"
