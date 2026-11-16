#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src:$PWD/diagnostics:$PWD/artifacts/perception_models"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OIPA_PYTHON:-/root/stet-r1-venv/bin/python}"
run_dir=/root/autodl-tmp/OIPA-Net

run_stage() {
    local module="$1" study="$2" status=0
    local marker="$run_dir/$study.exit"
    if [[ -f "$marker" && "$(<"$marker")" == 0 && -f "$run_dir/$study/completion.json" ]]; then
        return
    fi
    "$python_bin" -u -m "place_retrieval.study.$module" \
        --config "configs/${module}_sfxl_autodl.yaml" >> "$run_dir/$study.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$marker.tmp"
    mv "$marker.tmp" "$marker"
    return "$status"
}

run_stage oipa study_sfxl_v1
run_stage pe_lora study_sfxl_pe_lora_v1
run_stage oipa_backbones study_sfxl_backbones_v1
run_stage backbone_lora study_sfxl_backbone_lora_v1
run_stage oipa_itm study_sfxl_itm_v1
