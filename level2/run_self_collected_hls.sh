#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
output_dir="${LEVEL2_SELF_COLLECTED_OUTPUT:-$repo_root/level2/results/self_collected}"
blob="$output_dir/preprocessed_self_collected.bin"
result_csv="$output_dir/hls_results.csv"
workspace="${LENET_HLS_WORKSPACE:-$repo_root/level2/hls_work/self_collected_w16}"
hls_command="${VIVADO_HLS_CMD:-vivado_hls}"

if [[ ! -f "$blob" ]]; then
    echo "ERROR: missing $blob; run evaluate-dataset first." >&2
    exit 2
fi

if [[ -n "${VIVADO_HLS_ROOT:-}" && -f "$VIVADO_HLS_ROOT/settings64.sh" ]]; then
    # shellcheck disable=SC1090
    source "$VIVADO_HLS_ROOT/settings64.sh"
fi
if ! command -v "$hls_command" >/dev/null 2>&1 && [[ "$hls_command" == "vivado_hls" ]] \
    && [[ -x "${XILINX_VITIS:-}/bin/loader" ]]; then
    hls_command="${XILINX_VITIS}/bin/loader"
fi
if ! command -v "$hls_command" >/dev/null 2>&1; then
    echo "ERROR: Vivado HLS command '$hls_command' is unavailable." >&2
    exit 3
fi

hls_argv=("$hls_command")
if [[ "$(basename "$hls_command")" == "loader" ]]; then
    hls_argv+=("-exec" "vitis_hls")
fi

cd "$repo_root"
LENET_DATA_W=16 \
LENET_SKIP_SYNTH=1 \
LENET_SKIP_CSIM=0 \
LENET_ACCURACY_THRESHOLD=0 \
LENET_HLS_WORKSPACE="$workspace" \
LENET_ACCURACY_BLOB="$blob" \
LENET_RESULT_CSV="$result_csv" \
"${hls_argv[@]}" -f "$repo_root/level1/run_hls.tcl"

python3 "$repo_root/level1/tools/lenet_validation.py" compare \
    --float-results "$output_dir/python_results.csv" \
    --hls-results "$result_csv" \
    --report "$output_dir/python_hls_comparison.json" \
    --mismatches "$output_dir/python_hls_mismatches.csv" \
    --threshold 0

python3 "$repo_root/level2/tools/level2_validation.py" finalize-hls \
    --summary "$output_dir/summary.json" \
    --comparison "$output_dir/python_hls_comparison.json" \
    --report "$output_dir/experiment_report.md" \
    --tool-version "${HLS_TOOL_VERSION:-Vitis HLS 2025.2.1}"
