#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/workspace/lenet_level1/.venv/bin/python}"
data_dir="${MNIST_DATA_DIR:-/home/ubuntu/Documents/case1_mlp/data}"
result_dir="${QAT_HLS_RESULT_DIR:-$repo_root/bonus/results/w8a8_i6_qat_hls}"
software_metrics="${QAT_SOFTWARE_METRICS:-$repo_root/bonus/results/w8a8_i6_qat/metrics.csv}"
state_path="${QAT_STATE:-$repo_root/bonus/results/w8a8_i6_qat/qat_quantized_state.pt}"
data_width="${LENET_DATA_W:-8}"
accuracy_threshold="${QAT_HLS_ACCURACY_THRESHOLD:-90}"
hls_workspace="${QAT_HLS_WORKSPACE:-$repo_root/bonus/hls_work/w8a8_i6_qat_csim}"

mkdir -p "$result_dir"

input_blob="$result_dir/qat_hls_input.bin"
blob_metadata="$result_dir/qat_hls_input.json"
predictions_csv="$result_dir/hls_predictions.csv"
run_log="$result_dir/run.log"
summary_json="$result_dir/hls_summary.json"
report_md="$result_dir/hls_experiment_report.md"

"$python_bin" "$repo_root/bonus/export_qat_hls_blob.py" \
  --state "$state_path" \
  --data-dir "$data_dir" \
  --output "$input_blob" \
  --metadata "$blob_metadata" \
  --data-bits "$data_width" \
  --data-integer-bits 6

if [[ -n "${HLS_ENV:-}" && -f "$HLS_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$HLS_ENV"
elif [[ -f /home/ubuntu/workspace/lenet_level1/scripts/env_hls2019.sh ]]; then
  # shellcheck source=/dev/null
  source /home/ubuntu/workspace/lenet_level1/scripts/env_hls2019.sh
fi

if [[ -z "${VIVADO_HLS_BIN:-}" ]]; then
  if command -v vivado_hls >/dev/null 2>&1; then
    VIVADO_HLS_BIN="$(command -v vivado_hls)"
  elif [[ -x /opt/vlab/vivado/Xilinx/Vivado/2019.1/bin/vivado_hls ]]; then
    VIVADO_HLS_BIN="/opt/vlab/vivado/Xilinx/Vivado/2019.1/bin/vivado_hls"
  else
    echo "vivado_hls not found. Set VIVADO_HLS_BIN=/path/to/vivado_hls" >&2
    exit 127
  fi
fi

set +e
LENET_DATA_W="$data_width" \
LENET_ACCURACY_BLOB="$input_blob" \
LENET_RESULT_CSV="$predictions_csv" \
LENET_ACCURACY_THRESHOLD=0 \
LENET_SKIP_SYNTH=1 \
LENET_CSIM_OPT=1 \
LENET_HLS_WORKSPACE="$hls_workspace" \
"$VIVADO_HLS_BIN" -f "$repo_root/level1/run_hls.tcl" > "$run_log" 2>&1
hls_return_code=$?
set -e

"$python_bin" "$repo_root/bonus/summarize_qat_hls.py" \
  --predictions "$predictions_csv" \
  --log "$run_log" \
  --summary "$summary_json" \
  --report "$report_md" \
  --software-metrics "$software_metrics" \
  --input-blob "$input_blob" \
  --hls-workspace "$hls_workspace" \
  --data-width "$data_width" \
  --hls-return-code "$hls_return_code" \
  --accuracy-threshold "$accuracy_threshold"

exit "$hls_return_code"
