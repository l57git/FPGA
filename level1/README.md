# LeNet Level 1

Vivado HLS 2019.2 implementation of LeNet inference for MNIST.

## Network

`Conv1 + bias + ReLU -> MaxPool -> Conv2 + bias + ReLU -> MaxPool -> FC1 + bias + ReLU -> FC2 + bias + ReLU -> FC3 + bias -> Argmax`

The hardware inference path intentionally omits softmax because softmax does not change the argmax classification result.

## Environment

- Vivado HLS 2019.2
- Target: `xc7z020clg400-1`
- Clock: 10 ns

## Run C simulation

Open a Vivado HLS Command Prompt in this directory and run:

```bat
set LENET_ACCURACY_BLOB=%CD%\data\lenet_accuracy_1.bin
vivado_hls -f run_hls.tcl
```

The included blob contains one smoke-test sample. Its verified result is label 7, prediction 7.

## Complete MNIST validation

The validation tools use the official MNIST test split and normalize pixels with `uint8 / 255.0`. Download and unpack these files into `data/mnist/`:

```sh
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz -o data/mnist/t10k-images-idx3-ubyte.gz
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz -o data/mnist/t10k-labels-idx1-ubyte.gz
gzip -df data/mnist/t10k-images-idx3-ubyte.gz data/mnist/t10k-labels-idx1-ubyte.gz
```

Validate the inputs, generate the HLS batch blob, and run the NumPy reference model:

```sh
python3 tools/lenet_validation.py validate-mnist \
  --images data/mnist/t10k-images-idx3-ubyte \
  --labels data/mnist/t10k-labels-idx1-ubyte
python3 tools/lenet_validation.py make-blob \
  --parameters data/lenet_accuracy_1.bin \
  --images data/mnist/t10k-images-idx3-ubyte \
  --labels data/mnist/t10k-labels-idx1-ubyte \
  --output data/lenet_accuracy_10000.bin
python3 tools/lenet_validation.py run-float \
  --blob data/lenet_accuracy_10000.bin \
  --results results/float_results.csv \
  --summary results/float_summary.json \
  --expected-count 10000
```

Run HLS C simulation with the optional result CSV enabled, using the HLS environment described in the repository `CLAUDE.md`:

```sh
mkdir -p results
LENET_SKIP_SYNTH=1 \
LENET_ACCURACY_BLOB="$PWD/data/lenet_accuracy_10000.bin" \
LENET_RESULT_CSV="$PWD/results/hls_results.csv" \
vivado_hls -f run_hls.tcl
```

Compare both result files and apply the Level 1 fixed-point threshold:

```sh
python3 tools/lenet_validation.py compare \
  --float-results results/float_results.csv \
  --hls-results results/hls_results.csv \
  --report results/validation_report.json \
  --mismatches results/mismatches.csv \
  --threshold 90
```

The complete IDX files, batch blob, and per-sample CSV files are reproducible inputs or outputs and are ignored by Git. The JSON summaries, mismatch CSV, and validation report can be retained as experiment evidence.

## Windows and Vivado HLS 2019.2

After placing the two unpacked MNIST test files in `data/mnist/`, run:

```bat
run_validation_windows.bat
```

The script validates the MNIST files, creates the 10,000-sample blob, runs the Python float32 reference, runs Vivado HLS 2019.2 C simulation, and compares all predictions. Results are written to `results/windows_2019_2/` so they do not overwrite evidence produced by another tool version.

The default installation directory is `E:\use\cpu\Vivado\2019.2`. For a different installation, set it before running:

```bat
set VIVADO_HLS_ROOT=D:\Xilinx\Vivado\2019.2
run_validation_windows.bat
```

## 定点位宽/数值精度实验

`tools/precision_experiment.py` 按固定规则扫描共享 `data_t` 的 8～16 位：
`ap_fixed<W,6,AP_RND,AP_SAT>`，累加器固定为
`ap_fixed<32,14,AP_RND,AP_SAT>`。每档使用完整官方 MNIST test split，先运行
16 位基线，再运行 8～15 位；HLS CSim 与综合结果分别保存在独立目录。Python
结果标记为层边界量化近似，正式位宽选择只使用 HLS CSim。

完整实验要求使用 `make-blob` 生成的 10,000 样本 blob、相邻或显式指定的元数据，
以及可用的 HLS 命令环境。输出目录必须是新目录或空目录：

```sh
python3 tools/precision_experiment.py run \
  --blob data/lenet_accuracy_10000.bin \
  --blob-metadata results/blob_metadata.json \
  --hls-executable /path/to/vivado_hls \
  --output results/numerical_precision_YYYYMMDD
```

没有 HLS 环境时只能显式运行软件检查；该结果不会生成正式推荐：

```sh
python3 tools/precision_experiment.py run \
  --blob data/lenet_accuracy_10000.bin \
  --blob-metadata results/blob_metadata.json \
  --python-only \
  --output results/numerical_precision_python_YYYYMMDD
```

已有完整结果可离线重新汇总，不会再次调用 HLS：

```sh
python3 tools/precision_experiment.py report \
  --output results/numerical_precision_YYYYMMDD
```

选择规则固定为 HLS 准确率至少 90%，且相对本次 16 位 HLS 基线下降不超过
0.5 个百分点；只有九档 CSim、综合和报告解析都成功时才推荐最小位宽。输出包含
`manifest.json`、`ranges.json`、`precision_sweep.csv`、`summary.json`、
`report.md` 和两张曲线图。资源和延迟是指定 HLS 工具的综合估计，不是上板实测；
动态范围重缩放、累加器扫描、混合精度、QAT 和训练不在本实验范围内。

当前仓库环境可用 Vitis HLS 2025.2.1；Vivado HLS 2019.2 必须在独立环境、独立
输出目录复跑，不能混用两套工具链的资源数据。

本次完整扫描的整理版结果、CSV 汇总和曲线图见
[`results/numerical_precision/`](results/numerical_precision/)。完整逐样本 CSV、
HLS 日志和综合工作区仍保存在本地的 `results/numerical_precision_*` 目录中，
并由 Git 忽略；它们可通过上面的命令和 `report` 子命令重新生成。
