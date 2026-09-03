# Level 1 MNIST 验证摘要

- 状态：**PASS**
- 样本数：10000
- 归一化：`uint8 / 255.0`
- 定点数据类型：`ap_fixed<16, 6, AP_RND, AP_SAT>`
- 定点累加类型：`ap_fixed<32, 14, AP_RND, AP_SAT>`
- HLS 工具：`Vitis HLS 2025.2.1`
- 工具状态：开发验证；当前环境未提供 Vivado HLS 2019.2

## 数据摘要

- MNIST 图像：`data/mnist/t10k-images-idx3-ubyte`
- MNIST 图像 SHA-256：`0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7`
- MNIST 标签：`data/mnist/t10k-labels-idx1-ubyte`
- MNIST 标签 SHA-256：`ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2`
- 参数源：`data/lenet_accuracy_1.bin`
- 参数源 SHA-256：`72982df94c34bc968ff37f8b4c68d0df9e4434545c00d7128600dd8c6931c340`
- 参数数量：`44426`

## 结果

| 指标 | 数值 |
| --- | ---: |
| Python 浮点正确数 | 9837 |
| Python 浮点准确率 | 98.37% |
| HLS 定点正确数 | 9838 |
| HLS 定点准确率 | 98.38% |
| 准确率绝对差值 | 0.01% |
| 预测一致率 | 99.99% |
| 不一致样本数 | 1 |
| 90% 门槛 | 通过 |

## 复现命令

```sh
python3 tools/lenet_validation.py make-blob --parameters data/lenet_accuracy_1.bin --images data/mnist/t10k-images-idx3-ubyte --labels data/mnist/t10k-labels-idx1-ubyte --output data/lenet_accuracy_10000.bin
```

```sh
python3 tools/lenet_validation.py run-float --blob data/lenet_accuracy_10000.bin --results results/float_results.csv --summary results/float_summary.json --expected-count 10000
```

```sh
LENET_SKIP_SYNTH=1 LENET_ACCURACY_BLOB="$PWD/data/lenet_accuracy_10000.bin" LENET_RESULT_CSV="$PWD/results/hls_results.csv" vivado_hls -f run_hls.tcl
```

```sh
python3 tools/lenet_validation.py compare --float-results results/float_results.csv --hls-results results/hls_results.csv --report results/validation_report.json --mismatches results/mismatches.csv --threshold 90
```

结果文件：

- 浮点结果：`results/float_results.csv`
- HLS 结果：`results/hls_results.csv`
- 不一致列表：`results/mismatches.csv`
