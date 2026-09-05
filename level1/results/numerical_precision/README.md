# LeNet 定点位宽/数值精度实验

这是完整扫描的仓库内整理版证据。原始逐样本 CSV、HLS 日志和综合工作区没有纳入
Git；它们位于本地的 `level1/results/numerical_precision_vitis_20260905_fast/`，
可用 `precision_experiment.py report` 离线重新汇总。

## 实验设置

- 数据：官方 MNIST test split，10,000 个样本，输入归一化为 `uint8 / 255.0`。
- 扫描变量：共享 `data_t = ap_fixed<W,6,AP_RND,AP_SAT>`，W=8..16；小数位为 W-6。
- 固定变量：累加器 `ap_fixed<32,14,AP_RND,AP_SAT>`、LeNet 结构、参数、器件 `xc7z020clg400-1`、目标周期 10 ns。
- 工具：Vitis HLS 2025.2.1；每一档均完成 HLS CSim、综合和综合报告解析。
- 选型门槛：HLS 准确率 ≥90%，且相对 16 位 HLS 基线损失 ≤0.50 个百分点。

## 结论

推荐 `W=10`，因为它是满足全部证据完整性和精度门槛的最小位宽：HLS 准确率
98.00%，相对 16 位损失 0.38 个百分点。W8 和 W9 不满足门槛；W11～W16 均满足，
但继续增加位宽没有带来可观的准确率收益。

相对 W16，W10 的综合估计 LUT 从 18,020 降至 17,770（节省 1.39%），BRAM18K
从 14 降至 11（节省 21.43%），DSP 保持 13，FF 从 12,815 增至 13,140（增加
2.54%）。最大延迟周期为 484,541，按 10 ns 目标周期折算约 4,845,410 ns；该数值
来自综合估计，不是上板实测。

## 扫描结果

| W | 小数位 F | HLS 准确率 | 相对 W16 损失 (pp) | HLS/浮点预测一致率 | LUT | FF | DSP | BRAM18K | 最大周期 | 选型 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8 | 2 | 55.99% | 42.39 | 56.34% | 17,775 | 11,846 | 13 | 10 | 484,443 | — |
| 9 | 3 | 96.49% | 1.89 | 97.30% | 17,779 | 13,071 | 13 | 10 | 484,541 | — |
| 10 | 4 | 98.00% | 0.38 | 99.00% | 17,770 | 13,140 | 13 | 11 | 484,541 | 推荐 |
| 11 | 5 | 98.29% | 0.09 | 99.54% | 17,777 | 13,220 | 13 | 13 | 484,541 | 可选 |
| 12 | 6 | 98.30% | 0.08 | 99.85% | 17,793 | 13,301 | 13 | 13 | 484,541 | 可选 |
| 13 | 7 | 98.40% | -0.02 | 99.88% | 17,835 | 13,382 | 13 | 13 | 484,541 | 可选 |
| 14 | 8 | 98.35% | 0.03 | 99.94% | 17,882 | 13,451 | 13 | 14 | 484,541 | 可选 |
| 15 | 9 | 98.39% | -0.01 | 99.98% | 17,951 | 13,535 | 13 | 14 | 484,541 | 可选 |
| 16 | 10 | 98.38% | 0.00 | 99.99% | 18,020 | 12,815 | 13 | 14 | 484,541 | 基线 |

`Python approx` 是层边界量化近似，用于辅助检查；正式位宽选择只使用 HLS CSim。
W8 只有 2 个小数位，准确率显著下降是固定小数位预算下的预期结果。本实验没有
引入动态重缩放、混合精度或累加器扫描。

![Accuracy versus data width](accuracy_vs_width.png)

![Synthesis resources versus data width](resources_vs_width.png)

## 文件

- [`precision_sweep.csv`](precision_sweep.csv)：九档完整指标的机器可读汇总。
- [`summary.json`](summary.json)：去除本机绝对路径后的结论摘要。
- `accuracy_vs_width.png`：准确率—位宽曲线。
- `resources_vs_width.png`：LUT、FF、DSP、BRAM18K—位宽曲线。

## 复现

从 `level1/` 执行：

```sh
python3 tools/precision_experiment.py run \
  --blob data/lenet_accuracy_10000.bin \
  --blob-metadata results/blob_metadata.json \
  --hls-executable /path/to/vivado_hls \
  --output results/numerical_precision_YYYYMMDD
```

已有完整输出时只重建报告：

```sh
python3 tools/precision_experiment.py report \
  --output results/numerical_precision_YYYYMMDD
```

本次证据的 blob SHA-256 为
`398a7b39c9b7cc44cb529c7b35c884505c1ab220caa8b719495f37803406cf72`，参数区 SHA-256
为 `d4041b04b172a8c19ecfed1c43e779f35af0f0c6fd6aacbbc70f1c0a3dd855bf`。
