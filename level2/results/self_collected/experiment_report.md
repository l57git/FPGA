# Level 2 自采数据 Python 验证报告

## 数据与来源

输入目录为 `data/data/raw`，共 503 张有效图片，按父目录 0–9 作为标签。来源状态：`pending_confirmation`；项目组仍需确认这些图片是否为成员实际采集。报告不根据文件名推断数据来源。

有效文件全部可解码；尺寸分布为 {'760x1280': 503}，模式分布为 {'RGB': 503}。排除文件 0 个，详见 `manifest.csv`。

## 预处理

灰度化 → 背景估计与极性统一 → ROI 裁剪 → 保持比例缩放 → 质心对齐 → 28×28 单通道归一化。直接缩放作为无 ROI 对照。

## Python 结果

| 输入 | 正确数 | 样本数 | 准确率 |
| --- | ---: | ---: | ---: |
| 直接缩放 | 112 | 503 | 22.27% |
| 完整预处理 | 56 | 503 | 11.13% |
| 预处理变化 | — | — | -11.13 个百分点 |

生成的预处理输入已回读校验，标签和 28×28 浮点图像逐项一致，可直接用于现有 HLS testbench。

## 失败分析

样例图显示，部分照片在纸张上下边缘带有深色页框；当前基于边缘背景偏差的 ROI 检测会把页框当作前景，预处理结果中因此出现横向条带并压缩数字。作为定位探针，在每类取 2 张的 20 张样本上，原流程正确 3/20；先裁掉约 5% 边缘后为 7/20，裁掉约 8% 后为 8/20。该探针不是全量结果，正式报告仍采用上表的原始预处理结果；需要在全量数据上复验边缘抑制规则后才能替换预处理版本。

## HLS 与 Python 定点对照

使用 Vitis HLS 2025.2.1 对同一批 503 张预处理图片执行 C Simulation，目标器件为 `xc7z020-clg400-1`、时钟为 10 ns；日志报告 `CSim done with 0 errors`。HLS 定点结果为 56/503，准确率 11.13%；Python 与 HLS 逐样本预测一致率为 100.00%，不一致样本数为 0。

比较命令使用 `threshold=0` 仅表示取消自采域数据的 90% 门槛，不能把工具返回的 PASS 当作准确率达标。当前自采数据的绝对准确率仍为 11.13%，低准确率主要反映数据域和预处理问题。

## 背景测试

当前没有提供独立背景图片；十分类器没有拒识输出，因此尚未计算背景误检率。

## 复现

```sh
python3 level2/tools/level2_validation.py evaluate-dataset \
  --input data/data/raw \
  --parameters level1/data/lenet_accuracy_1.bin \
  --output level2/results/self_collected
```

复现时使用 `level2/run_self_collected_hls.sh`（或 Windows 等价脚本）运行 HLS C Simulation；脚本会自动使用 `XILINX_VITIS/bin/loader`，并用 `level1/tools/lenet_validation.py compare` 对照 `python_results.csv`。
