# Level 2 自采数据验证执行记录

执行日期：2026-09-09
执行入口：[Level2 自采数据验证行动 spec](Level2自采数据验证行动spec.md)

## 逐项结果

| spec 项目 | 状态 | 实测结果与证据 |
| --- | --- | --- |
| 数据核对 | 完成 | `data/data/raw` 共 503 张有效 PNG；类别 0–9，数量为 50、50、50、51、50、50、50、52、50、50；全部 760×1280、RGB，可解码；文件名 digit 标签与父目录一致；无重复内容、无损坏文件、无排除项。证据：`level2/results/self_collected/manifest.csv`、`summary.json`。 |
| 来源核实 | 阻塞 | 文件名含 `zimage`，但仓库没有采集记录，无法确认是否为组员实拍。当前报告标记 `pending_confirmation`，没有据文件名推断来源。 |
| 分类目录适配 | 完成 | 新增 `evaluate-dataset`，支持 `0..9` 分类子目录、metadata 过滤、文件哈希、尺寸/模式记录和标签冲突检查。 |
| Python 全量评估 | 完成 | 直接缩放 112/503 = 22.27%；当前完整预处理 56/503 = 11.13%；预处理变化 -11.13 个百分点。两组逐样本结果和混淆矩阵已生成。 |
| 预处理输入协议 | 完成 | `preprocessed_self_collected.bin` 含 503 张 28×28 图像，值域 `[0, 0.9882]`；写入后用 `read_lenet_blob(expected_count=503)` 回读，标签和图像逐项一致。该 blob 已被 `.gitignore` 忽略。 |
| Python 测试 | 完成 | `level2/tools` 测试 6/6 通过；Level 1 数据协议测试 6/6 通过；Python 编译、shell 语法和 `git diff --check` 通过。 |
| 失败分析 | 初步完成 | 样例中纸张上下深色页框被 ROI 检测识别为前景，造成横条和数字压缩。20 张定位探针中，边缘裁 5% 为 7/20、裁 8% 为 8/20，优于原流程 3/20；尚未在 503 张上完成全量复验，因此没有替换正式结果。 |
| HLS C Simulation | 完成 | 本机 Vitis HLS 2025.2.1 通过 `loader -exec vitis_hls` 执行，503/503 样本，`CSim done with 0 errors`，耗时约 296.91 秒；目标 `xc7z020-clg400-1`、10 ns。 |
| Python/HLS 对照 | 完成 | Python 与 HLS 均为 56/503、11.1332%；逐样本预测一致率 100%，不一致 0。`python_hls_comparison.json` 已生成；比较阈值设为 0，仅用于允许低准确率数据完成对照。 |
| 背景测试 | 阻塞 | 当前数据目录没有独立背景图片；十分类器没有拒识输出，不能把任意数字预测直接称为背景误检率。 |
| 中期材料 | 部分完成 | 已有数据清单、准确率、混淆矩阵、预处理对比图和可复现命令；来源说明、HLS CSim、背景负样本和预处理修正版仍需补齐。 |

## 当前可复现命令

```sh
python3 level2/tools/level2_validation.py evaluate-dataset \
  --input data/data/raw \
  --parameters level1/data/lenet_accuracy_1.bin \
  --output level2/results/self_collected

python3 -m unittest discover -s level2/tools -p 'test_*.py'
python3 -m unittest discover -s level1/tools -p 'test_lenet_validation.py'

bash level2/run_self_collected_hls.sh
```

本机已安装 Vitis HLS 2025.2.1，脚本会自动使用 `XILINX_VITIS/bin/loader`；项目要求是仿真与综合，不要求完整 Vivado 上板流程。自采数据只改变 testbench 输入，不改变 RTL，因此无需为这批数据重复综合；Level 1 的综合证据继续沿用已有结果。当前仍未在任务书预期的 Vivado HLS 2019.2 下复跑。

## 剩余阻塞

1. 确认图片来源、采集方式和标签规则；若不是组员实拍，报告应改称补充压力数据。
2. 提供无数字背景图片，并明确拒识规则后再计算背景误检率。
3. 如教师要求严格复现 2019.2，需在具备该版本的环境复跑 C Simulation；当前 2025.2.1 结果已作为开发验证证据保留。
4. 在全量数据上复验 5%～8% 边缘抑制方案，再决定是否替换当前 11.13% 的正式预处理结果。
