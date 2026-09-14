# FPGA LeNet HLS

本仓库实现面向 MNIST 的 LeNet 推理加速器，使用 Vivado/Vitis HLS 完成功能仿真和
综合验证。当前主线为任务 2（LeNet 路线），Level 1 的标准 MNIST 验证已通过；
Level 2 自采数据已完成一轮可复现的 Python 与 HLS C Simulation 开发验证，正式验收仍需
补齐数据来源和背景负样本说明。跨层FC资源复用与逐行权重缓存两个模块均已完成C仿真和HLS综合。

完整的课程要求对照见[当前验收状态](docs/当前验收状态.md)，模块框图及地址/控制说明见[架构说明](docs/架构与调度说明.md)。课程仅要求仿真与综合，不要求上板。

## 当前结果

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| Level 1 MNIST（10,000 样本） | PASS | [Level 1 摘要](level1/results/level1_summary.md) |
| 16 位浮点/定点对照 | 98.37% / 98.38%，预测一致率 99.99% | [机器可读报告](level1/results/validation_report.json) |
| 定点位宽扫描 W=8..16 | COMPLETE | [整理版报告与图表](level1/results/numerical_precision/) |
| 推荐配置 | `data_t = ap_fixed<10,6,AP_RND,AP_SAT>` | HLS ≥90%，相对 16 位损失 ≤0.5 pp |
| 混合精度/流水化/AXI组织 | 五个新配置，1000张；W8 I1准确率98.3%，保留时钟未达标反例 | [实验报告和图表](mixed_precision/README.md) |
| Bonus QAT | W8 I6 PTQ 56.93%，QAT软件97.02%，HLS CSim 96.98%，标准MNIST 10,000张 | [QAT补充实验](bonus/README.md) |
| C/RTL协同仿真 | 16位逐行缓存版，两次真实参数调用Verilog PASS | [RTL周期与日志](rtl_validation/README.md) |
| 分层参考/故障定位 | 1000张定点参考与已有HLS一致；混合精度10000张Python为98.40%；20张HLS七层逐位一致、故障定位通过 | [结果与边界](layer_validation/README.md) |
| Level 2 自采数据 | 开发验证完成：503张，直接缩放22.27%，当前预处理11.13%；HLS CSim 503/503，Python/HLS一致率100% | [Level 2 说明](level2/README.md)、[执行记录](docs/Level2自采数据执行记录.md) |
| 逐行权重缓存 | 1000张全部logits一致；共享版BRAM 46→14，DSP保持3 | [代码、结果与报告](row_cache/README.md) |
| 跨层FC资源复用 | 1000张全部logits一致；DSP 10→3，BRAM 13→46 | [实验报告、原始结果和RTL](resource_reuse/README.md) |

定点扫描使用官方 MNIST test split、XC7Z020、10 ns 目标周期和 Vitis HLS 2025.2.1。
W10 是满足门槛的最小位宽：HLS 准确率 98.00%，相对 W16 损失 0.38 个百分点；
综合估计 LUT 节省 1.39%、BRAM18K 节省 21.43%，DSP 不变，延迟处于同一估计量级。

![Accuracy versus data width](level1/results/numerical_precision/accuracy_vs_width.png)

![Synthesis resources versus data width](level1/results/numerical_precision/resources_vs_width.png)

## Bonus QAT 复现与 PTQ 对比

该 bonus 用于回答低比特 PTQ 误差过大的问题。实验不改动现有 HLS 源码；所有新增输入、
日志和预测输出都放在 `bonus/` 下，HLS 工作目录为 `bonus/hls_work/`。

在远程服务器 `/home/ubuntu/workspace/FPGA` 运行软件 QAT：

```sh
/home/ubuntu/workspace/lenet_level1/.venv/bin/python bonus/qat_mnist.py \
  --data-dir /home/ubuntu/Documents/case1_mlp/data \
  --output-dir bonus/results/w8a8_i6_qat \
  --weight-bits 8 \
  --weight-integer-bits 6 \
  --activation-bits 8 \
  --activation-integer-bits 6 \
  --epochs 2 \
  --batch-size 256
```

继续运行 QAT 权重的 HLS C Simulation：

```sh
bonus/run_qat_hls.sh
```

该脚本会导出 `qat_quantized_state.pt` 为 HLS testbench 使用的 blob，并设置
`LENET_DATA_W=8`、`LENET_SKIP_SYNTH=1` 和独立的 `LENET_HLS_WORKSPACE`。它只执行
C Simulation，不运行综合，不覆盖 Level 1 的原始工作区。

同一标准 MNIST test split（10,000 张）上的结果如下：

| 实验 | 数据格式 | 样本数 | 准确率 | 正确数 | logit RMSE vs float | 证据 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Float baseline | float32 | 10,000 | 98.37% | 9837 | - | `bonus/results/w8a8_i6_qat/metrics.csv` |
| PTQ 软件同配置 | `ap_fixed<8,6>` fake-quant | 10,000 | 56.93% | 5693 | 5.6483 | `bonus/results/w8a8_i6_qat/metrics.csv` |
| QAT 软件同配置 | `ap_fixed<8,6>` fake-quant | 10,000 | 97.02% | 9702 | 3.2756 | `bonus/results/w8a8_i6_qat/metrics.csv` |
| QAT HLS CSim | `ap_fixed<8,6,AP_RND,AP_SAT>` | 10,000 | 96.98% | 9698 | - | `bonus/results/w8a8_i6_qat_hls/hls_summary.json` |

结论：直接 PTQ 在 W8 I6 下只剩 56.93%，QAT 两轮微调后恢复到 97.02%；同一 QAT
权重经过 HLS CSim 得到 96.98%，与软件 fake-quant 仅差 0.04 个百分点，并通过 90%
准确率门槛。已有 `mixed_precision` 中的 `w8i6_serial` 是另一个 PTQ 控制实验：
1000 张 MNIST、权重 8 位整数 6 位、激活/bias 保持 16 位，准确率 59.7%；它用于说明
低比特权重在粗整数位配置下同样会明显退化，但控制变量与本 bonus 不完全相同。

详细方法、命令和边界见 [`bonus/README.md`](bonus/README.md)。本 bonus 目前只完成
软件训练和 HLS C Simulation，未新增综合、RTL 协同仿真或资源时序结论。

## Level 2 自采数据复现

当前本地数据位于 `data/data/raw/0` 到 `data/data/raw/9`，原始图片不随仓库提交，根目录
`data/` 已由 `.gitignore` 忽略。数据来源目前仍待项目组确认，因此结果应称为开发验证数据，
不能直接替代正式的组员实拍验收数据。

在仓库根目录执行 Python 全量评估：

```sh
python3 level2/tools/level2_validation.py evaluate-dataset \
  --input data/data/raw \
  --parameters level1/data/lenet_accuracy_1.bin \
  --output level2/results/self_collected
```

若本机已配置 Vivado HLS 或 Vitis HLS，再执行同批数据的 C Simulation 和 Python/HLS 对照：

```sh
bash level2/run_self_collected_hls.sh
```

Windows 环境可在完成 Python 评估后运行 `level2/run_self_collected_hls.bat`。该脚本只执行
C Simulation；新增输入不改变 RTL，因此不需要为这批数据重复综合、启动 Vivado GUI 或上板。
当前实测使用 Vitis HLS 2025.2.1，完整结果为 56/503（11.13%），与 Python 预测逐样本一致。

结果和限制见：[Level 2 执行记录](docs/Level2自采数据执行记录.md)、[实验报告](level2/results/self_collected/experiment_report.md)、[机器可读摘要](level2/results/self_collected/summary.json)。

## 目录

- [`level1/`](level1/)：LeNet HLS 源码、testbench、验证工具和运行说明。
- [`level1/results/numerical_precision/`](level1/results/numerical_precision/)：可直接审阅的位宽扫描结果、CSV 和图表。
- [`level2/`](level2/)：真实场景预处理和压力测试工具。
- [`level2/results/self_collected/`](level2/results/self_collected/)：自采数据开发验证的清单、CSV、HLS 对照、图表和报告。
- [`bonus/`](bonus/)：QAT量化感知训练补充实验，比较低比特PTQ与QAT在标准MNIST上的误差。
- [`layer_validation/`](layer_validation/)：独立整数参考、分层分析与故障定位器；20张HLS逐层验证已通过。
- [`mixed_precision/`](mixed_precision/)：权重/激活异构精度、共享核流水与AXI接口组织的对照。
- [`rtl_validation/`](rtl_validation/)：两交易Verilog协同仿真、原始日志与周期可视化。
- [`row_cache/`](row_cache/)：逐行权重缓存，含三架构资源对比、逐样本一致性验证和复现脚本。
- [`resource_reuse/`](resource_reuse/)：16位、同工具原版/共享FC对比，含源码、1000张逐样本结果、综合报告、RTL与图表。
- [`docs/任务要求与LeNet路线核查.md`](docs/任务要求与LeNet路线核查.md)：课程要求与当前完成度核查。
- [`openspec/changes/add-numerical-precision-experiment/`](openspec/changes/add-numerical-precision-experiment/)：本次实验的 OpenSpec 设计与任务记录。

## 复现提示

完整实验需要 10,000 样本 blob、相邻元数据文件和可用的 HLS 环境。命令和数据准备
说明见 [`level1/README.md`](level1/README.md)。逐样本 CSV、HLS 日志和综合工作区
体量较大，按 `.gitignore` 保留在本地；仓库中的汇总材料足以审阅结果并指导复跑。
