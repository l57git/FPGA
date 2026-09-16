# FPGA LeNet HLS

本仓库实现面向 MNIST 的 LeNet 推理加速器，使用 Vivado/Vitis HLS 完成功能仿真和综合验证。当前主线为任务 2（LeNet 路线），Level 1 的标准 MNIST 验证已通过。跨层 FC 资源复用与逐行权重缓存两个模块均已完成 C 仿真和 HLS 综合。

Level 2 已完成旧预处理的 Python 与 HLS C Simulation 开发验证，并新增预处理改进 v1。在保持原始图片、标签和模型参数不变的情况下，503 张开发验证图片的 Python 浮点识别准确率由旧预处理的 11.13% 提升至 85.29%；其中，内部图像级留出测试集为 126/145（86.90%）。新版本已完成 Python 浮点与独立整数参考评估，尚未完成新输入的 HLS C Simulation。

数据来源说明、真实背景负样本测试和新预处理的 HLS 对照仍需补齐，不能将当前结果直接视为 Level 2 正式验收完成。

完整的课程要求对照见[当前验收状态](docs/当前验收状态.md)，模块框图及地址/控制说明见[架构说明](docs/架构与调度说明.md)。课程仅要求仿真与综合，不要求上板。新增预处理结果及其验证边界见[预处理改进说明](level2/ROBUST_PREPROCESSING.md)。

## 当前结果

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| Level 1 MNIST（10,000 样本） | PASS | [Level 1 摘要](level1/results/level1_summary.md) |
| 浮点 / 16 位定点对照 | 98.37% / 98.38%，预测一致率 99.99% | [机器可读报告](level1/results/validation_report.json) |
| 定点位宽扫描 W=8..16 | COMPLETE | [整理版报告与图表](level1/results/numerical_precision/) |
| 推荐配置 | `data_t = ap_fixed<10,6,AP_RND,AP_SAT>` | HLS ≥90%，相对 16 位损失 ≤0.5 pp |
| 混合精度/流水化/AXI组织 | 五个新配置，1000张；W8 I1准确率98.3%，保留时钟未达标反例 | [实验报告和图表](mixed_precision/README.md) |
| Bonus QAT | W8 I6 PTQ 56.93%，QAT软件97.02%，HLS CSim 96.98%，标准MNIST 10,000张 | [QAT补充实验](bonus/README.md) |
| C/RTL协同仿真 | 16位逐行缓存版，两次真实参数调用Verilog PASS | [RTL周期与日志](rtl_validation/README.md) |
| 分层参考/故障定位 | 1000张定点参考与已有HLS一致；混合精度10000张Python为98.40%；20张HLS七层逐位一致、故障定位通过 | [结果与边界](layer_validation/README.md) |
| Level 2 旧预处理基线 | 503张：直接缩放22.27%，旧预处理11.13%；旧预处理HLS CSim完成503张，Python/HLS预测一致率100% | [Level 2 说明](level2/README.md)、[历史执行记录](docs/Level2自采数据执行记录.md) |
| Level 2 预处理改进 v1 | Python浮点：全量429/503（85.29%），内部留出126/145（86.90%）；8张拒识计错；新HLS CSim待完成 | [改进说明](level2/ROBUST_PREPROCESSING.md)、[改进实验报告](level2/results/preprocessing_v1/改进实验报告.md) |
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

## Level 2 旧预处理基线复现

以下命令和 HLS 结果属于旧预处理，保留用于历史追溯和对照，不会自动切换到改进 v1。

数据目录约定为仓库根目录下的 `data/data/raw/0` 到 `data/data/raw/9`。原始图片不随
仓库提交，根目录 `data/` 已由 `.gitignore` 忽略。如果本机图片放在其他位置，请修改
`--input` 为实际路径。

数据来源目前仍待项目组确认，因此结果应称为开发验证数据，不能直接替代正式的组员实拍验收数据。

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

旧预处理实测使用 Vitis HLS 2025.2.1，完整结果为 56/503（11.13%），与旧预处理的
Python 预测逐样本一致。“一致率 100%”表示两种实现预测相同，不表示数字识别准确率为 100%，
也不适用于尚未运行 HLS 的新预处理输入。

结果和限制见：[Level 2 执行记录](docs/Level2自采数据执行记录.md)、[实验报告](level2/results/self_collected/experiment_report.md)、[机器可读摘要](level2/results/self_collected/summary.json)。

## Level 2 预处理改进 v1

### 改进目标与方法

本次改进针对照片中的背景、边框、阴影及数字位置与尺度不一致问题，使输入更接近
LeNet 训练时使用的 MNIST 图像形式。改动仅位于 CPU 端图片预处理与评估工具，
没有修改原始图片、标签、LeNet 模型参数或 HLS 推理源码，也没有进行模型再训练。

新流程为：

1. 读取图片、校正 EXIF 方向并转为灰度，限制处理分辨率。
2. 根据图像内部的背景亮度判断明暗极性，提取相对于背景的笔画对比度。
3. 阈值分割并分析连通区域，过滤明显的宽边框和部分边缘干扰。
4. 选择主要数字区域，合并符合距离和重叠条件的邻近断裂笔画；存在多个大目标等异常时拒识。
5. 裁剪主体、归一化笔画强度，保持长宽比缩放到 20×20 范围内。
6. 放入 28×28 黑色画布，并按笔画重心居中。

预处理器只使用图像像素，不读取真实标签，也不按文件名、数字类别或模型预测结果选择不同规则。
最终采用不加粗版本。

### 数据划分与配置选择

本次使用现有 503 张图片，根据解码后像素内容的 SHA-256 哈希进行确定性图像级划分：

- 开发集：358 张，用于流程消融和最终配置选择。
- 内部留出测试集：145 张，在配置冻结后评估。
- 相同像素内容的图片会分配到同一集合；本次未发现精确像素重复。

开发集对最终“不加粗”和“加粗”两个候选进行比较，两者均为 303/358，
因此按预先约定的平局规则选择更简单的不加粗版本。早期处理阶段用于消融分析。

评估保存模型参数、预处理源码、数据清单与配置的哈希记录，用于检查复现时是否发生变化。
留出结果揭示后，没有继续针对这些测试图片调参。

### 实测结果

下表为同一模型参数下的 Python 浮点识别结果：

| 方法 | 全部 503 张 | 开发集 358 张 | 内部留出测试集 145 张 |
| --- | ---: | ---: | ---: |
| 直接缩放 | 112/503，22.27% | 79/358，22.07% | 33/145，22.76% |
| 旧预处理 | 56/503，11.13% | 37/358，10.34% | 19/145，13.10% |
| 新预处理 v1 | 429/503，85.29% | 303/358，84.64% | 126/145，86.90% |

全部 503 张中有 8 张被新预处理拒识，均按识别错误计入统计，没有从准确率分母中删除。

在 145 张内部留出图片上，相比旧预处理，新版本纠正了 109 张原本错误的图片，
同时使 2 张原本正确的图片变为错误。改善并非每张图片都成立，失败样本仍完整保留。

![内部留出测试集准确率对比](level2/results/preprocessing_v1/heldout_accuracy.png)

![内部留出测试集混淆矩阵](level2/results/preprocessing_v1/heldout_confusion.png)

### 浮点、整数参考与 HLS 的区别

新输入已完成 Python 浮点与独立 W16 整数参考评估：

- 两者全量正确数均为 429/503。
- 两者内部留出测试集正确数均为 126/145。
- 按全量评估记录统计，浮点与整数参考的预测一致数为 502/503，并非完全一致。
- 整数参考是在 Python 中模拟定点运算，不是 HLS C Simulation，也不是 RTL 验证。

新预处理的 HLS C Simulation 尚未完成，因此不能沿用旧预处理的
“Python/HLS 预测一致率 100%”结论。

### 复现方法

从仓库根目录执行。建议使用独立 Python 虚拟环境，并安装新增预处理依赖：

```sh
python -m pip install -r level2/requirements-preprocessing.txt
```

运行测试并建立新的复现实验目录：

```sh
python -m unittest discover -s level2/tools -p "test*.py"
python level2/tools/preprocessing_experiment.py --input "data/data/raw" --output level2/results/replication_v1
python level2/tools/evaluate_robust.py dev --output level2/results/replication_v1
python level2/tools/evaluate_robust.py final --output level2/results/replication_v1
python level2/tools/report_robust.py --output level2/results/replication_v1
```

请将 `--input` 改成实际图片目录。如果图片位于代码根目录的上一层，
路径可能为 `../data/data/raw`。

输出目录应使用尚不存在的新目录，避免覆盖已冻结结果。对相同图片和相同配置的重复运行
仅用于复现，不是新的独立测试。

处理一张不带标签的新照片：

```sh
python level2/tools/preprocess_photo.py "你的照片.png" --output "level2/results/new_photo_01"
```

该命令输出 28×28 PNG、float32 NPY 和诊断 JSON，本身不完成 LeNet 分类。
若状态不是 `ok`，应重新拍摄或人工复核，而不是将拒识样本强行送入模型。

### HLS 交接

已导出用于新输入验证的文件：

- `level2/results/preprocessing_v1/preprocessed_accepted.bin`：495 张通过预处理检测的图片输入。
- `level2/results/preprocessing_v1/hls_input_mapping.csv`：HLS 输入与原始 503 张图片的索引及划分对应关系。
- `level2/results/preprocessing_v1/python_results.csv`：Python 浮点参考结果。
- `level2/results/preprocessing_v1/integer_reference_results.csv`：Python 独立整数参考结果。

在已配置兼容 HLS 环境的 Windows 机器上，可于仓库根目录运行：

```powershell
$env:VIVADO_HLS_ROOT="实际安装目录"
cmd /c level2\run_robust_hls.bat
```

脚本固定读取 `level2/results/preprocessing_v1` 中的交接文件，使用独立工作目录执行
C Simulation，保留旧预处理的历史结果。脚本尚待在可用 HLS 环境中实际验证。

需要区分两个统计口径：

- 通过预处理的样本识别率：分母为 495。
- 全流程端到端准确率：分母为 503，另外 8 张拒识计错。

不得仅报告通过预处理样本的识别率而隐去拒识。比较工具中的 `threshold=0`
仅表示不设置准确率通过门槛，不代表课程验收达标。

### 研究限制与待完成事项

1. **数据来源仍待确认。** 不能仅凭文件名判断图片是否为组员手拍；需补充采集者、采集方式、设备和批次说明。
2. **留出测试是内部图像级划分。** 原项目此前已评估过全部 503 张；本次划分不是外部独立盲测，也没有按书写者或拍摄批次隔离，不能据此证明跨场景泛化能力。
3. **预处理存在适用条件。** 当前假设背景占多数、主要目标为单个数字且明暗反差足够。严重阴影、相连符号、复杂背景及多目标仍可能失败。
4. **数字 9 是明显短板。** 全量正确数为 23/50，内部留出测试为 5/15。不能只展示总体准确率而忽略类别差异。
5. **缺少真实背景负样本测试。** 当前拒识规则及单元测试不能证明真实背景误识率已达标，需另行采集无数字图片进行验证。
6. **新输入的 HLS 对照尚未完成。** 需要检查新的逐样本预测及差异，不能用旧版 HLS 结果替代。
7. **预处理当前运行在 CPU 上。** 本次结果不代表预处理已硬件化，也没有新增其 FPGA 资源、延迟或功耗结论。

下一步应优先完成新输入 HLS C Simulation、补齐采集来源，并建立按书写者或拍摄批次
隔离的全新测试集及真实背景负样本集。后续若继续改进算法，不应继续将已用于分析的
145 张图片当作未见测试集。

详细材料：

- [预处理改进使用说明](level2/ROBUST_PREPROCESSING.md)
- [改进实验报告](level2/results/preprocessing_v1/改进实验报告.md)
- [冻结配置](level2/results/preprocessing_v1/selection.json)
- [最终评估摘要](level2/results/preprocessing_v1/final_summary.json)
- [逐样本预测](level2/results/preprocessing_v1/predictions.csv)
- [全部 19 张内部留出测试失败样本](level2/results/preprocessing_v1/all_test_failures.jpg)

完整样本 HTML 审查页可通过 `report_robust.py` 在本地生成。
该页面依赖配套的 `gallery/` 图片目录，共享时应一起保留。

## 目录

- [`level1/`](level1/)：LeNet HLS 源码、testbench、验证工具和运行说明。
- [`level1/results/numerical_precision/`](level1/results/numerical_precision/)：可直接审阅的位宽扫描结果、CSV 和图表。
- [`level2/`](level2/)：真实场景预处理和压力测试工具。
- [`level2/results/self_collected/`](level2/results/self_collected/)：旧预处理开发验证的清单、CSV、HLS 对照、图表和报告。
- [`level2/ROBUST_PREPROCESSING.md`](level2/ROBUST_PREPROCESSING.md)：新增预处理 v1 的使用、复现、HLS 交接与限制说明。
- [`level2/results/preprocessing_v1/`](level2/results/preprocessing_v1/)：新增预处理的冻结配置、逐样本结果、图表、实验报告和 HLS 输入。
- [`bonus/`](bonus/)：QAT量化感知训练补充实验，比较低比特PTQ与QAT在标准MNIST上的误差。
- [`layer_validation/`](layer_validation/)：独立整数参考、分层分析与故障定位器；20张HLS逐层验证已通过。
- [`mixed_precision/`](mixed_precision/)：权重/激活异构精度、共享核流水与AXI接口组织的对照。
- [`rtl_validation/`](rtl_validation/)：两交易Verilog协同仿真、原始日志与周期可视化。
- [`row_cache/`](row_cache/)：逐行权重缓存，含三架构资源对比、逐样本一致性验证和复现脚本。
- [`resource_reuse/`](resource_reuse/)：16位、同工具原版/共享FC对比，含源码、1000张逐样本结果、综合报告、RTL与图表。
- [`docs/任务要求与LeNet路线核查.md`](docs/任务要求与LeNet路线核查.md)：课程要求与当前完成度核查。
- [`openspec/changes/add-numerical-precision-experiment/`](openspec/changes/add-numerical-precision-experiment/)：位宽实验的 OpenSpec 设计与任务记录。

## 复现提示

Level 1 完整实验需要 10,000 样本 blob、相邻元数据文件和可用的 HLS 环境。
命令和数据准备说明见 [`level1/README.md`](level1/README.md)。

Level 2 完整预处理复现还需要原始图片；仅下载代码及汇总结果不能重新完成图片处理。
新增预处理说明见 [`level2/ROBUST_PREPROCESSING.md`](level2/ROBUST_PREPROCESSING.md)。

大型 HLS 日志、综合工作区和部分中间材料按 `.gitignore` 保留在本地。
历史实验与新增实验分别保存，引用结果时应明确数据集、预处理版本、数值格式和验证阶段。
