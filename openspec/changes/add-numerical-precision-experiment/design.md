## Context

现有权重与样本以 float32 存入 blob，Testbench 赋值到 `data_t` 时量化。`data_t` 同时用于输入、权重、偏置、层输出和 logits；MAC 显式转换到 `acc_t`。因此本实验研究共享数据类型的位宽，不能把它描述成整个乘法器也使用相同位宽。

当前 `tb_lenet.cpp` 仅在 `argc == 3` 时打开 CSV，而自定义阈值需要 `argc == 4`，实现时须修正为兼容两种情况。现有 Tcl 缺少 blob 时会运行 smoke test，新实验必须先验证输入，避免将 smoke test 当作完整实验。

本设计遵循仓库 `spec-driven` 格式。`specs/numerical-precision-experiment/spec.md` 定义验收行为，本文确定实现边界，`tasks.md` 给出执行顺序。当前交付仅为设计，任务中的实验尚未执行。

## Goals / Non-Goals

**Goals:**

- 回答“在当前共享定点格式下，多少位能保持准确率，综合资源和延迟如何变化”。
- 用一份输入、一条串行扫描流程和一套汇总结果完成实验。
- 下游 agent 能明确区分软件近似、HLS 实测和缺失证据。

**Non-Goals:**

- 不重训练、不修改网络结构、循环/并行度、AXI 配置或默认部署位宽。
- 不做累加器扫描、4/6 位、分层/混合精度、QAT、二值化或上板测量。
- 不引入自动缩放、校准集搜索或逐 MAC 的 Python 位精确仿真。本实验保留当前数值尺度；动态范围归一化要求仍需后续独立实验。
- 不建立通用 DSE 框架、并行调度、缓存或自动断点恢复。

## Decisions

### 1. 固定实验矩阵和判断标准

| 项目 | 约定 |
| --- | --- |
| 数据 | 复用现有流程生成并验证的官方 MNIST test split 全部 10,000 样本 |
| 参数与归一化 | 同一 blob 中的 float32 参数、原样本顺序、`uint8 / 255.0` |
| 数据类型 | `ap_fixed<W,6,AP_RND,AP_SAT>`，W=8,9,10,11,12,13,14,15,16 |
| 累加器 | `ap_fixed<32,14,AP_RND,AP_SAT>`，所有档位固定 |
| 综合条件 | `xc7z020clg400-1`，目标周期 10 ns，相同 HLS 版本和源码 |
| 执行顺序 | 先 16 位基线，再 8～15 位，每档 CSim 后综合 |
| 可用标准 | HLS 准确率 ≥90%，且相对本次 16 位基线下降 ≤0.5 个百分点 |

6 个整数位包含符号位；小数位 `F=W-6`，步长 `2^-F`，范围 `[-32, 32-2^-F]`。这是固定动态范围、改变分辨率的实验，不代表优化了所有可能的定点格式。0.5 个百分点是本实验的工程选型规则，并非任务书另行规定的门槛；在运行前固定，不能根据结果事后放宽。

只有九档的 CSim、综合、报告解析全部完成，且 16 位基线达到 90%，才生成正式推荐。取满足标准的最小 W；无候选则输出“无可用配置”。资源变化不作为准确率门槛，报告须分别说明 LUT/FF/DSP/BRAM 是否节省。不要预设低位宽一定省 DSP 或加速，也不要将最低可用位宽自动写回默认类型。

### 2. Python 分析采用明确标记的层边界量化近似

复用现有网络布局和数据读取，先运行未改动的 Python 浮点参考，再对九档运行如下近似模型：

```text
F = W - 6
Q(x) = clip(floor(float64(x) * 2^F + 0.5), -2^(W-1), 2^(W-1)-1) / 2^F
```

- 输入、所有权重和偏置在进入网络时各量化一次；卷积/全连接使用 float64 乘加。
- 每个卷积/全连接结果先 Q，再 ReLU；末层只 Q。最大池化直接选取已有量化值，argmax 平局时取最小索引。
- Q 对齐 `AP_RND` 最近舍入、恰好半格时向正无穷，以及 `AP_SAT` 饱和边界；不能使用默认银行家舍入的 `np.round` 替代。
- 不模拟 `acc_t` 每次赋值和累加的舍入/饱和，输出名称固定为 `python_approx`，不要求与 HLS 逐位或预测完全一致，也不能用它替代任一档 HLS 数据或提前跳过低分档位。
- 在同一浮点参考遍历中记录输入、各参数数组和各卷积/全连接量化前输出的 min/max/max_abs，写入 `ranges.json`。它只解释当前尺度，不用于依据测试集重新选整数位或缩放。

这保留了 Python 位宽敏感度实验，同时把硬件数值语义交给 HLS 验证。无须新增层级特征图导出或运行时饱和计数系统。

### 3. 只增加必要的编译与运行入口

- `lenet.hpp` 增加 `LENET_DATA_W` 编译宏，默认 16，限定为 8～16；整数位、舍入模式、饱和模式及 `acc_t` 保持常量。
- `run_hls.tcl` 从同名环境变量读取位宽，给设计源码和 Testbench 一致传入编译宏；增加 `LENET_HLS_WORKSPACE` 可选环境变量，未设置时仍使用原 `level1/hls_work`。
- 每档使用独立 `<output>/w<W>/hls_work` 和 `hls.csv`；启动时打印实际位宽、完整类型、器件和周期，供日志核查。
- 扫描设置已有的 `LENET_ACCURACY_BLOB`、`LENET_RESULT_CSV`、`LENET_ACCURACY_THRESHOLD=0`，确保低分档位正常结束并继续综合。原独立 Level 1 命令仍默认使用 90%。
- 修正 CSV 的 `argc` 判断；Tcl 构造 CSim 参数时保留带空格路径的边界。无需修改 blob 协议或 CSV 列名。

新增脚本提供以下入口（这是待实现接口）：

```text
python3 level1/tools/precision_experiment.py run --blob <完整blob> --hls-executable <HLS可执行文件> --output <新目录>
python3 level1/tools/precision_experiment.py run --blob <完整blob> --python-only --output <新目录>
python3 level1/tools/precision_experiment.py report --output <已有目录>
```

`run` 始终要求完整数据。输入元数据默认读取 `blob.with_suffix(".metadata.json")`（与现有 `make-blob` 一致），允许用 `--blob-metadata <路径>` 指定；校验 `sample_count=10000`、参数数量、协议、归一化及重新计算的 blob SHA-256 与 `output_sha256` 一致，保留其中的 MNIST images/labels 摘要。README 要求用既有官方 IDX 准备流程生成这对输入；只有样本数而没有来源摘要不能作为正式输入。

HLS 所需环境由用户事先配置，脚本用参数列表调用可执行文件和仓库 Tcl，不执行任意 shell 字符串。扫描须显式清除或设为 0 的 `LENET_SKIP_CSIM`、`LENET_SKIP_SYNTH`，避免继承外部环境而跳过阶段。输出目录已非空则拒绝覆盖；重新实验使用新目录。`report` 仅重算汇总与绘图，不运行 HLS。不再增加通用配置文件或多套平台专用调度器。

`--python-only` 用于无 HLS 环境的开发检查，明确记录实验不完整、硬件字段缺失、无正式推荐；该显式模式可正常返回。完整模式缺少 HLS 时明确失败，不能自动降级。单样本与少量数据仅用于测试函数及 Testbench 回归，不进入正式 `run`。

### 4. 固定输出字段与失败语义

`<output>/manifest.json` 记录源码 Git revision 和工作区是否有修改、受影响源码/Tcl 的 SHA-256、blob 和参数区 SHA-256、MNIST 来源摘要、样本数、归一化、扫描矩阵、完整类型、工具版本、器件、时钟、实际命令及选型阈值。参数区哈希覆盖 blob 的 8 字节头之后、首个样本之前的原始参数字节。浮点逐样本结果保存在根目录 `float.csv`，各档保存 `python_approx.csv`、`hls.csv`、日志、`metrics.json` 及本档顶层 csynth XML/文本报告；报告只读取 manifest 指向的本次目录。

汇总 `precision_sweep.csv` 每个 W 一行，至少包含：

| 字段 | 口径 |
| --- | --- |
| `width`, `fraction_bits`, `sample_count` | W、W-6、10000 |
| `float_accuracy_pct`, `python_approx_accuracy_pct`, `hls_accuracy_pct` | 准确率统一为 0～100 的百分数 |
| `loss_vs_16_pp`, `loss_vs_float_pp` | 对应参考准确率减本档 HLS 准确率，允许负值 |
| `agreement_pct` | HLS 与原浮点参考预测相同的样本百分比 |
| `logit_mae`, `logit_max_abs_error` | HLS 相对原浮点 logits 的全样本、全 10 类误差 |
| `lut`, `ff`, `dsp`, `bram_18k` | 顶层 csynth 报告资源计数 |
| `latency_cycles_max`, `latency_ns_target` | 顶层最坏周期数；该周期数 × 目标周期 10 ns |
| `estimated_clock_ns` | 综合报告估计周期；不写成实测频率或布局布线时序 |
| `csim_status`, `synth_status`, `parse_status` | 每阶段 `pending / ok / failed / skipped` |
| `eligible`, `error` | 是否达到选型条件、失败或缺失原因 |

机器数据以未四舍五入的计数计算阈值，展示时再格式化。逐样本 CSV 必须是 0～9999 的唯一连续索引、标签与 blob 一致、10 个有限 logits、合法预测；缺行、重复、NaN 或错位均不能计算为有效准确率。

运行失败不等于准确率不合格：低准确率是有效观测，继续综合；编译/仿真失败则保留错误并跳过本档综合，继续其他档；综合失败仍保留本档准确率。缺失、不可解析或非数值的报告字段在 JSON 中为 null、CSV 中为空，禁止填 0 或借用其他档位报告。各阶段结束即写入本档状态，全部尝试后汇总；完整模式存在任一阶段失败时返回非零且不推荐。

### 5. 最小报告与复现材料

- `summary.json`：实验是否完整、各档数据、推荐 W 或 null、理由、所用门槛。
- `report.md`：实验条件、格式解释、九档对照表、位宽选择、各资源相对 16 位变化和低位宽误差解释。
- `accuracy_vs_width.png`：HLS/Python 近似准确率曲线及浮点参考线；`resources_vs_width.png`：四种资源分别以子图展示，避免混合单位。缺失值留空。
- 资源节省百分比使用 `(R16-RW)/R16*100`；基线为 0 时记为不可计算。曲线无明显平台或收益时如实说明，不强制宣称存在拐点。
- 大型输入、逐样本 CSV、HLS 工作目录忽略；提交脚本、说明、精简汇总、图表和必要的顶层综合报告。保留本地逐样本结果以便重新汇总。
- 同一次比较必须使用同一工具版本。Vitis 开发结果与 Vivado HLS 2019.2 复跑使用不同输出目录；2019.2 不可用时明确该项待完成，不能混合两套资源数据或标记正式工具链验收通过。

## Risks / Trade-offs

- 九档完整 CSim 耗时较长：先完成小样本回归，再串行执行正式矩阵；Python 结果不代替完整 CSim。
- 固定 6 个整数位会使低位宽权重分辨率较差：这是当前格式的实验结论，不能推广成“8 位量化普遍不可用”。
- MAC 转换到固定 `acc_t`，AXI 和综合优化也影响资源：资源不一定随 W 单调下降，保留真实报告并解释。
- Python 近似忽略累加器误差：明确标记近似，选型只用 HLS 数据；异常差异先核对量化边界及编译宏。
- 工具环境或完整数据缺失：可以完成实现与局部测试，但保留对应执行任务为未完成，不生成虚构实验数字。
