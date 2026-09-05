## Purpose

本能力在现有 LeNet 与 MNIST 验证流程上开展受控的 8～16 位定点精度实验，通过 Python 敏感度分析、HLS 仿真和综合记录准确率与硬件开销，提供可复现的最小可用位宽选择依据。

## ADDED Requirements

### Requirement: 统一输入和单变量扫描
实验 SHALL 对官方 MNIST test split 全部 10,000 样本扫描 W=8～16 的每一整数位宽，使用同一组参数、样本顺序与归一化；数据类型 SHALL 为 `ap_fixed<W,6,AP_RND,AP_SAT>`，累加器 SHALL 固定为 `ap_fixed<32,14,AP_RND,AP_SAT>`，器件 SHALL 为 `xc7z020clg400-1`，目标周期 SHALL 为 10 ns，同一比较中的工具版本及其他设计条件 SHALL 一致。

#### Scenario: 完整运行矩阵
- **WHEN** 提供经验证的完整 blob 和可用 HLS 环境启动完整实验
- **THEN** 工具先运行 16 位基线，再运行 8～15 位，并为九档分别记录 CSim 和综合结果

#### Scenario: 拒绝不完整输入
- **WHEN** blob 缺失、损坏、不是完整官方测试集或与输入摘要不一致
- **THEN** 工具在启动 HLS 前以非零状态停止，不回退为 smoke test

### Requirement: 明确 Python 量化分析的数值边界
工具 SHALL 运行原 Python 浮点参考和各档 `python_approx` 层边界量化模型；近似模型 SHALL 按 design.md 的 Q 公式量化输入、参数及卷积/全连接输出，并在 ReLU 前量化；工具 SHALL 导出浮点输入、参数和各层量化前输出的范围统计。近似模型 SHALL 不被描述为 HLS 位精确仿真或用于替代 HLS 准确率。

#### Scenario: 半格舍入和饱和
- **WHEN** W=8，Q 接收 0.125、-0.125、32、-33
- **THEN** 分别得到 0.25、0、31.75、-32，且正负半格舍入及上下界行为具有自动测试

#### Scenario: 软件与硬件存在差异
- **WHEN** Python 近似准确率或 logits 与 HLS 不一致
- **THEN** 分别保存双方结果，以 HLS 结果进行位宽选择，不强制修改数据使两者一致

### Requirement: 编译配置隔离且兼容现有流程
设计源码与 Testbench SHALL 使用一致的 `LENET_DATA_W` 宏，每档 SHALL 使用独立工作目录和结果路径；未指定宏与工作目录时 SHALL 保持原 16 位配置及 Level 1 行为；`run` SHALL 拒绝覆盖非空输出目录。

#### Scenario: 阈值与 CSV 同时传入
- **WHEN** Testbench 同时接收有效 blob、CSV 路径和阈值 0，且实际准确率低于 90%
- **THEN** 导出全部逐样本记录，正常完成仿真并允许后续综合，汇总仍将该档标记为准确率不达标

#### Scenario: 默认调用回归
- **WHEN** 运行原 smoke test、单 blob 或 blob 加 CSV 命令且不指定新配置
- **THEN** 维持原参数接口、16 位类型和默认 90% 准确率门槛

#### Scenario: 带空格路径
- **WHEN** 输入、输出或 HLS 可执行文件位于带空格的有效路径
- **THEN** 路径作为完整参数传递，设计和 Testbench 均使用指定档位编译

### Requirement: 真实测量和完整性校验
每档 SHALL 记录 design.md 定义的准确率、相对差值、预测一致率、logits 误差、四项资源、顶层最坏周期数、目标周期延迟和估计周期；HLS 准确率 SHALL 来自完整逐样本 CSV，硬件数据 SHALL 来自本档顶层综合报告。

#### Scenario: 正常汇总
- **WHEN** 某档 CSV 和综合报告完整有效
- **THEN** 准确率按正确数除以 10,000 计算，延迟按顶层最坏周期数乘 10 ns 计算，并保留原始证据路径

#### Scenario: 拒绝无效预测结果
- **WHEN** CSV 缺行、重复索引、标签错位、预测越界或包含非有限 logits
- **THEN** 标记本档解析失败并保留原因，不将其作为有效精度结果

#### Scenario: 硬件指标缺失
- **WHEN** 综合报告缺失、无法解析或关键指标不是数值
- **THEN** 缺失指标写为 null 或 CSV 空值，本档不满足完整性要求，禁止用 0 或其他档位数据补齐

### Requirement: 实验失败与低准确率分别处理
工具 SHALL 保留各阶段状态和日志，并在个别档位失败后继续尝试其他档位；低准确率 SHALL 作为有效观测保留且继续综合；完整模式发生执行或解析失败 SHALL 返回非零并禁止正式推荐。

#### Scenario: 某档综合失败
- **WHEN** 某档已完成有效 CSim 但综合失败
- **THEN** 保留准确率、记录综合失败及错误信息、继续其他档位，并输出不完整报告

#### Scenario: 仅运行 Python
- **WHEN** 用户显式指定 `--python-only`
- **THEN** 输出浮点及九档近似结果，HLS 阶段为 skipped，实验标记不完整且无正式推荐

### Requirement: 按预先固定的规则选择位宽
工具 SHALL 仅在九档证据完整且本次 16 位 HLS 基线达到 90% 时，从 HLS 准确率 ≥90% 且相对该基线下降 ≤0.5 个百分点的配置中推荐最小 W；判断 SHALL 使用未四舍五入的数据。报告 SHALL 将此结论限定为当前共享格式和扫描范围内的选择。

#### Scenario: 门槛边界
- **WHEN** 基线准确率为 98.38%，某档分别取得 97.88% 或 97.87%
- **THEN** 前者满足下降门槛，后者不满足，最终选择同时满足全部条件的最小 W

#### Scenario: 基线或实验不完整
- **WHEN** 16 位基线低于 90% 或任一必需档位的仿真、综合、解析未完成
- **THEN** 推荐值为 null，报告给出原因且不宣称完成位宽选型

### Requirement: 结果可复现且结论不过度外推
工具 SHALL 生成 manifest、范围统计、九档 CSV、JSON 摘要、Markdown 报告、准确率曲线及资源曲线；报告 SHALL 记录输入和源码摘要、完整类型、工具版本、命令、阶段状态、选型门槛及各资源相对基线的变化。报告 SHALL 区分 Python 近似、HLS CSim、综合估计及指定工具链验收状态。

#### Scenario: 离线重新汇总
- **WHEN** 原始逐样本结果、manifest 和综合报告完整存在，运行 `report`
- **THEN** 不调用 HLS 即可重新生成相同口径的汇总、图表和推荐；不存在的硬件数据保持缺失

#### Scenario: 资源未下降或指定工具不可用
- **WHEN** 某档资源没有下降，或实验仅在 Vitis 开发环境完成
- **THEN** 如实报告资源变化及工具版本，不能虚构资源收益或声称完成 Vivado HLS 2019.2 验收
