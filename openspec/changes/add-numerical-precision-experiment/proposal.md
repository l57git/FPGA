## Why

当前 LeNet 只有 `data_t=<16,6>`、`acc_t=<32,14>` 一档定点配置。已有 MNIST 浮点/定点对照不能说明最小可用位宽，也不能给出降低位宽后的资源收益。需要在现有验证流程上补齐单变量实验，为定点格式选择提供可复现依据。

## What Changes

- 在相同参数、官方 MNIST 10,000 样本、器件和时钟下，逐位扫描 8～16 位；整数位固定为 6，累加器固定为 `<32,14>`。
- 增加轻量 Python 层边界量化敏感度分析；以实际 HLS C simulation 作为定点准确率依据。
- 每档运行综合，汇总准确率、logits 误差、资源和延迟，生成两张曲线及最小可用位宽结论。
- 增加一个实验脚本及编译配置入口，复用现有 blob、浮点模型、CSV 和 HLS Tcl；修正 Testbench 同时指定 CSV 和阈值时不导出结果的问题。
- 明确失败、缺失报告和未完成实验的状态，保留低准确率档位的数据。

## Capabilities

### New Capabilities

- `numerical-precision-experiment`: 在统一输入和工具链下执行 LeNet 定点位宽扫描，形成可追溯的精度与硬件开销对照及选型依据。

### Modified Capabilities

无。原 `mnist-validation` 的默认 16 位配置和 90% 验收行为保持兼容；新实验独立判断各档位是否适用。

## Impact

- 小幅修改 `level1/src/lenet.hpp`、`level1/run_hls.tcl`、`level1/tb/tb_lenet.cpp`。
- 新增 `level1/tools/precision_experiment.py` 和针对性测试；复用 `level1/tools/lenet_validation.py`，如需量化钩子，默认浮点路径必须兼容。
- 更新 `level1/README.md` 和必要的忽略规则；结果放在独立实验目录。
- 使用现有 NumPy，绘图使用 Matplotlib；不引入训练框架、工作流框架或数据库。
- 本次补齐基础位宽选择实验及任务 3 的扫描/曲线证据，不宣称完成动态范围重缩放、混合精度或全部进阶评分项。
