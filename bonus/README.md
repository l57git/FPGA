# Bonus: QAT量化误差实验

本目录用于补充当前PTQ路线的低比特误差对照。脚本从仓库已有
`level1/data/lenet_accuracy_1.bin` 读取LeNet浮点参数，在MNIST train split上进行
fake-quant QAT微调，并在标准MNIST test split的10000张图片上比较：

- 浮点基线准确率；
- 同一位宽的PTQ准确率和logit误差；
- QAT后的准确率和logit误差；
- 权重量化值相对训练潜变量的MSE。

## 远程复现

课程虚拟机已有可用环境：

```sh
cd /home/ubuntu/workspace/FPGA
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

输出文件：

- `summary.json`：机器可读配置、训练历史和汇总指标；
- `metrics.csv`：float/PTQ/QAT三组准确率与logit误差；
- `weight_quantization_error.csv`：逐层权重量化MSE；
- `experiment_report.md`：可直接放入报告的中文摘要；
- `qat_state.pt` 和 `qat_quantized_state.pt`：复现实验用模型权重，默认不提交仓库。

## 已完成结果

远程服务器 `/home/ubuntu/workspace/FPGA` 已完成一次完整实验，汇总结果保存在
`bonus/results/w8a8_i6_qat/`：

|模型|MNIST test准确率|相对浮点损失|预测与浮点一致率|logit RMSE|
|---|---:|---:|---:|---:|
|float|98.37%|0.00 pp|||
|PTQ|56.93%|41.44 pp|57.17%|5.6483|
|QAT|97.02%|1.35 pp|97.78%|3.2756|

这说明在同一 `ap_fixed<8,6>` fake-quant网格下，直接PTQ会产生明显低比特误差；
两轮QAT后，分类准确率从56.93%恢复到97.02%，logit RMSE也明显下降。

作为参考，仓库已有 `mixed_precision/results/metrics.csv` 中的 `w8i6_serial`
PTQ控制实验为1000张MNIST、权重8位整数6位、激活/bias保持16位，准确率59.7%。
该结果和本 bonus 的 `ap_fixed<8,6>` 全路径fake-quant不是完全相同控制变量，但两者都
说明粗粒度低比特 PTQ 会造成明显精度下降。

## HLS C Simulation

QAT权重已继续完成 HLS CSim。流程使用独立的 `bonus/run_qat_hls.sh`：

```sh
cd /home/ubuntu/workspace/FPGA
bonus/run_qat_hls.sh
```

该脚本会把 `qat_quantized_state.pt` 和标准 MNIST test split 导出为 HLS blob，然后设置：

- `LENET_DATA_W=8`
- `LENET_SKIP_SYNTH=1`
- `LENET_HLS_WORKSPACE=bonus/hls_work/w8a8_i6_qat_csim`
- `LENET_RESULT_CSV=bonus/results/w8a8_i6_qat_hls/hls_predictions.csv`

远程服务器已完成 10000 张 MNIST test 的 HLS CSim：

|项目|准确率|正确数|状态|
|---|---:|---:|---|
|QAT软件fake-quant|97.02%|9702/10000|已完成|
|QAT HLS CSim|96.98%|9698/10000|`BATCH COMPLETE`，CSim 0 errors|

HLS CSim 与软件 QAT 只差 0.04 pp，且超过 90% 门槛。结果保存在
`bonus/results/w8a8_i6_qat_hls/`，其中 `hls_experiment_report.md` 是报告摘要，
`hls_predictions.csv` 是逐样本输出，`run.log` 是原始 HLS 日志。

默认配置使用 `ap_fixed<8,6>` 风格的权重、激活和输出fake quant，用来正面比较
低位宽PTQ误差。若只想贴近已有 `mixed_precision` 的低比特权重实验，可加
`--no-activation-quant --weight-integer-bits 1`，即只训练权重低比特适配。

## 方法边界

QAT脚本只提供软件训练和标准MNIST测试证据，不改变现有HLS源码，也不产生新的综合资源、
时序或RTL仿真结论。若后续要把QAT权重并入硬件路径，需要再将
`qat_quantized_state.pt` 导出为当前testbench使用的blob或HLS常量数组。

HLS CSim 仍只覆盖功能仿真；本 bonus 未运行综合、RTL协同仿真或资源时序评估。
