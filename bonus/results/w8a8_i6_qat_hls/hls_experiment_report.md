# QAT HLS C Simulation

## 方法

使用 bonus 导出的 QAT MNIST blob 运行 `level1` 的 LeNet HLS C simulation。工作目录、输入blob和输出CSV均位于 `bonus/`，未覆盖已有 Level 1 或混合精度结果。

## 配置

- `LENET_DATA_W`：8，即 `ap_fixed<8,6,AP_RND,AP_SAT>`
- 输入blob：`/home/ubuntu/workspace/FPGA/bonus/results/w8a8_i6_qat_hls/qat_hls_input.bin`
- HLS工作目录：`/home/ubuntu/workspace/FPGA/bonus/hls_work/w8a8_i6_qat_csim`
- 样本数：10000
- HLS返回码：0

## 结果

|项目|准确率|正确数|预测一致率|备注|
|---|---:|---:|---:|---|
|QAT软件fake-quant|97.02%|9702/10000|97.78%|来自 `bonus/results/w8a8_i6_qat/metrics.csv`|
|QAT HLS CSim|96.98%|9698/10000||真实 HLS C simulation 输出|

QAT软件fake-quant与HLS CSim准确率差值：0.04 pp。

## 状态

- CSim完成：True
- `BATCH COMPLETE`：True
- 达到准确率阈值 90.00%：True

## 边界

本文件只报告 QAT 权重在现有 LeNet HLS C 模型上的功能仿真；未运行综合、RTL协同仿真或资源时序评估。
