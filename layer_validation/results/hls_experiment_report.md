# HLS逐层对照与故障注入验证

E盘恢复后，在Vivado HLS 2019.2中实际完成正常版和故障版C仿真，两组均有CSim done with 0 errors。原先hls_unavailable目录作为历史失败记录保留。

## 正常版

每个数字类别选择2张，共20张。Conv1、Pool1、Conv2、Pool2、FC1、FC2和logits七个观测层共116280个原始定点值，与独立Python整数参考逐位相同。Conv/FC隐藏层观测点在ReLU之后。输入、插桩源码、头文件、testbench均有哈希核对。

## 故障版

在C仿真中将每张图的pool1[0][0][0]增加1.0，再继续推理。20/20均在Pool1索引0发现首次分歧；Conv1没有分歧。故障传播后的所有层也与Python注入同样故障的参考逐位一致。

|层|正常版检查元素|正常版不一致|故障版相对正常参考不一致|
|---|---:|---:|---:|
|conv1|69120|0|0|
|pool1|17280|0|20|
|conv2|20480|0|123|
|pool2|5120|0|38|
|fc1|2400|0|922|
|fc2|1680|0|655|
|logits|200|0|142|

## 证据与复现

clean和fault目录分别保存真实HLS日志、预测CSV和压缩中间轨迹。hls_layer_verification.json记录验证状态与每张的首次故障位置，hls_layer_metrics.csv记录分层数量。运行tools/run.py后依次运行tools/check_hls.py、tools/report_hls.py。已有case目录须先备份，禁止覆盖原实验。

## 边界

这是HLS C仿真分层验证，不是新RTL仿真、综合或上板结果。20张用于功能及故障定位覆盖，不是完整MNIST准确率评估。故障通过仿真插桩主动注入，不能声称发现原设计存在该故障。此前独立rtl_validation模块的RTL结果仍单独保留。混合精度候选剩余9000张的HLS验证、UNROLL并行度边界、自采数据及最终团队材料仍待完成。
