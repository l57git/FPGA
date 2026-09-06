# C/RTL协同仿真实验报告

## 目的与配置

验证逐行缓存16位版本生成的Verilog，而不只检查C++函数。采用Vivado HLS 2019.2和xsim、XC7Z020、10ns，输入为原MNIST测试集前两张及真实训练参数，标签7和2。保留连续两次顶层调用，覆盖静态缓存的复用。代码、头文件和testbench哈希与此前1000张C仿真实验一致。

## 结果

HLS报告Verilog Pass，C post checking通过，两张预测分别为7和2。交易0延迟2822332周期，交易1延迟2822327周期；对应28.22332ms和28.22327ms。唯一可测调用间隔为2822333周期。它们位于此前综合估计的1657356～3100026周期范围内。

这是RTL仿真中实际记录的周期，时间按10ns仿真时钟换算，不是上板测量。两次调用不足以统计稳定吞吐分布，也不能把2/2写成MNIST完整准确率100%。准确率证据仍引用1000张C仿真98.2%及同学完整Level 1结果。

## 原始证据

row_cache/lenet_accel_cosim.rpt为HLS协同仿真汇总；result.transaction.rpt为逐交易周期；lenet_accel.log为RTL模拟器日志；run.log包括C参考测试、RTL模拟和C后检查。predictions.csv与cosim_predictions.csv是testbench记录，RTL通过状态依赖HLS的协同仿真比较器，不能仅靠两个CSV相同判断RTL通过。

transactions.csv与rtl_latency.png由tools/report.py直接解析原始报告生成。没有启用波形跟踪，因此没有声称留存波形截图。任务书只要求仿真与综合，不要求开发板照片。

## 覆盖边界

本次仅验证row_cache的16位串行架构，不能自动覆盖mixed_precision中的其他接口、位宽和流水变体。尚未进行全测试集RTL仿真、随机压力验证、逐层特征图定位和布局布线。
