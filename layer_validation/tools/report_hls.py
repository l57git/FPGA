"""Render only verified HLS intermediate-trace evidence."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'layer_validation/results'
s=json.loads((out/'hls_layer_verification.json').read_text())
assert s['status']=='PASS'
with (out/'hls_layer_metrics.csv').open() as f: rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(9,4))
bars=ax.bar([r['layer'] for r in rows],[int(r['fault_mismatches']) for r in rows],color='#cd7754')
ax.bar_label(bars,padding=3)
ax.set(ylabel='Mismatched elements across 20 images',title='HLS C simulation: injected Pool1 error propagation',ylim=(0,1100))
fig.tight_layout();fig.savefig(out/'hls_fault_propagation.png',dpi=180);plt.close(fig)
lines=['# HLS逐层对照与故障注入验证','','E盘恢复后，在Vivado HLS 2019.2中实际完成正常版和故障版C仿真，两组均有CSim done with 0 errors。原先hls_unavailable目录作为历史失败记录保留。','','## 正常版','','每个数字类别选择2张，共20张。Conv1、Pool1、Conv2、Pool2、FC1、FC2和logits七个观测层共116280个原始定点值，与独立Python整数参考逐位相同。Conv/FC隐藏层观测点在ReLU之后。输入、插桩源码、头文件、testbench均有哈希核对。','','## 故障版','','在C仿真中将每张图的pool1[0][0][0]增加1.0，再继续推理。20/20均在Pool1索引0发现首次分歧；Conv1没有分歧。故障传播后的所有层也与Python注入同样故障的参考逐位一致。','','|层|正常版检查元素|正常版不一致|故障版相对正常参考不一致|','|---|---:|---:|---:|']
for r in rows: lines.append('|'+ '|'.join(r[k] for k in ['layer','elements','clean_mismatches','fault_mismatches'])+'|')
lines+=['','## 证据与复现','','clean和fault目录分别保存真实HLS日志、预测CSV和压缩中间轨迹。hls_layer_verification.json记录验证状态与每张的首次故障位置，hls_layer_metrics.csv记录分层数量。运行tools/run.py后依次运行tools/check_hls.py、tools/report_hls.py。已有case目录须先备份，禁止覆盖原实验。','','## 边界','','这是HLS C仿真分层验证，不是新RTL仿真、综合或上板结果。20张用于功能及故障定位覆盖，不是完整MNIST准确率评估。故障通过仿真插桩主动注入，不能声称发现原设计存在该故障。此前独立rtl_validation模块的RTL结果仍单独保留。混合精度候选剩余9000张的HLS验证、UNROLL并行度边界、自采数据及最终团队材料仍待完成。']
(out/'hls_experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
