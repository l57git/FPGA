"""Create figures and a report using only completed measured results."""
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'resource_reuse/results'
rows = json.loads((out/'summary.json').read_text())
comparison = json.loads((out/'comparison.json').read_text())
if len(rows)!=2: raise ValueError('both experiments are required')
base, shared = rows
core=ET.parse(out/'shared/dense_shared_csynth.xml')
core_resources={k:core.findtext('.//Resources/'+v) for k,v in [('DSP','DSP48E'),('FF','FF'),('LUT','LUT')]}
rtl = (out/'shared/rtl/lenet_accel.v').read_text()
instances = re.findall(r'^dense_shared\s+(\w+)\s*\(', rtl, flags=re.M)
if len(instances)!=1: raise ValueError('expected exactly one shared core instance')
(out/'reuse_evidence.json').write_text(json.dumps({'module':'dense_shared','instance_count':len(instances),'instances':instances,'top_rtl':'shared/rtl/lenet_accel.v','core_report':'shared/dense_shared_csynth.rpt'},indent=2))
fig, axes = plt.subplots(1,4,figsize=(13,3.8))
for ax,key in zip(axes,['DSP','LUT','FF','BRAM']):
    values=[int(r[key]) for r in rows]
    bars=ax.bar(['Baseline','Shared FC'],values,color=['#527da3','#da8b39'])
    ax.bar_label(bars,padding=3); ax.set_title(key); ax.set_ylim(0,max(values)*1.25+1)
fig.suptitle('Same tool, 16-bit arithmetic, XC7Z020, 10 ns target')
fig.tight_layout(); fig.savefig(out/'resources.png',dpi=180); plt.close(fig)
fig, ax = plt.subplots(figsize=(8,4))
for i, r in enumerate(rows):
    low, high = int(r['cycles_min'])*1e-5, int(r['cycles_max'])*1e-5
    ax.plot([low,high],[i,i],linewidth=7,marker='o',color=['#527da3','#da8b39'][i])
    ax.text((low+high)/2,i+.13,f'{low:.2f}–{high:.2f} ms',ha='center')
ax.set(yticks=[0,1],yticklabels=['Baseline','Shared FC'],xlabel='HLS estimated latency range at 10 ns (ms)',ylim=(-.5,1.6),xlim=(0,35))
ax.grid(axis='x',alpha=.25); fig.tight_layout(); fig.savefig(out/'latency.png',dpi=180); plt.close(fig)
fields=['variant','samples','accuracy_percent','DSP','LUT','FF','BRAM','cycles_min','cycles_max','estimated_clock_ns']
with (out/'metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
lines=['# 跨层全连接计算资源复用实验报告','','## 实验设置','',
       f"同一MNIST测试集前{base['samples']}张，同一参数，{base['width']}位数据、6位整数，32位累加器、14位整数。两版均使用Vivado HLS 2019.2、XC7Z020和10ns目标周期，采用优化C仿真。原版来源为仓库b5edae1。", '',
       '这是体系结构控制变量实验，未重做同学的8～16位精度扫描。1000张用于验证改造的一致性，不能把子集准确率代替10000张结果。', '',
       '## 设计变更','',
       '原版三层FC拥有各自循环；复用版设置统一输入缓冲256、权重缓冲30720、bias缓冲120、输出缓冲120。阶段计数器stage依次选择FC1、FC2、FC3的参数、输入长度和输出长度，装载当前层参数后调用同一个dense_shared函数。函数禁止内联，源码只有一个调用点。', '',
       '这项改造同时引入片上权重缓存与装载阶段，因此资源/时延差值是整套复用架构的结果，不能全部归因于单独的乘法器共享。', '',
       '## 实测结果','', '|指标|原版|共享FC|', '|---|---:|---:|']
for k in fields[1:]: lines.append(f'|{k}|{base[k]}|{shared[k]}|')
lines += ['',f"两版预测一致 {comparison['prediction_matches']}/{comparison['samples']}；全部10个logits逐样本完全相同：{comparison['all_logits_equal']}。", '',
          f"RTL顶层仅发现一个dense_shared实例：`{instances[0]}`。共享函数资源为{core_resources}。", '',
          '## 工程结论', '',
          '资源变化（正数为增加，负数为减少）：'+', '.join(f"{k}: {(int(shared[k])/int(base[k])-1)*100:+.2f}%" for k in ['DSP','LUT','FF','BRAM'])+'。选择配置时同时核查存储和计算资源预算。', '',
          f"按目标周期换算，原版延迟{int(base['cycles_max'])*1e-5:.3f}ms，共享版综合范围{int(shared['cycles_min'])*1e-5:.3f}～{int(shared['cycles_max'])*1e-5:.3f}ms。动态循环上界导致范围变宽，尚不足以证明加速，需RTL协同仿真测量实际周期。", '',
          'DSP变化为 '+str(int(shared['DSP'])-int(base['DSP']))+'，BRAM变化为 '+str(int(shared['BRAM'])-int(base['BRAM']))+'。复用必须同时考虑缓存容量、控制与访存代价；是否适合部署不能只看DSP。', '',
          'HLS延迟和时钟为综合估算，不是布局布线或上板测量。共享函数带有动态循环边界，若报告给出范围或未知值，不能将最大tripcount估算当作实测总延迟。', '',
          '## 证据与复现','',
          '- 两版逐样本predictions.csv、run.log、全部综合XML/RPT保存在对应子目录。',
          '- shared/rtl中的顶层模块及dense_shared模块用于核查硬件实例。',
          '- 运行tools/run.py后运行tools/report.py生成本报告与图。',
          '- 10位接口在2019.2的AXI主接口综合失败日志保存在w10_interface_failure；此限制不能推广到同学的2025.2.1。', '',
          '## 适用范围','',
          '本次完成C仿真和HLS综合。未执行C/RTL协同仿真、布局布线或上板。真实照片采集及Level 2实测仍由组员补齐。']
(out/'experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
