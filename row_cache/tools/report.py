"""Verify identical inputs and outputs, then plot three measured architectures."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'row_cache/results'
rows=json.loads((ROOT/'resource_reuse/results/summary.json').read_text())+json.loads((out/'summary.json').read_text())
for key in ['width','samples','tool','target_period_ns','input_sha256']:
    assert len({r[key] for r in rows})==1, key
# The prior summary predates header hashes; check its recorded Git revision.
for rel, key in [('level1/src/lenet.hpp','header_sha256'),('level1/tb/tb_lenet.cpp','testbench_sha256')]:
    old=subprocess.check_output(['git','show','6ae5cea:'+rel],cwd=ROOT)
    current=(ROOT/rel).read_bytes()
    assert old.replace(b'\r\n',b'\n')==current.replace(b'\r\n',b'\n'), rel
    assert hashlib.sha256(current).hexdigest()==rows[-1][key], key
pred=[]
for path in [ROOT/'resource_reuse/results/baseline/predictions.csv',ROOT/'resource_reuse/results/shared/predictions.csv',out/'row_cache/predictions.csv']:
    with path.open() as f: pred.append(list(csv.DictReader(f)))
assert len(pred[0])==rows[0]['samples']
assert pred[0]==pred[1]==pred[2], 'per-sample values differ'
rtl=(out/'row_cache/rtl/lenet_accel.v').read_text()
instances=re.findall(r'^dense_row\s+(\w+)\s*\(',rtl,re.M)
assert len(instances)==1, instances
(out/'verification.json').write_text(json.dumps({'samples':len(pred[0]),'all_csv_fields_identical':True,'input_sha256':rows[0]['input_sha256'],'shared_row_core_instances':instances},indent=2))
fields=['variant','samples','accuracy_percent','DSP','LUT','FF','BRAM','cycles_min','cycles_max','estimated_clock_ns']
with (out/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
labels=['Baseline','Shared/full layer','Shared/row cache']
colors=['#527da3','#da8b39','#39977c']
fig,axes=plt.subplots(1,4,figsize=(14,4.4))
for ax,key in zip(axes,['DSP','LUT','FF','BRAM']):
    values=[int(r[key]) for r in rows]
    bars=ax.bar(labels,values,color=colors); ax.bar_label(bars,padding=3)
    ax.set_title(key); ax.set_ylim(0,max(values)*1.23+1); ax.tick_params(axis='x',labelrotation=25)
fig.suptitle('HLS estimates: 16 bit, Vivado HLS 2019.2, XC7Z020, 10 ns')
fig.tight_layout(); fig.savefig(out/'resources.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
for i,r in enumerate(rows):
    lo,hi=[int(r[k])*1e-5 for k in ['cycles_min','cycles_max']]
    ax.plot([lo,hi],[i,i],lw=6,marker='o',color=colors[i]); ax.text((lo+hi)/2,i+.14,f'{lo:.3f} - {hi:.3f} ms',ha='center')
ax.set(yticks=range(3),yticklabels=labels,xlabel='HLS latency estimate at target clock (ms)',ylim=(-.5,2.6),xlim=(0,max(int(r['cycles_max'])*1e-5 for r in rows)*1.1))
ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(out/'latency.png',dpi=180); plt.close(fig)
lines=['# 逐行权重缓存实验报告','','## 方法','','在已完成的跨层共享FC架构上，将30720元素的整层权重缓冲改为256元素的逐行缓冲，计算一个输出神经元前加载其权重和bias。三个FC层共用一个dense_row乘加核。每层全部输出计算完成后才更新输入缓冲，避免覆盖仍在使用的输入。卷积、池化、bias、ReLU、argmax及累加顺序保持不变，不使用softmax。','','16位权重缓冲理论容量从61440字节降至512字节，缩小120倍；这不是整个设计的BRAM降幅。取消整层bias缓冲，保留256元素输入和120元素输出缓冲。每次推理仍需读取全部FC权重，未减少外存数据量。','','## 控制变量与验证','','同一MNIST前1000张、同一模型参数、16位数据/6位整数、32位累加器/14位整数、Vivado HLS 2019.2、XC7Z020、10ns。复用上一个实验的两组真实结果，不重新运行同学的精度扫描。脚本强制检查输入、头文件和testbench哈希一致，并逐字段比较全部CSV记录。','','|指标|原版|整层缓存共享FC|逐行缓存共享FC|','|---|---:|---:|---:|']
for k in fields[1:]: lines.append('|'+k+'|'+'|'.join(str(r[k]) for r in rows)+'|')
lines+=['',f'三版全部{len(pred[0])}条记录一致，包括预测、标签和10个logits。逐行缓存的RTL顶层只有一个dense_row实例：`{instances[0]}`。','', '相对整层缓存，资源变化：'+', '.join(f'{k} {(int(rows[2][k])/int(rows[1][k])-1)*100:+.2f}%' for k in ['DSP','LUT','FF','BRAM'])+'。','','## 结论与边界','','本模块验证的是片上存储优化，以及共享计算核在缩小缓存后能否保持数值一致。资源取自HLS综合估计；动态循环的延迟范围不能当作实测吞吐或加速比。缓存减小可能改变突发传输和调度，需进一步C/RTL协同仿真、布局布线和上板确认。1000张准确率不能代替完整10000张测试准确率。','','## 留存证据与待收集材料','','results/row_cache保存逐样本结果、HLS完整日志、综合XML/RPT和生成RTL；verification.json记录一致性与单实例检查，metrics.csv及两张PNG用于报告和PPT。上一个实验结果保留在resource_reuse/results，没有覆盖。','','尚需组员收集：开发板型号及连接照片、实际运行输出截图、实现后的资源和时序截图；若报告要求GUI截图，可打开留存的综合报告截取，不能把本次图表称为上板截图。本模块没有完成上板或RTL协同仿真。']
(out/'experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
manifest={}
for p in sorted((ROOT/'row_cache').rglob('*')):
    if p.is_file() and 'hls_work' not in p.parts and '__pycache__' not in p.parts and p.suffix!='.bin' and p.name!='manifest.json' and ('results' in p.parts or p.suffix=='.cpp'):
        manifest[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(rows[-1],indent=2)); print('PASS: all records identical; one shared row core')
