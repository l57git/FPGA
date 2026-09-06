"""Compare measured cases; never substitute a requested II for achieved II."""
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'mixed_precision/results'
names = ['w16i6_serial', 'w16i6_pipeline', 'w8i6_serial', 'w8i1_serial', 'w8i1_pipeline', 'w8i1_split']
paths = [ROOT/'row_cache/results/row_cache'] + [OUT/n for n in names[1:]]
base = json.loads((ROOT/'row_cache/results/summary.json').read_text())[0]
base.update(activation_width=16, weight_width=16, weight_integer=6, fc_pipeline=0)
rows = [base] + [json.loads((p/'summary.json').read_text()) for p in paths[1:]]
predictions = []
for name, row, path in zip(names, rows, paths):
    row['variant'] = name
    assert row['input_sha256'] == base['input_sha256']
    assert row['samples'] == base['samples'] == 1000
    assert row['tool'] == base['tool'] and row['target_period_ns'] == 10
    if name != 'w16i6_serial':
        source = ROOT / ('mixed_precision/src/lenet_split.cpp' if name=='w8i1_split' else 'mixed_precision/src/lenet_mixed.cpp')
        for file,key in [(source,'source_sha256'),(ROOT/'mixed_precision/src/lenet.hpp','header_sha256'),(ROOT/'mixed_precision/tb/tb_lenet.cpp','testbench_sha256')]:
            assert hashlib.sha256(file.read_bytes()).hexdigest()==row[key], str(file)
        assert 'CSim done with 0 errors' in (path/'run.log').read_text()
    with (path/'predictions.csv').open() as f:
        pred = list(csv.DictReader(f))
    assert len(pred) == row['samples']
    predictions.append(pred)
    if len(predictions)>1:
        assert [(x['index'],x['expected']) for x in pred] == [(x['index'],x['expected']) for x in predictions[0]]
    row['prediction_matches_base'] = sum(x['prediction']==y['prediction'] for x,y in zip(predictions[0],pred))
    row['logit_mae_base'] = float(np.mean([abs(float(x[f'logit_{i}'])-float(y[f'logit_{i}'])) for x,y in zip(predictions[0],pred) for i in range(10)]))
    core = ET.parse(path/'dense_row_csynth.xml')
    row['core_ii'] = core.findtext('.//SummaryOfLoopLatency/DENSE_I/PipelineII') or 'not pipelined'
    row['core_cycles_min'] = int(core.findtext('.//SummaryOfOverallLatency/Best-caseLatency'))
    row['core_cycles_max'] = int(core.findtext('.//SummaryOfOverallLatency/Worst-caseLatency'))
    rtl = (path/'rtl/lenet_accel.v').read_text()
    assert len(re.findall(r'^dense_row\s+\w+\s*\(',rtl,re.M)) == 1
    row['estimated_clock_within_target'] = float(row['estimated_clock_ns']) <= 10
    row['eligible_subset'] = row['accuracy_percent'] >= 90 and base['accuracy_percent']-row['accuracy_percent']<=0.5+1e-9
assert 'bitwidth mismatch' in (OUT/'w8i1_serial/run.log').read_text()
assert 'bitwidth mismatch' not in (OUT/'w8i1_split/run.log').read_text()
assert predictions[0] == predictions[1], '16-bit pipeline changed arithmetic'
assert predictions[3] == predictions[5], 'interface change altered outputs'
assert predictions[3] == predictions[4], '8-bit pipeline changed arithmetic'
(OUT/'comparison.json').write_text(json.dumps({'samples':1000,'w16_pipeline_all_fields_identical':True,'w8i1_pipeline_all_fields_identical':True,'interface_all_fields_identical':True,'single_core_all_cases':True,'input_sha256':base['input_sha256']},indent=2))
(OUT/'metrics.json').write_text(json.dumps(rows,indent=2))
fields = ['variant','weight_width','weight_integer','fc_pipeline','accuracy_percent','prediction_matches_base','logit_mae_base','DSP','BRAM','LUT','FF','cycles_min','cycles_max','estimated_clock_ns','core_ii','core_cycles_min','core_cycles_max','estimated_clock_within_target','eligible_subset']
with (OUT/'metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
labels=['W16 I6\nserial','W16 I6\npipeline','W8 I6\nserial','W8 I1\nserial','W8 I1\npipeline','W8 I1\nsplit AXI']
colors=['#527da3','#78a5bd','#ce6552','#529b70','#90b64d','#846bb0']
fig, axes=plt.subplots(1,3,figsize=(14,4.5))
for ax,key in zip(axes,['accuracy_percent','LUT','BRAM']):
    values=[float(r[key]) for r in rows]
    bars=ax.bar(labels,values,color=colors);ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set_title(key); ax.set_ylim(0,max(values)*1.16+1);ax.tick_params(axis='x',labelsize=8)
fig.suptitle('1000-sample C simulation and HLS synthesis; activations/biases fixed at 16 bits')
fig.tight_layout();fig.savefig(OUT/'accuracy_resources.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,keys,title in [(axes[0],('cycles_min','cycles_max'),'Top-level HLS estimated cycles'),(axes[1],('core_cycles_min','core_cycles_max'),'Shared FC row-core estimated cycles')]:
    for i,r in enumerate(rows):
        lo,hi=int(r[keys[0]]),int(r[keys[1]])
        ax.plot([lo,hi],[i,i],lw=5,marker='o',color=colors[i])
    ax.set(yticks=range(6),yticklabels=[x.replace('\n',' ') for x in labels],xlabel=title);ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'pipeline_latency.png',dpi=180);plt.close(fig)
quant=json.loads((OUT/'parameter_ranges.json').read_text())
fig,ax=plt.subplots(figsize=(8,4))
for i,(w,b) in enumerate([(16,6),(8,6),(8,1)]):
    selected=[r for r in quant if r['weight_width']==w and r['weight_integer']==b]
    ax.bar(np.arange(5)+(i-1)*.25,[r['mse'] for r in selected],width=.25,label=f'W{w} I{b}')
ax.set(yscale='log',xticks=range(5),xticklabels=[r['tensor'] for r in selected],ylabel='Weight quantization MSE');ax.legend()
fig.tight_layout();fig.savefig(OUT/'quantization_error.png',dpi=180);plt.close(fig)
lines=['# 混合精度与共享FC流水化实验报告','','## 对应课程要求','','原任务书第5页提出权重与激活采用异构定点格式；第6页任务3进阶实验可选择该方向。这里完成权重/激活差异化位分配，并增加独立的流水化调度实验。PIPELINE不是UNROLL，不将本实验称为已完成并行度展开扫描或资源溢出边界搜索。','','## 设计与控制变量','','基于逐行缓存架构，卷积和三个FC层的权重统一使用weight_t；输入像素、激活、logits与bias均保持ap_fixed<16,6>，累加器保持ap_fixed<32,14>。所有类型采用AP_RND和AP_SAT。三个FC层仍共享一个dense_row实例。流水化只在该核心的内层循环请求II=1，保持累加顺序，不引入并行部分和。','','共44190个权重，实际范围约[-0.423,0.395]。W8 I1包含符号位，范围[-1,0.9921875]、步长1/128；W8 I6步长1/4。参数范围分析记录每层MSE、零值比例和饱和数量，三种格式均无参数饱和。此处没有重新缩放、重新训练或更改bias，也没有softmax。','','权重有效载荷理论上从88380字节减至44190字节。AXI接口打包、burst和总线占用由综合决定，不能直接声称实测带宽减半。','','同一前1000张MNIST、同一模型和数据顺序、Vivado HLS 2019.2、XC7Z020、10ns目标周期。沿用已完成W16串行结果，只运行五个新配置，不重复同学的完整位宽扫描。','','## 结果','','|配置|准确率%|DSP|BRAM18K|LUT|FF|核心实际II|核心周期范围|整网周期范围|','|---|---:|---:|---:|---:|---:|---|---|---|']
for r in rows:
    lines.append(f"|{r['variant']}|{r['accuracy_percent']:.2f}|{r['DSP']}|{r['BRAM']}|{r['LUT']}|{r['FF']}|{r['core_ii']}|{r['core_cycles_min']}–{r['core_cycles_max']}|{r['cycles_min']}–{r['cycles_max']}|")
lines += ['', '两组流水开关对照中，1000条CSV全部字段（包括10个logits）完全一致。不同权重格式的预测一致数量和logit误差另见metrics.csv，不要求量化后的输出与16位完全相同。','','## 工程判断','','接口对照将16位bias移至gmem0，与16位image/logits共用接口，gmem1只保留8位权重，AXI主接口仍为两个。日志中conv1/conv2权重因8/16位混用而禁用burst的警告消失，整网延迟估计范围由3206196–4559226降至1414836–2767866周期。LUT 7010→6866，FF 4180→4066，DSP和BRAM不变。全部1000条输出保持一致。这是接口与调度的综合结果，尚非实测带宽或加速比。','','16位流水版虽然达到核心II=1，但估计时钟10.601ns超过10ns目标，不能作为满足时钟约束的配置。8位流水版的时钟估计为8.750ns。综合成功与达到目标时钟是两件事。','','子集筛选门槛为准确率≥90%且相对16位下降≤0.5个百分点。通过的配置：'+', '.join(r['variant'] for r in rows if r['eligible_subset'])+'。这是1000张上的候选筛选，不能替代10000张完整评估或修改同学基于完整数据集的W10结论。','','在本次1000张筛选和HLS估计范围内，W8 I1串行分接口版是后续完整数据集确认的优先候选，不能据此声称全局最优。W8 I6仅59.7%，而W8 I1达到98.3%，说明权重小数精度比不必要的整数范围更重要。','','流水化II以dense_row_csynth.xml实际报告为准；请求II=1不等于实现II=1。整网仍包括卷积、参数访存、行缓存装载及控制开销，核心周期降低不代表同等比例的整网加速。动态tripcount使顶层延迟给出范围，不能用最小值声称实测FPS。','','全部资源和时钟数字均为HLS综合估计，没有布局布线、功耗实测或上板数据。任务书第1和第5页明确只要求仿真和综合，不要求上板。本目录的1000张准确率来自C仿真；RTL协同仿真见独立rtl_validation模块，不能把其通过自动推广至本目录所有变体。','','## 复现和证据','','每个配置目录保存源码/输入哈希、逐样本CSV、运行日志、顶层和核心综合XML/RPT及RTL。tools/build_variant.py从既有源码生成独立版本，不修改Level 1或row_cache。tools/run.py运行单配置，tools/report.py验证对照后生成报告和图。参数分布图来自模型原始浮点权重的定点舍入分析，不是替代HLS推理的近似准确率。','','仍需完成：正式候选的完整测试集确认、自采数据Level 2验收、循环展开并行度/资源边界探索，以及团队最终报告和现场演示。']
(OUT/'experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('PASS: two pipeline equivalence checks, six single-core checks; reports generated')
