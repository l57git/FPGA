"""Read completed experiment outputs; create reproducible diagnostics, no tuning."""
import argparse,csv,html,json,platform,sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from scipy.stats import binomtest
import scipy,PIL,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing_experiment import ROOT,save_json
from lenet_validation import sha256_file

def wilson(k,n):
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*(c-r),100*(c+r)]

def main(out):
    manifest=json.loads((out/'manifest.json').read_text('utf8'));rows=manifest['samples'];data=Path(manifest['input_root'])
    summary=json.loads((out/'final_summary.json').read_text('utf8'));base=json.loads((out/'baseline_summary.json').read_text('utf8'));dev=json.loads((out/'development_results.json').read_text('utf8'))
    final=np.load(out/f"final_{summary['selected']}.npz");old=np.load(out/'baseline_inputs.npz');legacy=np.load(out/'legacy_predictions.npz')['prediction'];raw=np.load(out/'direct_predictions.npz')['prediction']
    p=final['predictions'];y=old['labels'];test=old['split']=='test';new=final['images'];diagnostics=json.loads((out/'diagnostics.json').read_text('utf8'))
    stats={'held_out_wilson_95_percent':wilson(int(((p==y)&test).sum()),int(test.sum())),'paired':{},'rejection_reasons':dict(Counter(d['status'] for d in diagnostics if d['status']!='ok'))}
    for label,b in [('legacy',legacy),('direct',raw)]:
        stats['paired'][label]={}
        for split,m in [('all',np.ones(len(y),bool)),('test',test)]:
            gain=int(((b!=y)&(p==y)&m).sum());loss=int(((b==y)&(p!=y)&m).sum())
            stats['paired'][label][split]={'corrected':gain,'regressed':loss,'both_wrong':int(((b!=y)&(p!=y)&m).sum()),'exact_mcnemar_p':float(binomtest(min(gain,loss),gain+loss,.5).pvalue) if gain+loss else 1.0}
    save_json(out/'paired_statistics.json',stats)
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(3);a=[base['baseline']['direct']['test']['accuracy_percent'],base['baseline']['legacy']['test']['accuracy_percent'],summary['python_float']['test']['accuracy_percent']]
    bars=ax.bar(x,a,color=['#8899AA','#B87961','#2459A6']);ax.bar_label(bars,labels=[f'{v:.2f}%' for v in a],padding=3);ax.set(xticks=x,xticklabels=['Direct resize','Legacy','Robust v1'],ylim=(0,105),ylabel='Accuracy (%)',title='Held-out image split (145 images; rejects count as errors)');fig.tight_layout();fig.savefig(out/'heldout_accuracy.png',dpi=160);plt.close(fig)
    cm=np.zeros((10,11),int)
    for a,b in zip(y[test],p[test]):cm[a,10 if b==-1 else b]+=1
    fig,ax=plt.subplots(figsize=(9,6));ax.imshow(cm,cmap='Blues');ax.set(xticks=range(11),xticklabels=list('0123456789')+['Reject'],yticks=range(10),xlabel='Prediction',ylabel='True class',title='Held-out confusion matrix (145 images)')
    for a in range(10):
        for b in range(11):ax.text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]>10 else 'black')
    fig.tight_layout();fig.savefig(out/'heldout_confusion.png',dpi=160);plt.close(fig)
    (out/'gallery').mkdir(exist_ok=True)
    sections=[]
    for i,r in enumerate(rows):
        thumb=Image.new('RGB',(490,205),'white');draw=ImageDraw.Draw(thumb)
        with Image.open(data/r['relative_path']) as im:thumb.paste(ImageOps.contain(im.convert('RGB'),(115,185)),(0,20))
        for j,(name,img) in enumerate([('Legacy',old['legacy'][i]),('Robust v1',new[i])]):
            thumb.paste(Image.fromarray((img*255).astype('uint8')).resize((160,160)),(125+j*180,28));draw.text((125+j*180,6),name,fill='black')
        thumb.save(out/'gallery'/f'{i:03}.jpg',quality=90)
        state='wrong' if p[i]!=y[i] else 'correct';typ='test' if test[i] else 'dev'
        sections.append(f'<article class="{state} {typ}"><p>#{i} {typ} 标签={y[i]} 旧预测={legacy[i]} 新预测={p[i]} 状态={diagnostics[i]["status"]}</p><img loading="lazy" src="gallery/{i:03}.jpg"><small>{html.escape(r["relative_path"])}</small></article>')
    page='''<!doctype html><meta charset="utf-8"><title>预处理完整审查</title><style>body{font-family:Arial,sans-serif;background:#edf2f7;color:#172638;margin:24px}main{display:flex;flex-wrap:wrap;gap:18px}article{background:white;padding:12px;width:490px}.wrong{border-left:4px solid #a14d19}small{overflow-wrap:anywhere}button{padding:8px;margin:6px}</style><h1>全部 503 张：原图、旧预处理、新预处理</h1><p>所有样本保留。-1 表示拒识，计为错误。测试结果已冻结，页面只用于事后审查。</p><button onclick="filter('all')">全部</button><button onclick="filter('wrong')">全部失败</button><button onclick="filter('test')">保留测试集</button><button onclick="filter('testwrong')">测试集失败</button><main>'''+''.join(sections)+'''</main><script>function filter(t){document.querySelectorAll('article').forEach(a=>a.style.display=(t==='all'||(t==='testwrong'?a.classList.contains('test')&&a.classList.contains('wrong'):a.classList.contains(t)))?'block':'none')}</script>'''
    (out/'完整样本审查.html').write_text(page,encoding='utf8')
    # Every test failure is displayed, in stable dataset order, not curated.
    failed=np.flatnonzero(test&(p!=y));sheet=Image.new('RGB',(1000,((len(failed)+1)//2)*240),'white');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(failed):
        x=j%2*500;yy=j//2*240;thumb=Image.open(out/'gallery'/f'{i:03}.jpg');sheet.paste(thumb,(x,yy+25));draw.text((x+5,yy+3),f'index={i} true={y[i]} old={legacy[i]} new={p[i]}',fill='black')
    sheet.save(out/'all_test_failures.jpg',quality=95)
    versions={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'Pillow':PIL.__version__,'matplotlib':matplotlib.__version__}
    save_json(out/'environment.json',versions)
    save_json(out/'source_hashes.json',{p.name:sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('preprocess_robust.py'),Path(__file__).with_name('evaluate_robust.py'),Path(__file__).with_name('preprocessing_experiment.py')]})
    lines=['# Level 2 预处理改进：冻结配置后的实测结果','', '## 结论','模型参数和原始图片保持不变。新流程的完整 503 张准确率为 **429/503 = 85.29%**；预留的 145 张图像级测试集为 **126/145 = 86.90%**。8 张拒识样本全部计入 503 的分母，计为错误。','', '这些是本机 Python 浮点结果。独立整数定点参考同为 429/503，但与浮点有 1 张类别不同（502/503 一致）。**未运行新的 HLS C Simulation，也未验证 FPGA 预处理实现。**','', '## 实验设计与诚实边界','按图像解码后的像素 SHA256 固定划分 358 张开发集和 145 张测试集，不按文件名或类别制定预处理规则。没有完全相同的像素副本。没有书写者/拍摄批次信息，不能保证近似照片跨集独立，因此这是内部图像级留出，不能称作跨书写者或外部盲测。原项目已经整体评估过本数据集。','只在开发集选择两个最终候选（居中、不加粗 / 居中、2×2 灰度加粗）。两者开发集都为 303/358，按预先约定的平局规则选择不加粗。预处理源码、参数与清单哈希写入 selection.json 后才评估新方法的测试集表现；之后没有修改预处理器。','', '## 已复现的基线和最终对照','|划分|直接缩放|旧预处理|新预处理|','|---|---:|---:|---:|']
    for label,key in [('全部 503 张','all'),('开发集 358 张','dev'),('测试集 145 张','test')]:
        v=[base['baseline']['direct'][key],base['baseline']['legacy'][key],summary['python_float'][key]]
        lines.append('|'+label+'|'+'|'.join(f"{z['correct']}/{z['count']} ({z['accuracy_percent']:.2f}%)" for z in v)+'|')
    lines+=['','## 开发集逐步消融','|阶段|正确数|准确率|','|---|---:|---:|']
    names={'contrast':'内部背景估计＋单向笔画对比','frame':'再筛除宽大近边框区域','component':'再选择主体＋几何居中','centered':'再质心居中（最终选用）','thickened':'额外加粗（未选用）'}
    for k,v in dev.items():lines.append(f"|{names[k]}|{v['correct']}/358|{v['accuracy_percent']:.2f}%|")
    lines+=['','消融是顺序添加模块的对照，不代表各模块完全独立的因果贡献。','', '## 新流程','1. 处理 EXIF 方向，灰度化并等比例降采样到最长边 512，降低 CPU 处理开销。','2. 使用内部区域的中位灰度估计占多数的背景，按极性只提取暗笔画或亮笔画，避免绝对差把纸张也当成前景。','3. 使用 Otsu 阈值并设置最低对比度 18，提取连通区域。','4. 根据相对跨度、宽高和靠边位置筛除页框，禁止逐图固定裁剪。','5. 选择主要连通区域，保留满足面积、距离和方向条件的邻近断笔。存在其他较大区域时拒识。小符号只按统一几何规则处理，不使用“%”模板，也不识别后按正确标签选择裁剪。','6. 保留主体及一像素笔画边缘的灰度信息，按笔画灰度归一化，等比例放入 20×20 区域，再补黑边为 28×28。','7. 质心对齐，输出 float32 [0,1]。不加粗，不修改网络，不重训，不使用类别专属规则。','', '## 未解决问题','完整集数字 9 只有 23/50，测试集数字 9 为 5/15，显著弱于其他类别。某些 9 的书写形态、伴随符号和纸张阴影仍引发误判。部分框线与阴影连成 L 形，宽大页框筛除并不能完全覆盖；邻近的百分号也可能被当作断笔并入。不能把剩余 74 张失败都归咎于模型或都归咎于预处理。','没有把这些样本删除、改标签或改用正确答案进行裁剪。所有 19 张测试集失败见 all_test_failures.jpg，全部 503 张见完整样本审查.html。','缺少真实背景负样本，不能报告现实背景误检率。合成空白、页框、反色和多目标等 13 项测试（含原有 6 项）通过，仅支持程序行为，不替代实拍背景实验。','数据来源仍缺采集记录。文件名不足以证明人工实拍或生成图，报告不作推断。','', '## 配对变化与不确定性']
    for k in ['legacy','direct']:
        v=stats['paired'][k]['test'];lines.append(f"相对 {k}，测试集纠正 {v['corrected']} 张，同时使原来正确的 {v['regressed']} 张变错，两者都错 {v['both_wrong']} 张。")
    lo,hi=stats['held_out_wilson_95_percent'];lines.append(f'145 张测试集的 Wilson 95% 区间为 {lo:.2f}%–{hi:.2f}%。该区间假设样本独立，不包含未知书写者/近重复造成的偏差。')
    lines+=['','## HLS 交接','preprocessed_accepted.bin 含 495 张未拒识输入，已按原格式回读核对；python_results.csv 是浮点参考，integer_reference_results.csv 是独立 W16 整数参考。hls_input_mapping.csv 保留 blob 索引与原样本及划分的对应。','HLS 接受集准确率的分母是 495；端到端准确率必须用全部 503，并把 8 个拒识计为错误。当前十分类 HLS 无 -1 拒识输出，拒识发生在 CPU 预处理端。不能用 495 作分母宣称整条流程提高到更高准确率。','本机命令路径和仓库指定 E:/use/cpu/Vivado/2019.2 未找到可用 HLS。可在队友环境运行新增 run_robust_hls.bat 后比较，不能套用旧的 503/503 一致性结论。','', '## 后续工作','保持本版冻结。优先核实来源、补充不同书写者与拍摄批次的新测试图片和背景负样本。对数字 9、L 形阴影与伴随符号制定下一轮通用方案时，只在开发集迭代，并用新采样本检验。不要继续反复用这 145 张测试集挑参数。']
    (out/'改进实验报告.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'level2/results/preprocessing_v1');main(p.parse_args().output)
