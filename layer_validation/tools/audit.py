"""Offline verification against EXISTING HLS results, not a new HLS run."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'level1/tools'))
from lenet_validation import read_lenet_blob, write_lenet_blob, conv2d_valid, max_pool_2x2
from reference import forward, LAYERS
from compare import read_trace, locate

def float_layers(images,p):
    d={}
    d['conv1']=np.maximum(conv2d_valid(images[:,None],p['conv1_w'][:,None],p['conv1_b']),0)
    d['pool1']=max_pool_2x2(d['conv1'])
    d['conv2']=np.maximum(conv2d_valid(d['pool1'],p['conv2_w'],p['conv2_b']),0)
    d['pool2']=max_pool_2x2(d['conv2'])
    d['fc1']=np.maximum(d['pool2'].reshape(len(images),-1)@p['fc1_w'].T+p['fc1_b'],0)
    d['fc2']=np.maximum(d['fc1']@p['fc2_w'].T+p['fc2_b'],0)
    d['logits']=d['fc2']@p['fc3_w'].T+p['fc3_b']
    return d

def save_trace(path,layers):
    text=io.StringIO()
    w=csv.writer(text,lineterminator='\n')
    for sample in range(len(layers['logits'])):
        for layer in LAYERS:
            for index,raw in enumerate(layers[layer][sample].ravel()):
                w.writerow([sample,layer,index,int(raw)])
    path.write_bytes(gzip.compress(text.getvalue().encode(),mtime=0))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--blob',type=Path,required=True);a=parser.parse_args()
    out=ROOT/'layer_validation/results'
    blob=read_lenet_blob(a.blob)
    baseline=ROOT/'row_cache/results/row_cache/predictions.csv'
    meta=json.loads((ROOT/'row_cache/results/summary.json').read_text())[0]
    batch=out/'reference_input.bin'
    write_lenet_blob(batch,blob.parameters,blob.labels[:1000],blob.images[:1000])
    assert hashlib.sha256(batch.read_bytes()).hexdigest()==meta['input_sha256'], 'wrong input or parameter source'
    with baseline.open() as f: previous=list(csv.DictReader(f))
    assert len(previous)==1000
    logits=[]
    for start in range(0,1000,20):
        logits.append(forward(blob.images[start:start+20],blob.parameters)['logits'])
    logits=np.concatenate(logits)
    hls=np.array([[round(float(row[f'logit_{k}'])*1024) for k in range(10)] for row in previous],dtype=np.int64)
    assert np.array_equal(logits,hls), 'integer golden differs from existing HLS outputs'
    assert np.array_equal(blob.labels[:1000],[int(r['expected']) for r in previous])
    pred=logits.argmax(axis=1)
    with (out/'golden_vs_existing_hls.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','expected','python_fixed_prediction','existing_hls_prediction','all_logits_equal'])
        for i,row in enumerate(previous): w.writerow([i,int(blob.labels[i]),int(pred[i]),row['prediction'],True])
    indices=[int(i) for label in range(10) for i in np.flatnonzero(blob.labels==label)[:2]]
    images=blob.images[indices]
    expected=forward(images,blob.parameters)
    fault=forward(images,blob.parameters,inject_pool1=True)
    software=float_layers(images,blob.parameters)
    save_trace(out/'python_reference_trace.csv.gz',expected)
    save_trace(out/'python_fault_trace.csv.gz',fault)
    clean_checked=read_trace(out/'python_reference_trace.csv.gz',expected)
    assert all(x is None for x in locate(expected,clean_checked))
    diagnoses=locate(expected,read_trace(out/'python_fault_trace.csv.gz',expected))
    assert all(x and x['layer']=='pool1' and x['flat_index']==0 and x['observed_raw']-x['expected_raw']==1024 for x in diagnoses)
    (out/'fault_localization.json').write_text(json.dumps({'source':'Python synthetic fault, NOT HLS fault-run','injection':'pool1[0][0][0] += 1.0 in every sample','diagnoses':diagnoses},indent=2))
    errors=[]
    for layer in LAYERS:
        delta=expected[layer]/1024-software[layer]
        errors.append(dict(layer=layer,values=int(delta.size),mean_absolute_error=float(np.abs(delta).mean()),max_absolute_error=float(np.abs(delta).max()),output_at_limit=int(np.count_nonzero((expected[layer]==32767)|(expected[layer]==-32768)))))
    with (out/'layer_float_fixed_error.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(errors[0]));w.writeheader();w.writerows(errors)
    result=dict(status='offline_audit_pass',existing_hls_samples=1000,existing_hls_logit_values=10000,all_logits_bit_exact=True,correct=int((pred==blob.labels[:1000]).sum()),trace_samples=20,source_indices=indices,python_fault_localized=20,new_hls_trace_status='NOT RUN: E drive/HLS installation unavailable',baseline_csv_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),input_sha256=meta['input_sha256'])
    (out/'offline_summary.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(LAYERS,[r['mean_absolute_error'] for r in errors],color='#438c90');axes[0].set(ylabel='Mean absolute error',title='Python fixed vs Python float (20 images)')
    axes[0].tick_params(axis='x',rotation=30)
    axes[1].imshow(expected['conv1'][0,0]/1024,cmap='viridis');axes[1].set_title('Python fixed Conv1: sample 0, channel 0');axes[1].set_axis_off()
    fig.tight_layout();fig.savefig(out/'layer_diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
