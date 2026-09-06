"""Full-set PYTHON mixed-fixed inference; only first 1000 have existing HLS evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'level1/tools'))
from lenet_validation import read_lenet_blob,write_lenet_blob
from reference import forward

def main():
    p=argparse.ArgumentParser();p.add_argument('--blob',type=Path,required=True);a=p.parse_args()
    blob=read_lenet_blob(a.blob)
    assert blob.sample_count==10000
    out=ROOT/'layer_validation/results'
    meta=json.loads((ROOT/'mixed_precision/results/w8i1_split/summary.json').read_text())
    write_lenet_blob(out/'candidate_check.bin',blob.parameters,blob.labels[:1000],blob.images[:1000])
    assert hashlib.sha256((out/'candidate_check.bin').read_bytes()).hexdigest()==meta['input_sha256']
    parameters={k: (np.clip(np.floor(v.astype(np.float64)*128+.5),-128,127)/128).astype(np.float32) if k.endswith('_w') else v for k,v in blob.parameters.items()}
    # Q8.1 weights are exactly representable on the golden model's Q*.10 grid.
    logits=[]
    for start in range(0,10000,40):
        logits.append(forward(blob.images[start:start+40],parameters)['logits'])
        if (start+40)%2000==0: print('Python samples',start+40,flush=True)
    logits=np.concatenate(logits)
    with (ROOT/'mixed_precision/results/w8i1_split/predictions.csv').open() as f: hls=list(csv.DictReader(f))
    raw=np.array([[round(float(r[f'logit_{k}'])*1024) for k in range(10)] for r in hls])
    assert np.array_equal(logits[:1000],raw), 'mixed precision golden not validated'
    predictions=logits.argmax(axis=1)
    with (out/'python_mixed_full10000.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','expected','python_prediction']+[f'logit_raw_{i}' for i in range(10)])
        for i in range(10000): w.writerow([i,int(blob.labels[i]),int(predictions[i]),*logits[i].tolist()])
    result={'method':'Python integer fixed-point reference; NOT a new HLS run','samples':10000,'correct':int(np.count_nonzero(predictions==blob.labels)),'accuracy_percent':float(np.mean(predictions==blob.labels)*100),'weight_format':'ap_fixed<8,1,AP_RND,AP_SAT>','activation_format':'ap_fixed<16,6,AP_RND,AP_SAT>','existing_hls_checked_samples':1000,'existing_hls_all_logits_bit_exact':True,'full_hls_status':'pending','full_input_sha256':hashlib.sha256(a.blob.read_bytes()).hexdigest()}
    (out/'python_mixed_full10000.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
