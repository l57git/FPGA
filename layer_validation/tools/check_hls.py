"""Run only AFTER run.py succeeds; require real HLS traces for both cases."""
import csv
import numpy as np
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'level1/tools'))
from lenet_validation import read_lenet_blob
from reference import forward
from compare import read_trace,locate

out=ROOT/'layer_validation/results'
metadata=json.loads((out/'input_metadata.json').read_text())
assert hashlib.sha256((out/'input.bin').read_bytes()).hexdigest()==metadata['input_sha256']
for path,key in [('layer_validation/src/lenet_trace.cpp','source_sha256'),('level1/src/lenet.hpp','header_sha256'),('level1/tb/tb_lenet.cpp','testbench_sha256')]:
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==metadata[key], path
blob=read_lenet_blob(out/'input.bin')
expected=forward(blob.images,blob.parameters)
for case in ['clean','fault']:
    assert 'CSim done with 0 errors' in (out/case/'run.log').read_text(), case+' was not simulated successfully'
clean=locate(expected,read_trace(out/'clean/trace.csv.gz',expected))
assert all(x is None for x in clean),clean
observed_fault=read_trace(out/'fault/trace.csv.gz',expected)
fault=locate(expected,observed_fault)
modeled_fault=forward(blob.images,blob.parameters,inject_pool1=True)
assert all(np.array_equal(modeled_fault[k],observed_fault[k]) for k in expected), 'fault propagation differs'
rows=[{'layer':k,'elements':int(v.size),'clean_mismatches':0,'fault_mismatches':int(np.count_nonzero(v!=observed_fault[k]))} for k,v in expected.items()]
with (out/'hls_layer_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert all(x and x['layer']=='pool1' and x['flat_index']==0 for x in fault),fault
(out/'hls_layer_verification.json').write_text(json.dumps({'status':'PASS','samples':len(clean),'all_clean_layers_bit_exact':True,'checked_elements':sum(r['elements'] for r in rows),'fault_propagation_bit_exact':True,'validation_kind':'HLS C simulation (not RTL cosimulation)','fault_first_layer':'pool1','faults':fault},indent=2))
print('PASS: actual HLS clean and injected intermediate traces verified')
