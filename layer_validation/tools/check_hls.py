"""Run only AFTER run.py succeeds; require real HLS traces for both cases."""
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
blob=read_lenet_blob(out/'input.bin')
expected=forward(blob.images,blob.parameters)
for case in ['clean','fault']:
    assert 'CSim done with 0 errors' in (out/case/'run.log').read_text(), case+' was not simulated successfully'
clean=locate(expected,read_trace(out/'clean/trace.csv.gz',expected))
assert all(x is None for x in clean),clean
fault=locate(expected,read_trace(out/'fault/trace.csv.gz',expected))
assert all(x and x['layer']=='pool1' and x['flat_index']==0 for x in fault),fault
(out/'hls_layer_verification.json').write_text(json.dumps({'status':'PASS','samples':len(clean),'all_clean_layers_bit_exact':True,'fault_first_layer':'pool1','faults':fault},indent=2))
print('PASS: actual HLS clean and injected intermediate traces verified')
