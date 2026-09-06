"""Record model weight ranges and AP_RND/AP_SAT parameter quantization error."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'level1/tools'))
from lenet_validation import read_lenet_blob
blob=read_lenet_blob(Path(sys.argv[1]))
rows=[]
for name,values in blob.parameters.items():
    if not name.endswith('_w'): continue
    x=values.astype(np.float64)
    for width,integer in [(16,6),(8,6),(8,1)]:
        scale=2**(width-integer)
        rounded=np.floor(x*scale+0.5)
        lo,hi=-2**(width-1),2**(width-1)-1
        q=np.clip(rounded,lo,hi)/scale
        rows.append(dict(tensor=name,count=x.size,minimum=float(x.min()),maximum=float(x.max()),weight_width=width,weight_integer=integer,fractional_bits=width-integer,saturated=int(np.count_nonzero((rounded<lo)|(rounded>hi))),zero_fraction=float(np.mean(q==0)),mse=float(np.mean((q-x)**2)),max_abs_error=float(np.max(np.abs(q-x)))))
out=ROOT/'mixed_precision/results'
with (out/'parameter_ranges.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'parameter_ranges.json').write_text(json.dumps(rows,indent=2))
print('Weight count:',sum(x['count'] for x in rows if x['weight_width']==16))
print('Saturated parameters:',sum(x['saturated'] for x in rows))
