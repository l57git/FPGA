"""Validate trace structure and locate the first divergent layer per sample."""
import csv
import gzip
import numpy as np
from reference import LAYERS

def read_trace(path,reference):
    arrays={k:np.zeros_like(v).reshape(len(v),-1) for k,v in reference.items()}
    seen={k:np.zeros(v.shape,dtype=bool) for k,v in arrays.items()}
    with gzip.open(path,'rt',newline='') as f:
        for sample,layer,index,raw in csv.reader(f):
            sample,index,raw=int(sample),int(index),int(raw)
            if layer not in arrays or not (0<=sample<len(arrays[layer]) and 0<=index<arrays[layer].shape[1]):
                raise ValueError('Unknown layer or out-of-range trace coordinate')
            if seen[layer][sample,index]: raise ValueError('Duplicate trace coordinate')
            arrays[layer][sample,index]=raw;seen[layer][sample,index]=True
    if not all(v.all() for v in seen.values()): raise ValueError('Incomplete trace')
    return {k:v.reshape(reference[k].shape) for k,v in arrays.items()}

def locate(expected,observed):
    diagnostics=[]
    for sample in range(len(expected['logits'])):
        first=None
        for layer in LAYERS:
            a,b=expected[layer][sample].ravel(),observed[layer][sample].ravel()
            mismatch=np.flatnonzero(a!=b)
            if mismatch.size:
                i=int(mismatch[0])
                first=dict(sample=sample,layer=layer,flat_index=i,expected_raw=int(a[i]),observed_raw=int(b[i]),mismatch_elements=int(mismatch.size))
                break
        diagnostics.append(first)
    return diagnostics
