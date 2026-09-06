"""Integer golden model for data Q16.6 and accumulator Q32.14, AP_RND/AP_SAT.

Integer widths include the sign. AP_RND ties go towards positive infinity.
Accumulate in source order with saturation after every MAC, not a dot shortcut.
"""
import numpy as np

LAYERS = ['conv1','pool1','conv2','pool2','fc1','fc2','logits']

def quantize(x):
    return np.clip(np.floor(np.asarray(x,dtype=np.float64)*1024+0.5),-32768,32767).astype(np.int64)

def mac(acc,a,b):
    # Product of two Q*.10 data values, rounded into Q*.18 accumulator units.
    return np.clip(acc + (a*b+2)//4, -(1<<31), (1<<31)-1)

def output(acc):
    return np.clip((acc+128)//256,-32768,32767)

def conv(x,w,b):
    n,channels,h,width=x.shape
    outputs,_,kh,kw=w.shape
    result=np.empty((n,outputs,h-kh+1,width-kw+1),dtype=np.int64)
    for oc in range(outputs):
        acc=np.full((n,h-kh+1,width-kw+1),b[oc]*256,dtype=np.int64)
        for ic in range(channels):
            for ky in range(kh):
                for kx in range(kw):
                    acc=mac(acc,x[:,ic,ky:ky+h-kh+1,kx:kx+width-kw+1],w[oc,ic,ky,kx])
        result[:,oc]=np.maximum(output(acc),0)
    return result

def pool(x):
    n,c,h,w=x.shape
    return x.reshape(n,c,h//2,2,w//2,2).max(axis=(3,5))

def dense(x,w,b,activate):
    acc=np.broadcast_to(b*256,(len(x),len(b))).copy()
    for i in range(x.shape[1]):
        acc=mac(acc,x[:,i,None],w[None,:,i])
    result=output(acc)
    return np.maximum(result,0) if activate else result

def forward(images,parameters,inject_pool1=False):
    p={k:quantize(v) for k,v in parameters.items()}
    layers={}
    layers['conv1']=conv(quantize(images)[:,None],p['conv1_w'][:,None],p['conv1_b'])
    layers['pool1']=pool(layers['conv1'])
    if inject_pool1:
        layers['pool1'][:,0,0,0]=np.clip(layers['pool1'][:,0,0,0]+1024,-32768,32767)
    layers['conv2']=conv(layers['pool1'],p['conv2_w'],p['conv2_b'])
    layers['pool2']=pool(layers['conv2'])
    layers['fc1']=dense(layers['pool2'].reshape(len(images),-1),p['fc1_w'],p['fc1_b'],True)
    layers['fc2']=dense(layers['fc1'],p['fc2_w'],p['fc2_b'],True)
    layers['logits']=dense(layers['fc2'],p['fc3_w'],p['fc3_b'],False)
    return layers
