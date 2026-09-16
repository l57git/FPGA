"""Pixel-only preprocessing for a single dominant digit on a plain background.

No class labels, filenames, weights, predictions or per-image overrides enter here.
Blank/ambiguous frames are explicitly rejected instead of inventing a digit.
SciPy operates on an at-most-512px image; original files are never modified.
"""
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

@dataclass(frozen=True)
class Config:
    max_side: int = 512
    inner_size: int = 20
    center_mass: bool = True
    thicken: bool = False
    min_contrast: float = 18.0

def _otsu(a):
    h=np.bincount(np.clip(a,0,255).astype(np.uint8).ravel(),minlength=256).astype(float)
    weight=np.cumsum(h); mean=np.cumsum(h*np.arange(256)); total=weight[-1]
    variance=(mean[-1]*weight-mean*total)**2/np.maximum(weight*(total-weight),1)
    return int(np.argmax(variance[:-1]))

def _center(canvas):
    mass=canvas.sum()
    if mass<=0:return canvas
    yy,xx=np.indices(canvas.shape)
    dy=int(round(13.5-float((yy*canvas).sum()/mass)));dx=int(round(13.5-float((xx*canvas).sum()/mass)))
    result=np.zeros_like(canvas)
    y0,y1=max(0,-dy),min(28,28-dy);x0,x1=max(0,-dx),min(28,28-dx)
    result[y0+dy:y1+dy,x0+dx:x1+dx]=canvas[y0:y1,x0:x1]
    return result

def preprocess(image,config=Config(),stage='final',debug=False):
    """Return (float32[28,28], diagnostics). stage is an explicit ablation.

    contrast: robust polarity, all foreground ROI
    frame: additionally remove broad near-edge components
    component: additionally select the dominant digit, geometric centering
    final: optionally normalize thin strokes and use center of mass
    """
    if stage not in {'contrast','frame','component','final'}:raise ValueError(stage)
    if config.max_side<32 or not 4<=config.inner_size<=24:raise ValueError('invalid dimensions')
    im=ImageOps.exif_transpose(image).convert('L'); im.thumbnail((config.max_side,config.max_side),Image.Resampling.LANCZOS)
    g=np.asarray(im,dtype=np.float32);h,w=g.shape
    if min(h,w)<4:raise ValueError('image too small')
    margin_y=max(1,int(h*.08));margin_x=max(1,int(w*.08))
    interior=g[margin_y:h-margin_y,margin_x:w-margin_x]
    # Majority-background assumption. One-sided contrast avoids counting bright
    # paper as foreground when the outer frame happens to be dark.
    bg=float(np.median(interior)); dark=bg>=127.5
    ink=np.maximum(bg-g if dark else g-bg,0)
    threshold=max(config.min_contrast,float(_otsu(ink)))
    mask=ink>threshold
    labels,count=ndi.label(mask,structure=np.ones((3,3),bool));objects=ndi.find_objects(labels)
    sizes=np.bincount(labels.ravel());components=[];frame_ids=[]
    minimum=max(4,int(h*w*.00008))
    for k,sl in enumerate(objects,1):
        if sl is None or sizes[k]<minimum:continue
        y,x=sl;y0,y1=y.start,y.stop;x0,x1=x.start,x.stop
        ch,cw=y1-y0,x1-x0
        near_y=y0<.12*h or y1>.88*h; near_x=x0<.12*w or x1>.88*w
        frame=(cw>.65*w and ch<.22*h and near_y) or (ch>.65*h and cw<.22*w and near_x)
        # Very large border-connected background regions are not a single digit.
        border=x0==0 or y0==0 or x1==w or y1==h
        frame=frame or (border and sizes[k]>.35*h*w)
        if frame and stage!='contrast':frame_ids.append(k);continue
        components.append({'id':k,'area':int(sizes[k]),'bbox':[x0,y0,x1,y1]})
    info={'status':'ok','background':bg,'polarity':'dark_on_light' if dark else 'light_on_dark','threshold':threshold,'frame_components_removed':len(frame_ids),'components':components,'config':asdict(config),'stage':stage}
    def reject(reason):
        info['status']=reason
        if debug:info.update(gray=g,mask=mask,selected=np.zeros_like(mask))
        return np.zeros((28,28),np.float32),info
    if not components:return reject('no_foreground')
    components.sort(key=lambda c:c['area'],reverse=True)
    if stage in {'contrast','frame'}:chosen=[c['id'] for c in components]
    else:
        main=components[0];x0,y0,x1,y1=main['bbox'];span=max(x1-x0,y1-y0)
        if span<.05*min(h,w):return reject('foreground_too_small')
        chosen=[main['id']]
        for c in components[1:]:
            a,b,aa,bb=c['bbox'];gapx=max(0,x0-aa,a-x1);gapy=max(0,y0-bb,b-y1)
            aligned=(min(x1,aa)>max(x0,a)) or (min(y1,bb)>max(y0,b))
            if c['area']>=.04*main['area'] and max(gapx,gapy)<=.10*span and aligned:
                chosen.append(c['id'])
            elif c['area']>=.40*main['area']:
                return reject('multiple_large_objects')
    selected=np.isin(labels,chosen)
    yy,xx=np.nonzero(selected);x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1
    # Include anti-aliased stroke fringes while excluding disconnected clutter.
    support=ndi.binary_dilation(selected,iterations=1)
    peak=max(float(np.percentile(ink[selected],99)),1)
    clean=np.clip(ink/peak,0,1)*support
    pad=2;x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(w,x1+pad);y1=min(h,y1+pad)
    roi=clean[y0:y1,x0:x1];scale=config.inner_size/max(roi.shape)
    shape=(max(1,round(roi.shape[1]*scale)),max(1,round(roi.shape[0]*scale)))
    small=np.asarray(Image.fromarray(roi.astype(np.float32),mode='F').resize(shape,Image.Resampling.BILINEAR)).copy()
    canvas=np.zeros((28,28),np.float32);oy=(28-small.shape[0])//2;ox=(28-small.shape[1])//2
    canvas[oy:oy+small.shape[0],ox:ox+small.shape[1]]=small
    if stage=='final' and config.thicken:
        # One global, class-independent 2x2 grayscale dilation candidate. Never
        # selected by confidence, filename or true label on an individual image.
        canvas=ndi.maximum_filter(canvas,size=2,mode='constant',cval=0)
    if stage=='final' and config.center_mass:canvas=_center(canvas)
    info.update(bbox=[x0,y0,x1,y1],selected_components=len(chosen),foreground_area=int(selected.sum()))
    if debug:info.update(gray=g,mask=mask,selected=selected)
    return np.clip(canvas,0,1).astype(np.float32),info
