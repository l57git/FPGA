"""Reproducible Level 2 development/test experiment; never changes original data."""
from pathlib import Path
import argparse, csv, hashlib, json, sys
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from level2_validation import scan_labeled_dataset, raw_resize, preprocess_image
from lenet_validation import read_lenet_blob, predict_images, sha256_file

ROOT=Path(__file__).resolve().parents[2]

def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')

def metrics(y,p):
    return {'count':len(y),'correct':int((y==p).sum()),'accuracy_percent':float(np.mean(y==p)*100),
      'per_class':{str(i):{'count':int((y==i).sum()),'correct':int(((y==i)&(y==p)).sum())} for i in range(10)}}

def audit(args):
    args.output.mkdir(parents=True,exist_ok=False)
    samples,excluded=scan_labeled_dataset(args.input)
    old,raw,rows,hashes=[],[],[],[]
    for i,s in enumerate(samples):
        with Image.open(s['path']) as im:
            pixelhash=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            # Split by decoded pixels, not file names; exact duplicates stay together.
            score=int(hashlib.sha256(('lenet-preprocessing-v1:'+pixelhash).encode()).hexdigest()[:8],16)/2**32
            split='test' if score<0.30 else 'dev'
            old.append(preprocess_image(im)); raw.append(raw_resize(im))
            rows.append({**{k:v for k,v in s.items() if k!='path'},'pixel_sha256':pixelhash,'split':split,'exif_present':bool(im.getexif())})
            hashes.append(pixelhash)
        if i%100==0: print('audit',i,flush=True)
    save_json(args.output/'manifest.json',{'input_root':str(args.input.resolve()),'samples':rows,'excluded':excluded})
    y=np.array([s['label'] for s in samples]); split=np.array([r['split'] for r in rows])
    raw=np.stack(raw);old=np.stack(old)
    np.savez_compressed(args.output/'baseline_inputs.npz',raw=raw,legacy=old,labels=y,split=split)
    params=read_lenet_blob(args.parameters).parameters
    result={'parameter_sha256':sha256_file(args.parameters),'exact_pixel_duplicates':len(hashes)-len(set(hashes)),'split_counts':{s:int((split==s).sum()) for s in ['dev','test']},'baseline':{}}
    for name,a in [('direct',raw),('legacy',old)]:
        logits,p=predict_images(a,params)
        np.savez_compressed(args.output/f'{name}_predictions.npz',prediction=p,logits=logits)
        result['baseline'][name]={s:metrics(y[m],p[m]) for s,m in [('all',np.ones(len(y),bool)),('dev',split=='dev'),('test',split=='test')]}
    save_json(args.output/'baseline_summary.json',result)
    # Before design changes, inspect exactly two development images per class by hash.
    chosen=[]
    for c in range(10):
        ids=[i for i,r in enumerate(rows) if r['label']==c and r['split']=='dev']
        chosen.extend(sorted(ids,key=lambda i:rows[i]['pixel_sha256'])[:2])
    sheet=Image.new('RGB',(1200,5*280),'white');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(chosen):
        x=(j%4)*300; yy=(j//4)*280
        with Image.open(samples[i]['path']) as im: sheet.paste(ImageOps.contain(im.convert('RGB'),(135,232)),(x,yy+25))
        sheet.paste(Image.fromarray((old[i]*255).astype('uint8')).resize((140,140)),(x+145,yy+65))
        draw.text((x+4,yy+4),f'dev index={i} label={y[i]}',(0,0,0))
    sheet.save(args.output/'development_audit.png')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--parameters',type=Path,default=ROOT/'level1/data/lenet_accuracy_1.bin');p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())
