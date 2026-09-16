"""Development-only selection, then frozen held-out evaluation and HLS export."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from preprocess_robust import preprocess,Config
from preprocessing_experiment import metrics,save_json,ROOT
from lenet_validation import predict_images,read_lenet_blob,write_lenet_blob,write_result_csv,sha256_file

CASES={'contrast':('contrast',Config()),'frame':('frame',Config()),'component':('component',Config()),'centered':('final',Config()),'thickened':('final',Config(thicken=True))}

def load(out):
    manifest=json.loads((out/'manifest.json').read_text('utf8'))
    return manifest,manifest['samples'],Path(manifest['input_root'])

def evaluate(args):
    manifest,rows,data=load(args.output)
    ids=[i for i,r in enumerate(rows) if r['split']=='dev'] if args.phase=='dev' else list(range(len(rows)))
    if args.phase=='dev':
        cases=CASES
        if (args.output/'selection.json').exists():raise ValueError('selection already frozen; use a new experiment directory')
    else:
        locked=json.loads((args.output/'selection.json').read_text('utf8'))
        if sha256_file(Path(__file__).with_name('preprocess_robust.py'))!=locked['preprocessor_sha256']:raise ValueError('preprocessor changed after freeze')
        if sha256_file(args.parameters)!=locked['parameter_sha256']:raise ValueError('model changed after freeze')
        if sha256_file(args.output/'manifest.json')!=locked['manifest_sha256']:raise ValueError('manifest changed after freeze')
        for r in rows:
            if sha256_file(data/r['relative_path'])!=r['sha256']:raise ValueError('input changed after freeze: '+r['relative_path'])
        if asdict(CASES[locked['selected']][1])!=locked['config']:raise ValueError('configuration changed after freeze')
        cases={locked['selected']:CASES[locked['selected']]}
        if (args.output/'final_summary.json').exists():raise ValueError('test already evaluated; do not tune and rerun')
    params=read_lenet_blob(args.parameters).parameters;y=np.array([rows[i]['label'] for i in ids]);result={}; arrays={};preds={};diags={}
    for name,(stage,config) in cases.items():
        a=[];details=[]
        for j,i in enumerate(ids):
            with Image.open(data/rows[i]['relative_path']) as im: v,d=preprocess(im,config,stage)
            a.append(v);details.append(d)
        a=np.stack(a);logits,p=predict_images(a,params)
        rejected=np.array([d['status']!='ok' for d in details]);p[rejected]=-1
        result[name]=metrics(y,p);result[name]['rejected']=int(rejected.sum())
        arrays[name]=a;preds[name]=p;diags[name]=details
        print(args.phase,name,result[name]['correct'],len(y),result[name]['accuracy_percent'],'rejected',rejected.sum(),flush=True)
        np.savez_compressed(args.output/f'{args.phase}_{name}.npz',indices=ids,images=a,predictions=p,logits=logits)
    if args.phase=='dev':
        # Only the final two formulations are deployment candidates. Earlier
        # stages diagnose causes, not opportunities for sample-wise selection.
        selected=max(['centered','thickened'],key=lambda n:result[n]['correct'])
        save_json(args.output/'development_results.json',result)
        save_json(args.output/'selection.json',{'selected':selected,'rule':'higher DEV correct count; tie selects centered (no dilation)','config':asdict(CASES[selected][1]),'preprocessor_sha256':sha256_file(Path(__file__).with_name('preprocess_robust.py')),'parameter_sha256':sha256_file(args.parameters),'manifest_sha256':sha256_file(args.output/'manifest.json'),'test_used_for_selection':False})
        chosen=[]
        for c in range(10):chosen+=sorted([j for j,i in enumerate(ids) if rows[i]['label']==c],key=lambda j:rows[ids[j]]['pixel_sha256'])[:2]
        sheet=Image.new('RGB',(1440,5*270),'white');draw=ImageDraw.Draw(sheet)
        for k,j in enumerate(chosen):
            i=ids[j];x=k%4*360;yy=k//4*270
            with Image.open(data/rows[i]['relative_path']) as im:sheet.paste(ImageOps.contain(im.convert('RGB'),(135,230)),(x,yy+25))
            for z,name in enumerate(['centered','thickened']):
                img=Image.fromarray((arrays[name][j]*255).astype('uint8')).resize((100,100))
                sheet.paste(img,(x+142+z*106,yy+80));draw.text((x+142+z*106,yy+188),f'{name[:5]} p={preds[name][j]}',fill='black')
            draw.text((x+2,yy+5),f'dev index={i} label={y[j]}',fill='black')
        sheet.save(args.output/'development_candidates.png')
    else:
        name=next(iter(cases));p=preds[name];a=arrays[name];details=diags[name]
        splits=np.array([r['split'] for r in rows]);summary={'selected':name,'python_float':{s:metrics(y[m],p[m]) for s,m in [('all',np.ones(len(y),bool)),('dev',splits=='dev'),('test',splits=='test')]},'rejections':sum(d['status']!='ok' for d in details),'hls':'NOT_RUN','background':'no independent background images','source_status':'pending_confirmation','limitations':['Image-level split, not writer/session-held-out','This dataset had already been evaluated by the project; not an external blind benchmark','Exact pixel duplicates grouped; near-duplicate acquisition groups unavailable']}
        sys.path.insert(0,str(ROOT/'layer_validation/tools'))
        from reference import forward
        q=[]
        for start in range(0,len(a),64):
            q.append(forward(a[start:start+64],params)['logits']);print('fixed reference',min(start+64,len(a)),flush=True)
        q=np.concatenate(q);qp=q.argmax(axis=1);qp[np.array([d['status']!='ok' for d in details])]=-1
        summary['python_integer_reference_w16']={s:metrics(y[m],qp[m]) for s,m in [('all',np.ones(len(y),bool)),('dev',splits=='dev'),('test',splits=='test')]}
        summary['float_integer_prediction_matches']=int((p==qp).sum())
        accepted=np.array([d['status']=='ok' for d in details]);active=np.flatnonzero(accepted)
        write_lenet_blob(args.output/'preprocessed_accepted.bin',params,y[active],a[active])
        roundtrip=read_lenet_blob(args.output/'preprocessed_accepted.bin',expected_count=len(active))
        assert np.array_equal(roundtrip.images,a[active]) and np.array_equal(roundtrip.labels,y[active])
        summary['hls_export']={'count':len(active),'rejected_excluded_from_hls_blob_only':int((~accepted).sum()),'sha256':sha256_file(args.output/'preprocessed_accepted.bin'),'roundtrip':'PASS','note':'end-to-end accuracy always includes rejected digits as errors'}
        write_result_csv(args.output/'python_results.csv',y[active],p[active],logits[active])
        write_result_csv(args.output/'integer_reference_results.csv',y[active],qp[active],q[active]/1024.)
        with (args.output/'predictions.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['index','file','split','label','float_prediction','integer_prediction','status','bbox'])
            for i,r in enumerate(rows):w.writerow([i,r['relative_path'],r['split'],r['label'],p[i],qp[i],details[i]['status'],details[i].get('bbox')])
        with (args.output/'hls_input_mapping.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['blob_index','dataset_index','relative_path','split'])
            for j,i in enumerate(active):w.writerow([j,i,rows[i]['relative_path'],rows[i]['split']])
        save_json(args.output/'diagnostics.json',details);save_json(args.output/'final_summary.json',summary)
        print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dev','final']);p.add_argument('--output',type=Path,default=ROOT/'level2/results/preprocessing_v1');p.add_argument('--parameters',type=Path,default=ROOT/'level1/data/lenet_accuracy_1.bin');evaluate(p.parse_args())
