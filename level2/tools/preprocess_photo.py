"""Apply frozen pixel-only preprocessing to one new photo, without a label."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from preprocess_robust import preprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with Image.open(args.input) as im:array,diagnostic=preprocess(im)
    np.save(args.output/'input_28x28.npy',array)
    Image.fromarray(np.round(array*255).astype('uint8')).save(args.output/'input_28x28.png')
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf8')
    print(diagnostic['status'])
    return 0 if diagnostic['status']=='ok' else 2

if __name__=='__main__':raise SystemExit(main())
