"""Run clean/fault-injected C simulations on two examples of every class."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'level1/tools'))
from lenet_validation import read_lenet_blob, write_lenet_blob

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--blob',type=Path,required=True)
    parser.add_argument('--hls-root',type=Path,default=Path('E:/use/cpu/Vivado/2019.2'))
    args=parser.parse_args()
    if not (args.hls_root/'bin/vivado_hls.bat').is_file():
        raise FileNotFoundError('HLS not installed at --hls-root; restore the drive or specify its current path')
    out=ROOT/'layer_validation/results'
    if (out/'clean').exists() or (out/'fault').exists():
        raise ValueError('Archive previous case directories before rerunning')
    blob=read_lenet_blob(args.blob)
    indices=[]
    for label in range(10):
        candidates=np.flatnonzero(blob.labels==label)
        if len(candidates)<2: raise ValueError('Two examples per class required')
        indices.extend(candidates[:2].tolist())
    batch=out/'input.bin'
    write_lenet_blob(batch,blob.parameters,blob.labels[indices],blob.images[indices])
    src=ROOT/'layer_validation/src/lenet_trace.cpp'
    tb=ROOT/'level1/tb/tb_lenet.cpp'
    metadata={'source_indices':indices,'labels':blob.labels[indices].tolist(),
              'input_sha256':hashlib.sha256(batch.read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
              'header_sha256':hashlib.sha256((ROOT/'level1/src/lenet.hpp').read_bytes()).hexdigest(),
              'testbench_sha256':hashlib.sha256(tb.read_bytes()).hexdigest(),
              'tool':'Vivado HLS 2019.2','samples':len(indices)}
    (out/'input_metadata.json').write_text(json.dumps(metadata,indent=2))
    for name,fault in [('clean',0),('fault',1)]:
        case=out/name;case.mkdir()
        workspace=ROOT/'layer_validation/hls_work'/name;workspace.mkdir(parents=True,exist_ok=True)
        tcl=workspace/'run.tcl'
        tcl.write_text(f'''cd {{{workspace.as_posix()}}}
open_project -reset design
set_top lenet_accel
add_files {{{src.as_posix()}}} -cflags "-DLENET_DATA_W=16 -DTRACE_FAULT={fault}"
add_files -tb {{{tb.as_posix()}}} -cflags "-DLENET_DATA_W=16"
open_solution -reset solution1
set_part {{xc7z020clg400-1}}
create_clock -period 10 -name default
csim_design -O -argv "{batch.as_posix()} {(case/'predictions.csv').as_posix()} 0"
exit
''')
        launcher=workspace/'run.bat'
        hls=str(args.hls_root)
        launcher.write_text(f'@echo off\ncall "{hls}/settings64.bat"\nset "PATH={hls}/tps/win64/msys64/mingw64/bin;{hls}/tps/mingw/6.2.0/win64.o/nt/bin;%PATH%"\nset "LENET_TRACE_FILE={case / "trace.csv"}"\ncall "{hls}/bin/vivado_hls.bat" -f "{tcl}"\n')
        print('START '+name,flush=True)
        with (case/'run.log').open('w') as log:
            p=subprocess.run(['cmd.exe','/d','/c',str(launcher)],cwd=workspace,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode or 'CSim done with 0 errors' not in (case/'run.log').read_text():
            raise RuntimeError('Simulation failed: '+name)
        raw=case/'trace.csv'
        if not raw.exists() or raw.stat().st_size==0: raise RuntimeError('Missing trace')
        (case/'trace.csv.gz').write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
        print('DONE '+name,flush=True)

if __name__=='__main__': main()
