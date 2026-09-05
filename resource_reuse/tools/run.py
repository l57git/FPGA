"""Compare baseline and shared FC at identical width/tool/clock on fresh outputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'level1/tools'))
from lenet_validation import read_lenet_blob, write_lenet_blob

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blob', type=Path, required=True)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--width', type=int, choices=[8,16], default=16)
    p.add_argument('--hls-root', type=Path, default=Path('E:/use/cpu/Vivado/2019.2'))
    a = p.parse_args()
    output = ROOT / 'resource_reuse/results'
    blob = read_lenet_blob(a.blob)
    if not 1 <= a.samples <= blob.sample_count: raise ValueError('invalid sample count')
    batch = output / 'input.bin'
    write_lenet_blob(batch, blob.parameters, blob.labels[:a.samples], blob.images[:a.samples])
    rows = []
    for name, src in [('baseline', ROOT / 'level1/src/lenet.cpp'), ('shared', ROOT / 'resource_reuse/src/lenet_shared.cpp')]:
        case = output / name
        case.mkdir(parents=True, exist_ok=True)
        if (case / 'predictions.csv').exists(): raise ValueError('Output exists; archive the prior experiment first')
        workspace = ROOT / 'resource_reuse/hls_work' / name
        workspace.mkdir(parents=True, exist_ok=True)
        tcl = workspace / 'run.tcl'
        tcl.write_text(f'''cd {{{workspace.as_posix()}}}
open_project -reset design
set_top lenet_accel
add_files {{{src.as_posix()}}} -cflags "-DLENET_DATA_W={a.width}"
add_files -tb {{{(ROOT / 'level1/tb/tb_lenet.cpp').as_posix()}}} -cflags "-DLENET_DATA_W={a.width}"
open_solution -reset solution1
set_part {{xc7z020clg400-1}}
create_clock -period 10 -name default
csim_design -O -argv "{batch.as_posix()} {(case / 'predictions.csv').as_posix()} 0"
csynth_design
exit
''')
        launcher = workspace / 'run.bat'
        hls = str(a.hls_root)
        launcher.write_text(f'@echo off\ncall "{hls}/settings64.bat"\nset "PATH={hls}/tps/win64/msys64/mingw64/bin;{hls}/tps/mingw/6.2.0/win64.o/nt/bin;%PATH%"\ncall "{hls}/bin/vivado_hls.bat" -f "{tcl}"\n')
        print('START '+name, flush=True)
        with (case / 'run.log').open('w') as log:
            process = subprocess.run(['cmd.exe','/d','/c',str(launcher)],cwd=workspace,stdout=log,stderr=subprocess.STDOUT)
        xml = workspace / 'design/solution1/syn/report/lenet_accel_csynth.xml'
        if not xml.exists() or 'CSim done with 0 errors' not in (case/'run.log').read_text():
            raise RuntimeError(f'{name} failed; inspect {case}/run.log')
        tree = ET.parse(xml)
        row = {'variant':name,'width':a.width,'samples':a.samples,'tool':'Vivado HLS 2019.2','target_period_ns':10,
               'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
               'input_sha256':hashlib.sha256(batch.read_bytes()).hexdigest()}
        row['header_sha256'] = hashlib.sha256((ROOT/'level1/src/lenet.hpp').read_bytes()).hexdigest()
        row['testbench_sha256'] = hashlib.sha256((ROOT/'level1/tb/tb_lenet.cpp').read_bytes()).hexdigest()
        for key, xpath in {'LUT':'.//Resources/LUT','FF':'.//Resources/FF','DSP':'.//Resources/DSP48E','BRAM':'.//Resources/BRAM_18K','cycles_min':'.//SummaryOfOverallLatency/Best-caseLatency','cycles_max':'.//SummaryOfOverallLatency/Worst-caseLatency','estimated_clock_ns':'.//SummaryOfTimingAnalysis/EstimatedClockPeriod'}.items():
            row[key]=tree.findtext(xpath)
        with (case/'predictions.csv').open() as f: predictions=list(csv.DictReader(f))
        if len(predictions)!=a.samples: raise RuntimeError('incomplete CSV')
        row['correct']=sum(r['expected']==r['prediction'] for r in predictions)
        row['accuracy_percent']=100*row['correct']/a.samples
        for f in xml.parent.glob('*'):
            if f.suffix in ['.rpt','.xml']: shutil.copy2(f,case/f.name)
        rtl = workspace / 'design/solution1/syn/verilog'
        (case/'rtl').mkdir(exist_ok=True)
        for f in rtl.glob('*.v'): shutil.copy2(f,case/'rtl'/f.name)
        rows.append(row)
        (output/'summary.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True)
    with (output/'baseline/predictions.csv').open() as f: base=list(csv.DictReader(f))
    with (output/'shared/predictions.csv').open() as f: shared=list(csv.DictReader(f))
    comparison={'samples':a.samples,'prediction_matches':sum(x['prediction']==y['prediction'] for x,y in zip(base,shared)),
                'all_logits_equal':all(all(x[f'logit_{i}']==y[f'logit_{i}'] for i in range(10)) for x,y in zip(base,shared))}
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2))
    if comparison['prediction_matches']!=a.samples or not comparison['all_logits_equal']: raise RuntimeError('numeric mismatch')
    print(comparison)

if __name__=='__main__': main()
