"""Generate mixed weight/activation types without changing operation order."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'mixed_precision'
header = (ROOT / 'level1/src/lenet.hpp').read_text()
header = header.replace('typedef ap_fixed<32, 14, AP_RND, AP_SAT> acc_t;', '''typedef ap_fixed<32, 14, AP_RND, AP_SAT> acc_t;
#ifndef WEIGHT_W
#define WEIGHT_W 8
#endif
#ifndef WEIGHT_I
#define WEIGHT_I 1
#endif
#ifndef FC_PIPELINE
#define FC_PIPELINE 0
#endif
typedef ap_fixed<WEIGHT_W, WEIGHT_I, AP_RND, AP_SAT> weight_t;''')
source = (ROOT / 'row_cache/src/lenet_row.cpp').read_text()
source = source.replace('#include "../../level1/src/lenet.hpp"', '#include "lenet.hpp"')
source = source.replace('const data_t weights[256]', 'const weight_t weights[256]')
source = source.replace('static data_t row_weights[256]', 'static weight_t row_weights[256]')
source = source.replace('DENSE_I:\n    for (int i = 0; i < inputs; ++i) {', '''DENSE_I:
    for (int i = 0; i < inputs; ++i) {
#if FC_PIPELINE
#pragma HLS PIPELINE II=1
#endif''')
tb = (ROOT / 'level1/tb/tb_lenet.cpp').read_text()
tb = tb.replace('template <int N>', 'template <typename T, int N>')
tb = tb.replace('data_t (&dst)[N]', 'T (&dst)[N]')
for name in ['conv1_w', 'conv2_w', 'fc1_w', 'fc2_w', 'fc3_w']:
    header = header.replace('const data_t ' + name, 'const weight_t ' + name)
    source = source.replace('const data_t ' + name, 'const weight_t ' + name)
    tb = tb.replace('static data_t ' + name, 'static weight_t ' + name)
for name, text in [('src/lenet.hpp', header), ('src/lenet_mixed.cpp', source), ('tb/tb_lenet.cpp', tb)]:
    (out / name).write_text(text)
split = source
for name in ['conv1_b', 'conv2_b', 'fc1_b', 'fc2_b', 'fc3_b']:
    split = split.replace('port=' + name + ' offset=slave bundle=gmem1',
                          'port=' + name + ' offset=slave bundle=gmem0')
(out / 'src/lenet_split.cpp').write_text(split)
