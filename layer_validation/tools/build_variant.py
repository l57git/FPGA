"""Add simulation-only observation points to the validated row-cache design."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
source = (ROOT/'row_cache/src/lenet_row.cpp').read_text()
helper = '''
#ifndef __SYNTHESIS__
#include <cstdlib>
#include <fstream>
static int trace_sample = 0;
static std::ofstream &trace_file() {
    static std::ofstream out(std::getenv("LENET_TRACE_FILE"));
    return out;
}
static void trace_value(const char *layer, int index, data_t value) {
    ap_int<16> raw;
    raw.range(15,0) = value.range(15,0);
    trace_file() << trace_sample << ',' << layer << ',' << index << ',' << raw.to_int() << '\\n';
}
template<int C, int H, int W>
static void trace3(const char *layer, const data_t (&values)[C][H][W]) {
    for (int c=0;c<C;++c) for(int y=0;y<H;++y) for(int x=0;x<W;++x)
        trace_value(layer,(c*H+y)*W+x,values[c][y][x]);
}
static void trace1(const char *layer, const data_t *values, int count) {
    for(int i=0;i<count;++i) trace_value(layer,i,values[i]);
}
#endif
'''
source = source.replace('using namespace lenet_cfg;', 'using namespace lenet_cfg;\n'+helper)
for before, code in [
    ('POOL1_OC:', 'trace3("conv1",conv1);'),
    ('CONV2_OC:', '''#if TRACE_FAULT
    pool1[0][0][0] += data_t(1);
#endif
    trace3("pool1",pool1);'''),
    ('POOL2_OC:', 'trace3("conv2",conv2);'),
    ('    static data_t dense_input[256];', 'trace3("pool2",pool2);'),
    ('COPY_OUT:', 'trace1(stage==0 ? "fc1" : (stage==1 ? "fc2" : "logits"),dense_output,outputs);'),
]:
    assert source.count(before)==1, before
    source = source.replace(before, '#ifndef __SYNTHESIS__\n    '+code+'\n#endif\n'+before)
source = source.replace('    prediction = best_idx;', '    prediction = best_idx;\n#ifndef __SYNTHESIS__\n    ++trace_sample;\n#endif')
(ROOT/'layer_validation/src/lenet_trace.cpp').write_text(source)
