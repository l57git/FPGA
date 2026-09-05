"""Generate an auditable shared-FC variant from the unchanged baseline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'level1/src/lenet.cpp').read_text()
core = '''
static data_t dense_row(const data_t input[256], const data_t weights[256],
                        data_t bias, int inputs, bool activate) {
#pragma HLS INLINE off
    acc_t sum = bias;
DENSE_I:
    for (int i = 0; i < inputs; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=84 max=256
        sum += acc_t(input[i]) * acc_t(weights[i]);
    }
    data_t value = data_t(sum);
    return activate ? relu(value) : value;
}
'''
replacement = '''    static data_t dense_input[256];
    static data_t row_weights[256];
    static data_t dense_output[120];
    for (int i = 0; i < 256; ++i)
        dense_input[i] = pool2[i / 16][(i / 4) % 4][i % 4];
FC_STAGE:
    for (int stage = 0; stage < 3; ++stage) {
        int inputs = stage == 0 ? 256 : (stage == 1 ? 120 : 84);
        int outputs = stage == 0 ? 120 : (stage == 1 ? 84 : 10);
FC_ROW:
        for (int o = 0; o < outputs; ++o) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=120
LOAD_ROW:
            for (int i = 0; i < inputs; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=84 max=256
                int offset = o * inputs + i;
                row_weights[i] = stage == 0 ? fc1_w[offset] : (stage == 1 ? fc2_w[offset] : fc3_w[offset]);
            }
            data_t bias = stage == 0 ? fc1_b[o] : (stage == 1 ? fc2_b[o] : fc3_b[o]);
            dense_output[o] = dense_row(dense_input, row_weights, bias, inputs, stage != 2);
        }
COPY_OUT:
        for (int o = 0; o < outputs; ++o) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=120
            if (stage == 2) logits[o] = dense_output[o];
            else dense_input[o] = dense_output[o];
        }
    }

'''
source = source.replace('#include "lenet.hpp"', '#include "../../level1/src/lenet.hpp"')
source = source.replace('void lenet_accel(', core + 'void lenet_accel(', 1)
start, end = source.index('FC1_O:'), source.index('    data_t best = logits[0];')
source = source[:start] + replacement + source[end:]
source = source.replace('    static data_t fc1[FC1_OUT];\n', '').replace('    static data_t fc2[FC2_OUT];\n', '')
(ROOT / 'row_cache/src/lenet_row.cpp').write_text(source)
