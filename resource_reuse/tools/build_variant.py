"""Generate an auditable shared-FC variant from the unchanged baseline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'level1/src/lenet.cpp').read_text()
core = '''
// All three FC stages use these same buffers and this single function instance.
static void dense_shared(const data_t input[256], const data_t weights[30720],
                         const data_t biases[120], data_t output[120],
                         int inputs, int outputs, bool activate) {
#pragma HLS INLINE off
DENSE_O:
    for (int o = 0; o < outputs; ++o) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=120
        acc_t sum = biases[o];
DENSE_I:
        for (int i = 0; i < inputs; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=84 max=256
            sum += acc_t(input[i]) * acc_t(weights[o * inputs + i]);
        }
        data_t value = data_t(sum);
        output[o] = activate ? relu(value) : value;
    }
}

'''
replacement = '''    static data_t dense_input[256];
    static data_t dense_weights[30720];
    static data_t dense_biases[120];
    static data_t dense_output[120];
    for (int i = 0; i < 256; ++i)
        dense_input[i] = pool2[i / 16][(i / 4) % 4][i % 4];
FC_STAGE:
    for (int stage = 0; stage < 3; ++stage) {
        int inputs = stage == 0 ? 256 : (stage == 1 ? 120 : 84);
        int outputs = stage == 0 ? 120 : (stage == 1 ? 84 : 10);
LOAD_W:
        for (int i = 0; i < inputs * outputs; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=840 max=30720
            dense_weights[i] = stage == 0 ? fc1_w[i] : (stage == 1 ? fc2_w[i] : fc3_w[i]);
        }
LOAD_B:
        for (int o = 0; o < outputs; ++o) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=120
            dense_biases[o] = stage == 0 ? fc1_b[o] : (stage == 1 ? fc2_b[o] : fc3_b[o]);
        }
        dense_shared(dense_input, dense_weights, dense_biases, dense_output,
                     inputs, outputs, stage != 2);
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
(ROOT / 'resource_reuse/src/lenet_shared.cpp').write_text(source)
