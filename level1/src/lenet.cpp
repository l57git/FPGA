#include "lenet.hpp"

using namespace lenet_cfg;

static data_t relu(data_t x) {
    return x > data_t(0) ? x : data_t(0);
}

static data_t max4(data_t a, data_t b, data_t c, data_t d) {
    data_t m0 = a > b ? a : b;
    data_t m1 = c > d ? c : d;
    return m0 > m1 ? m0 : m1;
}

void lenet_accel(
    const data_t image[IMG_PIXELS],
    const data_t conv1_w[CONV1_W_COUNT],
    const data_t conv1_b[C1_OUT],
    const data_t conv2_w[CONV2_W_COUNT],
    const data_t conv2_b[C2_OUT],
    const data_t fc1_w[FC1_W_COUNT],
    const data_t fc1_b[FC1_OUT],
    const data_t fc2_w[FC2_W_COUNT],
    const data_t fc2_b[FC2_OUT],
    const data_t fc3_w[FC3_W_COUNT],
    const data_t fc3_b[CLASSES],
    data_t logits[CLASSES],
    ap_uint<4> &prediction) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0 depth=784
#pragma HLS INTERFACE m_axi port=conv1_w offset=slave bundle=gmem1 depth=150
#pragma HLS INTERFACE m_axi port=conv1_b offset=slave bundle=gmem1 depth=6
#pragma HLS INTERFACE m_axi port=conv2_w offset=slave bundle=gmem1 depth=2400
#pragma HLS INTERFACE m_axi port=conv2_b offset=slave bundle=gmem1 depth=16
#pragma HLS INTERFACE m_axi port=fc1_w offset=slave bundle=gmem1 depth=30720
#pragma HLS INTERFACE m_axi port=fc1_b offset=slave bundle=gmem1 depth=120
#pragma HLS INTERFACE m_axi port=fc2_w offset=slave bundle=gmem1 depth=10080
#pragma HLS INTERFACE m_axi port=fc2_b offset=slave bundle=gmem1 depth=84
#pragma HLS INTERFACE m_axi port=fc3_w offset=slave bundle=gmem1 depth=840
#pragma HLS INTERFACE m_axi port=fc3_b offset=slave bundle=gmem1 depth=10
#pragma HLS INTERFACE m_axi port=logits offset=slave bundle=gmem0 depth=10
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_w bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_b bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_w bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_b bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_w bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_b bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_w bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_b bundle=control
#pragma HLS INTERFACE s_axilite port=fc3_w bundle=control
#pragma HLS INTERFACE s_axilite port=fc3_b bundle=control
#pragma HLS INTERFACE s_axilite port=logits bundle=control
#pragma HLS INTERFACE s_axilite port=prediction bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static data_t conv1[C1_OUT][C1_H][C1_W];
    static data_t pool1[C1_OUT][P1_H][P1_W];
    static data_t conv2[C2_OUT][C2_H][C2_W];
    static data_t pool2[C2_OUT][P2_H][P2_W];
    static data_t fc1[FC1_OUT];
    static data_t fc2[FC2_OUT];

CONV1_OC:
    for (int oc = 0; oc < C1_OUT; ++oc) {
    CONV1_Y:
        for (int y = 0; y < C1_H; ++y) {
        CONV1_X:
            for (int x = 0; x < C1_W; ++x) {
                acc_t sum = conv1_b[oc];
            CONV1_KY:
                for (int ky = 0; ky < C1_K; ++ky) {
                CONV1_KX:
                    for (int kx = 0; kx < C1_K; ++kx) {
                        const int img_idx = (y + ky) * IMG_W + (x + kx);
                        const int w_idx = (oc * C1_K + ky) * C1_K + kx;
                        sum += acc_t(image[img_idx]) * acc_t(conv1_w[w_idx]);
                    }
                }
                conv1[oc][y][x] = relu(data_t(sum));
            }
        }
    }

POOL1_OC:
    for (int oc = 0; oc < C1_OUT; ++oc) {
    POOL1_Y:
        for (int y = 0; y < P1_H; ++y) {
        POOL1_X:
            for (int x = 0; x < P1_W; ++x) {
                pool1[oc][y][x] = max4(
                    conv1[oc][2 * y][2 * x],
                    conv1[oc][2 * y][2 * x + 1],
                    conv1[oc][2 * y + 1][2 * x],
                    conv1[oc][2 * y + 1][2 * x + 1]);
            }
        }
    }

CONV2_OC:
    for (int oc = 0; oc < C2_OUT; ++oc) {
    CONV2_Y:
        for (int y = 0; y < C2_H; ++y) {
        CONV2_X:
            for (int x = 0; x < C2_W; ++x) {
                acc_t sum = conv2_b[oc];
            CONV2_IC:
                for (int ic = 0; ic < C1_OUT; ++ic) {
                CONV2_KY:
                    for (int ky = 0; ky < C2_K; ++ky) {
                    CONV2_KX:
                        for (int kx = 0; kx < C2_K; ++kx) {
                            const int in_idx_y = y + ky;
                            const int in_idx_x = x + kx;
                            const int w_idx =
                                (((oc * C1_OUT + ic) * C2_K + ky) * C2_K + kx);
                            sum += acc_t(pool1[ic][in_idx_y][in_idx_x]) *
                                   acc_t(conv2_w[w_idx]);
                        }
                    }
                }
                conv2[oc][y][x] = relu(data_t(sum));
            }
        }
    }

POOL2_OC:
    for (int oc = 0; oc < C2_OUT; ++oc) {
    POOL2_Y:
        for (int y = 0; y < P2_H; ++y) {
        POOL2_X:
            for (int x = 0; x < P2_W; ++x) {
                pool2[oc][y][x] = max4(
                    conv2[oc][2 * y][2 * x],
                    conv2[oc][2 * y][2 * x + 1],
                    conv2[oc][2 * y + 1][2 * x],
                    conv2[oc][2 * y + 1][2 * x + 1]);
            }
        }
    }

FC1_O:
    for (int o = 0; o < FC1_OUT; ++o) {
        acc_t sum = fc1_b[o];
    FC1_C:
        for (int c = 0; c < C2_OUT; ++c) {
        FC1_Y:
            for (int y = 0; y < P2_H; ++y) {
            FC1_X:
                for (int x = 0; x < P2_W; ++x) {
                    const int in_idx = (c * P2_H + y) * P2_W + x;
                    const int w_idx = o * FC_IN + in_idx;
                    sum += acc_t(pool2[c][y][x]) * acc_t(fc1_w[w_idx]);
                }
            }
        }
        fc1[o] = relu(data_t(sum));
    }

FC2_O:
    for (int o = 0; o < FC2_OUT; ++o) {
        acc_t sum = fc2_b[o];
    FC2_I:
        for (int i = 0; i < FC1_OUT; ++i) {
            const int w_idx = o * FC1_OUT + i;
            sum += acc_t(fc1[i]) * acc_t(fc2_w[w_idx]);
        }
        fc2[o] = relu(data_t(sum));
    }

FC3_O:
    for (int o = 0; o < CLASSES; ++o) {
        acc_t sum = fc3_b[o];
    FC3_I:
        for (int i = 0; i < FC2_OUT; ++i) {
            const int w_idx = o * FC2_OUT + i;
            sum += acc_t(fc2[i]) * acc_t(fc3_w[w_idx]);
        }
        logits[o] = data_t(sum);
    }

    data_t best = logits[0];
    ap_uint<4> best_idx = 0;
ARGMAX:
    for (int i = 1; i < CLASSES; ++i) {
        if (logits[i] > best) {
            best = logits[i];
            best_idx = i;
        }
    }
    prediction = best_idx;
}

