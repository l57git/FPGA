#ifndef LENET_HPP
#define LENET_HPP

#include <ap_fixed.h>
#include <ap_int.h>

namespace lenet_cfg {
enum {
    IMG_H = 28,
    IMG_W = 28,
    IMG_PIXELS = IMG_H * IMG_W,

    C1_OUT = 6,
    C1_K = 5,
    C1_H = 24,
    C1_W = 24,
    P1_H = 12,
    P1_W = 12,

    C2_OUT = 16,
    C2_K = 5,
    C2_H = 8,
    C2_W = 8,
    P2_H = 4,
    P2_W = 4,

    FC_IN = C2_OUT * P2_H * P2_W,
    FC1_OUT = 120,
    FC2_OUT = 84,
    CLASSES = 10,

    CONV1_W_COUNT = C1_OUT * C1_K * C1_K,
    CONV2_W_COUNT = C2_OUT * C1_OUT * C2_K * C2_K,
    FC1_W_COUNT = FC1_OUT * FC_IN,
    FC2_W_COUNT = FC2_OUT * FC1_OUT,
    FC3_W_COUNT = CLASSES * FC2_OUT
};
}

#ifndef LENET_DATA_W
#define LENET_DATA_W 16
#endif

#if LENET_DATA_W < 8 || LENET_DATA_W > 16
#error "LENET_DATA_W must be in the range 8..16"
#endif

typedef ap_fixed<LENET_DATA_W, 6, AP_RND, AP_SAT> data_t;
typedef ap_fixed<32, 14, AP_RND, AP_SAT> acc_t;

void lenet_accel(
    const data_t image[lenet_cfg::IMG_PIXELS],
    const data_t conv1_w[lenet_cfg::CONV1_W_COUNT],
    const data_t conv1_b[lenet_cfg::C1_OUT],
    const data_t conv2_w[lenet_cfg::CONV2_W_COUNT],
    const data_t conv2_b[lenet_cfg::C2_OUT],
    const data_t fc1_w[lenet_cfg::FC1_W_COUNT],
    const data_t fc1_b[lenet_cfg::FC1_OUT],
    const data_t fc2_w[lenet_cfg::FC2_W_COUNT],
    const data_t fc2_b[lenet_cfg::FC2_OUT],
    const data_t fc3_w[lenet_cfg::FC3_W_COUNT],
    const data_t fc3_b[lenet_cfg::CLASSES],
    data_t logits[lenet_cfg::CLASSES],
    ap_uint<4> &prediction);

#endif
