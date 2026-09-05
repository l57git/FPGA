#include "../src/lenet.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace lenet_cfg;

static const int32_t BATCH_MAGIC = -20260902;

template <int N>
bool read_fixed_array(std::ifstream &in, data_t (&dst)[N]) {
    float value = 0.0f;
    for (int i = 0; i < N; ++i) {
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in) return false;
        dst[i] = value;
    }
    return true;
}

template <int N>
void zero_array(data_t (&dst)[N]) {
    for (int i = 0; i < N; ++i) dst[i] = 0;
}

int main(int argc, char **argv) {
    if (argc > 4) {
        std::cerr << "Usage: tb_lenet [accuracy_blob [result_csv [minimum_accuracy]]]\n";
        return 2;
    }

    static data_t image[IMG_PIXELS] = {};
    static data_t conv1_w[CONV1_W_COUNT] = {};
    static data_t conv1_b[C1_OUT] = {};
    static data_t conv2_w[CONV2_W_COUNT] = {};
    static data_t conv2_b[C2_OUT] = {};
    static data_t fc1_w[FC1_W_COUNT] = {};
    static data_t fc1_b[FC1_OUT] = {};
    static data_t fc2_w[FC2_W_COUNT] = {};
    static data_t fc2_b[FC2_OUT] = {};
    static data_t fc3_w[FC3_W_COUNT] = {};
    static data_t fc3_b[CLASSES] = {};
    static data_t logits[CLASSES] = {};

    if (argc >= 2) {
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) {
            std::cerr << "Cannot open " << argv[1] << '\n';
            return 2;
        }

        int32_t marker = 0;
        int32_t batch_count = 0;
        in.read(reinterpret_cast<char *>(&marker), sizeof(marker));
        in.read(reinterpret_cast<char *>(&batch_count), sizeof(batch_count));
        const bool ok = marker == BATCH_MAGIC && batch_count > 0 &&
                        read_fixed_array(in, conv1_w) &&
                        read_fixed_array(in, conv1_b) &&
                        read_fixed_array(in, conv2_w) &&
                        read_fixed_array(in, conv2_b) &&
                        read_fixed_array(in, fc1_w) &&
                        read_fixed_array(in, fc1_b) &&
                        read_fixed_array(in, fc2_w) &&
                        read_fixed_array(in, fc2_b) &&
                        read_fixed_array(in, fc3_w) &&
                        read_fixed_array(in, fc3_b);
        if (!ok) {
            std::cerr << "Bad or truncated LeNet accuracy blob header\n";
            return 2;
        }

        std::ofstream result_file;
        if (argc >= 3) {
            result_file.open(argv[2]);
            if (!result_file) {
                std::cerr << "Cannot open result CSV " << argv[2] << '\n';
                return 2;
            }
            result_file << "index,expected,prediction";
            for (int i = 0; i < CLASSES; ++i) {
                result_file << ",logit_" << i;
            }
            result_file << '\n';
        }

        int correct = 0;
        int shown_errors = 0;
        for (int sample = 0; sample < batch_count; ++sample) {
            int32_t expected = -1;
            in.read(reinterpret_cast<char *>(&expected), sizeof(expected));
            if (!in || !read_fixed_array(in, image)) {
                std::cerr << "Truncated accuracy blob at sample " << sample << '\n';
                return 2;
            }

            ap_uint<4> prediction = 0;
            lenet_accel(image,
                        conv1_w,
                        conv1_b,
                        conv2_w,
                        conv2_b,
                        fc1_w,
                        fc1_b,
                        fc2_w,
                        fc2_b,
                        fc3_w,
                        fc3_b,
                        logits,
                        prediction);

            if (sample < 5) {
                std::cout << "sample=" << sample
                          << ", expected=" << expected
                          << ", prediction=" << prediction.to_uint()
                          << "\nlogits:";
                for (int i = 0; i < CLASSES; ++i) {
                    std::cout << ' ' << logits[i].to_double();
                }
                std::cout << '\n';
            }

            if (result_file) {
                result_file << sample << ',' << expected << ','
                            << prediction.to_uint();
                result_file << std::setprecision(9);
                for (int i = 0; i < CLASSES; ++i) {
                    result_file << ',' << logits[i].to_double();
                }
                result_file << '\n';
            }

            if ((int)prediction == expected) {
                ++correct;
            } else if (shown_errors < 20) {
                std::cout << "misclassified sample=" << sample
                          << ", expected=" << expected
                          << ", prediction=" << prediction.to_uint() << '\n';
                ++shown_errors;
            }

            if ((sample + 1) % 100 == 0 || sample + 1 == batch_count) {
                std::cout << "progress=" << sample + 1 << '/' << batch_count << '\n';
            }
        }

        const double accuracy = 100.0 * correct / batch_count;
        const double minimum_accuracy = argc == 4 ? std::atof(argv[3]) : 90.0;
        std::cout << "correct=" << correct << '/' << batch_count
                  << ", fixed_point_accuracy=" << accuracy << "%\n"
                  << "BATCH COMPLETE\n";
        return accuracy >= minimum_accuracy ? 0 : 1;
    }

    zero_array(image);
    zero_array(conv1_w);
    zero_array(conv1_b);
    zero_array(conv2_w);
    zero_array(conv2_b);
    zero_array(fc1_w);
    zero_array(fc1_b);
    zero_array(fc2_w);
    zero_array(fc2_b);
    zero_array(fc3_w);
    zero_array(fc3_b);

    const int expected = 3;
    fc3_b[expected] = 1.0;
    ap_uint<4> prediction = 0;

    lenet_accel(image,
                conv1_w,
                conv1_b,
                conv2_w,
                conv2_b,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                fc3_w,
                fc3_b,
                logits,
                prediction);

    std::cout << "No blob supplied; running class-3 smoke test\n";
    std::cout << "expected=" << expected
              << ", prediction=" << prediction.to_uint() << "\nlogits:";
    for (int i = 0; i < CLASSES; ++i) {
        std::cout << ' ' << logits[i].to_double();
    }
    std::cout << '\n';

    if ((int)prediction != expected) {
        std::cerr << "FAIL\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
