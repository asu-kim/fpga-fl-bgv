#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <fstream>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t;

#define CONV1_WEIGHT_BIN "QuantLeNet5__conv1_weight.bin"
#define CONV1_BIAS_BIN "QuantLeNet5__conv1_bias.bin"
#define CONV2_WEIGHT_BIN "QuantLeNet5__conv2_weight.bin"
#define CONV2_BIAS_BIN "QuantLeNet5__conv2_bias.bin"
#define FC1_WEIGHT_BIN "QuantLeNet5__fc1_weight.bin"
#define FC1_BIAS_BIN "QuantLeNet5__fc1_bias.bin"
#define FC2_WEIGHT_BIN "QuantLeNet5__fc2_weight.bin"
#define FC2_BIAS_BIN "QuantLeNet5__fc2_bias.bin"
#define FC3_WEIGHT_BIN "QuantLeNet5__fc3_weight.bin"
#define FC3_BIAS_BIN "QuantLeNet5__fc3_bias.bin"
#define DATASET "t10k-images-idx3-ubyte"


template<int C, int KERNEL_SIZE>
void load_conv2d_weight(
        const char* filename,
        data_t weight[C][KERNEL_SIZE][KERNEL_SIZE]
        ) {
    std::ifstream file(filename, std::ios::binary);

    for(int channel=0; channel<C; ++channel) {
        for(int i=0; i<KERNEL_SIZE; ++i) {
            for(int j=0; j<KERNEL_SIZE; ++j) {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(float));
                weight[channel][i][j] = data_t(val);
            }
        }
    }
}

template<int OUT_DIM, int IN_DIM>
void load_fc_weigts(
        const char* filename,
        data_t fc_weight[OUT_DIM][IN_DIM]
        ) {
    std::ifstream file(filename, std::ios::binary);

    for(int o=0; o<OUT_DIM; ++o) {
        for(int i=0; i<IN_DIM; ++i) {
            float val;
            file.read(reinterpret_cast<char*>(&val), sizeof(float));
            fc_weight[o][i] = data_t(val);
        }
    }
}

template<int SIZE>
void load_bias(
        const char* filename,
        data_t bias[SIZE]
        ) {
    std::ifstream file(filename, std::ios::binary);

    for(int i=0; i<SIZE; ++i) {
        float val;
        file.read(reinterpret_cast<char*>(&val), sizeof(float));
        bias[i] = data_t(val);
    }
}

template<int IN_DIM>
void load_input_to_stream(
        const char* filename,
        hls::stream<data_t>& in_stream
        ) {
    std::ifstream file(filename, std::ios::binary);

    // skip MNIST headers (magic, num_images, rows, cols)
    file.seekg(16, std::ios::beg);

    for(int i=0; i<IN_DIM * IN_DIM; ++i) {
        unsigned char pixel;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        data_t val = data_t(pixel);
        val /= data_t(255.0);
        in_stream.write(val);
    }
}

#endif
