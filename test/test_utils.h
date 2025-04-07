#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <unistd.h>
#include <limits.h>

#include <fstream>
#include <ap_fixed.h>
#include <filesystem>
#include <string>

typedef ap_fixed<8, 3> data_t;

// base path
const std::string BASE_PATH="/home/megan/Vitis-AI/accelerator/matmul";
const std::string MODEL_DIR = BASE_PATH + "/models/quant_result_test/deploy_check_data_int/QuantLeNet5/";
const std::string DATASET_PATH = BASE_PATH + "/models/data/MNIST/raw/t10k-images-idx3-ubyte";
const std::string VALIDATE_DATA_PATH = BASE_PATH + "/models/data/MNIST/raw/t10k-labels-idx1-ubyte";

#define CONV1_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__conv1_weight.bin").c_str()
#define CONV1_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__conv1_bias.bin").c_str()
#define CONV2_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__conv2_weight.bin").c_str()
#define CONV2_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__conv2_bias.bin").c_str()
#define FC1_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__fc1_weight.bin").c_str()
#define FC1_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__fc1_bias.bin").c_str()
#define FC2_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__fc2_weight.bin").c_str()
#define FC2_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__fc2_bias.bin").c_str()
#define FC3_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__fc3_weight.bin").c_str()
#define FC3_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__fc3_bias.bin").c_str()
#define DATASET DATASET_PATH.c_str()
#define VALIDATE_DATA VALIDATE_DATA_PATH.c_str() 


template<int OC, int IC, int KERNEL_SIZE>
void load_conv2d_weight(
        const char* filename,
        data_t weight[OC][IC][KERNEL_SIZE][KERNEL_SIZE]
        ) {
    std::ifstream file(filename, std::ios::binary);

    if(!file.is_open()) {
        std::cerr << "Error: cannot open " << CONV1_WEIGHT_BIN << std::endl;
        exit(1);
    }

    for(int oc=0; oc<OC; ++oc) {
        for(int ic=0; ic<IC; ++ic) {
            for(int i=0; i<KERNEL_SIZE; ++i) {
                for(int j=0; j<KERNEL_SIZE; ++j) {
                    float val;
                    file.read(reinterpret_cast<char*>(&val), sizeof(float));
                    weight[oc][ic][i][j] = data_t(val);
                }
            }
        }
    }
}

template<int OUT_DIM, int IN_DIM>
void load_fc_weight(
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

int check_against_golden(int predicted) {
    std::ifstream file(VALIDATE_DATA, std::ios::binary);

    /*
     * for testing multiple input
    if(!file.is_open()) {
        std::cerr << "Error opening labels" << std::endl;
        return 1;
    }
    * end loop
    */

    file.seekg(8, std::ios::beg);

    unsigned char true_label;
    file.read(reinterpret_cast<char*>(&true_label), 1);

    int golden_class = static_cast<int>(true_label);

    if(predicted != golden_class) {
        std::cout << "ERROR: Predicted: " << predicted
            << ", Golden: " << golden_class << std::endl;

        return 1;
    }

    std::cout << "SUCCESS: Predict class: " << golden_class << std::endl;
    return 0;
}
#endif
