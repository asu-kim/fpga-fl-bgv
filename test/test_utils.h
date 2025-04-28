#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <unistd.h>
#include <limits.h>

#include <fstream>
#include <filesystem>
#include <string>

#include "data_type.hpp"

// typedef ap_fixed<8, 3> data_t;
// typedef ap_int<32> data_t;

// base path
// const std::string BASE_PATH="/home/megan/Vitis-AI/accelerator/matmul";
// const std::string MODEL_DIR = BASE_PATH + "/models/quant_result_test/deploy_check_data_int/QuantLeNet5/";
// const std::string DATASET_PATH = BASE_PATH + "/models/data/MNIST/raw/t10k-images-idx3-ubyte";
// const std::string VALIDATE_DATA_PATH = BASE_PATH + "/models/data/MNIST/raw/t10k-labels-idx1-ubyte";

// #define CONV1_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__conv1_weight.bin").c_str()
// #define CONV1_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__conv1_bias.bin").c_str()
// #define CONV2_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__conv2_weight.bin").c_str()
// #define CONV2_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__conv2_bias.bin").c_str()
// #define FC1_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__fc1_weight.bin").c_str()
// #define FC1_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__fc1_bias.bin").c_str()
// #define FC2_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__fc2_weight.bin").c_str()
// #define FC2_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__fc2_bias.bin").c_str()
// #define FC3_WEIGHT_BIN  (MODEL_DIR + "QuantLeNet5__fc3_weight.bin").c_str()
// #define FC3_BIAS_BIN  (MODEL_DIR + "QuantLeNet5__fc3_bias.bin").c_str()
// #define DATASET DATASET_PATH.c_str()
// #define VALIDATE_DATA VALIDATE_DATA_PATH.c_str() 

#define CONV1_OUT_CH 6
#define CONV1_IN_CH 1
#define KERNEL_SIZE 5
#define CONV1_IN_ROWS 28
#define CONV1_IN_COLS 28
#define CONV1_OUT_ROWS (CONV1_IN_ROWS - KERNEL_SIZE + 1)
#define CONV1_OUT_COLS (CONV1_IN_COLS - KERNEL_SIZE + 1)

#define CONV2_OUT_CH 16
#define CONV2_IN_CH 6
#define CONV2_IN_ROWS 12
#define CONV2_IN_COLS 12
#define CONV2_OUT_ROWS (CONV2_IN_ROWS - KERNEL_SIZE + 1)
#define CONV2_OUT_COLS (CONV2_IN_COLS - KERNEL_SIZE + 1)

#define FC1_IN_DIM 256
#define FC1_OUT_DIM 120

#define FC2_IN_DIM 120
#define FC2_OUT_DIM 84

#define FC3_IN_DIM 84
#define FC3_OUT_DIM 10

float SAMPLE_INPUT[784] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01176471, 0.07058824, 0.07058824, 0.07058824, 0.49411765, 0.53333333, 0.68627451, 0.10196078,
        0.65098039, 1., 0.96862745, 0.49803922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11764706, 0.14117647, 0.36862745, 0.60392157,
        0.66666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.88235294, 0.6745098, 0.99215686, 0.94901961, 0.76470588, 0.25098039, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0.19215686, 0.93333333, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.98431373, 0.36470588, 0.32156863, 0.32156863, 0.21960784, 0.15294118, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07058824, 0.85882353, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.71372549,
        0.96862745, 0.94509804, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.31372549, 0.61176471, 0.41960784, 0.99215686, 0.99215686, 0.80392157, 0.04313725, 0, 0.16862745, 0.60392157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05490196, 0.00392157, 0.60392157, 0.99215686, 0.35294118, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.54509804,
        0.99215686, 0.74509804, 0.00784314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0.04313725, 0.74509804, 0.99215686, 0.2745098, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1372549, 0.94509804, 0.88235294, 0.62745098,
        0.42352941, 0.00392157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.31764706, 0.94117647, 0.99215686, 0.99215686, 0.46666667, 0.09803922, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17647059, 0.72941176, 0.99215686, 0.99215686, 0.58823529, 0.10588235,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0.0627451, 0.36470588, 0.98823529, 0.99215686, 0.73333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.97647059, 0.99215686, 0.97647059, 0.25098039, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18039216, 0.50980392,
        0.71764706, 0.99215686, 0.99215686, 0.81176471, 0.00784314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.15294118, 0.58039216, 0.89803922, 0.99215686, 0.99215686, 0.99215686, 0.98039216, 0.71372549, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.09411765, 0.44705882, 0.86666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.78823529, 0.30588235, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.09019608, 0.25882353, 0.83529412, 0.99215686,
        0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.31764706, 0.00784314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.07058824, 0.67058824, 0.85882353, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.76470588, 0.31372549, 0.03529412, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0.21568627, 0.6745098, 0.88627451, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.95686275, 0.52156863, 0.04313725, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.53333333, 0.99215686, 0.99215686, 0.99215686,
        0.83137255, 0.52941176, 0.51764706, 0.0627451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

template<int OUT_C, int IN_C, int KERNEL, int IN_ROWS, int IN_COLS>
void conv_golden(
    const float in_data[IN_C * IN_ROWS * IN_COLS],
    float out_data[OUT_C * (IN_ROWS - KERNEL + 1) * (IN_COLS - KERNEL + 1)],
    const float weights[OUT_C * IN_C * KERNEL * KERNEL],
    const float bias[OUT_C]
) {
    const int OUT_ROWS = IN_ROWS - KERNEL + 1;
    const int OUT_COLS = IN_COLS - KERNEL + 1;
    
    // Loop over each output channel
    for (int oc = 0; oc < OUT_C; oc++) {
        // Loop over each output row
        for (int oh = 0; oh < OUT_ROWS; oh++) {
            // Loop over each output column
            for (int ow = 0; ow < OUT_COLS; ow++) {
                // Initialize accumulator with bias
                float acc = bias[oc];
                
                // Calculate convolution for current output position
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < KERNEL; kh++) {
                        for (int kw = 0; kw < KERNEL; kw++) {
                            // Input position
                            int ih = oh + kh;
                            int iw = ow + kw;
                            
                            // Calculate flattened indices
                            int in_idx = ic * IN_ROWS * IN_COLS + ih * IN_COLS + iw;
                            int w_idx = oc * IN_C * KERNEL * KERNEL + 
                                        ic * KERNEL * KERNEL + 
                                        kh * KERNEL + kw;
                            
                            // Accumulate weighted input
                            acc += in_data[in_idx] * weights[w_idx];
                        }
                    }
                }
                
                // Calculate output index and store result
                int out_idx = oc * OUT_ROWS * OUT_COLS + oh * OUT_COLS + ow;
                out_data[out_idx] = acc;
            }
        }
    }
}


template<int POOL_SIZE, int STRIDE, int IN_C, int IN_ROWS, int IN_COLS>
void pool_golden(
    const float in_data[IN_C*IN_ROWS*IN_COLS],
    float out_data[IN_C * ((IN_ROWS - POOL_SIZE) / STRIDE + 1) * ((IN_COLS - POOL_SIZE) / STRIDE + 1)]
) {
    int OUT_ROWS = (IN_ROWS - POOL_SIZE) / STRIDE + 1;
    int OUT_COLS = (IN_COLS - POOL_SIZE) / STRIDE + 1;

    // Calculate reference output (ground truth)
    for(int ch=0; ch < IN_C; ch++) {
        for(int out_r=0; out_r < OUT_ROWS; out_r++) {
            for(int out_c=0; out_c < OUT_COLS; out_c++) {
                float sum = 0.0f;
                for(int i=0; i < POOL_SIZE; i++) {
                    for(int j=0; j < POOL_SIZE; j++) {
                        int r = out_r * STRIDE + i;
                        int c = out_c * STRIDE + j;
                        sum += in_data[ch*IN_ROWS*IN_COLS + r*IN_COLS + c];
                    }
                }
                // Calculate average
                float avg = sum / (POOL_SIZE * POOL_SIZE);
                out_data[ch*OUT_ROWS*OUT_COLS + out_r*OUT_COLS + out_c] = avg;
            }
        }
    }
}

// template<int OC, int IC, int KERNEL_SIZE>
// void load_conv2d_weight(
//         const char* filename,
//         data_t weight[OC][IC][KERNEL_SIZE][KERNEL_SIZE]
//         ) {
//     std::ifstream file(filename, std::ios::binary);

//     if(!file.is_open()) {
//         std::cerr << "Error: cannot open " << CONV1_WEIGHT_BIN << std::endl;
//         exit(1);
//     }

//     for(int oc=0; oc<OC; ++oc) {
//         for(int ic=0; ic<IC; ++ic) {
//             for(int i=0; i<KERNEL_SIZE; ++i) {
//                 for(int j=0; j<KERNEL_SIZE; ++j) {
//                     float val;
//                     file.read(reinterpret_cast<char*>(&val), sizeof(float));
//                     weight[oc][ic][i][j] = data_t(val);
//                 }
//             }
//         }
//     }
// }

// template<int OUT_DIM, int IN_DIM>
// void load_fc_weight(
//         const char* filename,
//         data_t fc_weight[OUT_DIM][IN_DIM]
//         ) {
//     std::ifstream file(filename, std::ios::binary);

//     for(int o=0; o<OUT_DIM; ++o) {
//         for(int i=0; i<IN_DIM; ++i) {
//             float val;
//             file.read(reinterpret_cast<char*>(&val), sizeof(float));
//             fc_weight[o][i] = data_t(val);
//         }
//     }
// }

// template<int SIZE>
// void load_bias(
//         const char* filename,
//         data_t bias[SIZE]
//         ) {
//     std::ifstream file(filename, std::ios::binary);

//     for(int i=0; i<SIZE; ++i) {
//         float val;
//         file.read(reinterpret_cast<char*>(&val), sizeof(float));
//         bias[i] = data_t(val);
//     }
// }

// template<int IN_DIM>
// void load_input_to_stream(
//         const char* filename,
//         hls::stream<data_t>& in_stream
//         ) {
//     std::ifstream file(filename, std::ios::binary);

//     // skip MNIST headers (magic, num_images, rows, cols)
//     file.seekg(16, std::ios::beg);

//     for(int i=0; i<IN_DIM * IN_DIM; ++i) {
//         unsigned char pixel;
//         file.read(reinterpret_cast<char*>(&pixel), 1);
//         data_t val = data_t(pixel);
//         val /= data_t(255.0);
//         in_stream.write(val);
//     }
// }

// int check_against_golden(int predicted) {
//     std::ifstream file(VALIDATE_DATA, std::ios::binary);

//     /*
//      * for testing multiple input
//     if(!file.is_open()) {
//         std::cerr << "Error opening labels" << std::endl;
//         return 1;
//     }
//     * end loop
//     */

//     file.seekg(8, std::ios::beg);

//     unsigned char true_label;
//     file.read(reinterpret_cast<char*>(&true_label), 1);

//     int golden_class = static_cast<int>(true_label);

//     if(predicted != golden_class) {
//         std::cout << "ERROR: Predicted: " << predicted
//             << ", Golden: " << golden_class << std::endl;

//         return 1;
//     }

//     std::cout << "SUCCESS: Predict class: " << golden_class << std::endl;
//     return 0;
// }
#endif
