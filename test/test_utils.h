#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <unistd.h>
#include <limits.h>

#include <fstream>
#include <filesystem>
#include <string>

#include "constants.hpp"
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

data_ap_fixed_t SAMPLE_INPUT[784] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
    const data_ap_fixed_t in_data[IN_C * IN_ROWS * IN_COLS],
    data_ap_fixed_t out_data[OUT_C * (IN_ROWS - KERNEL + 1) * (IN_COLS - KERNEL + 1)],
    const data_ap_fixed_t weights[OUT_C * IN_C * KERNEL * KERNEL],
    const data_ap_fixed_t bias[OUT_C]
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
                data_ap_fixed_t acc = bias[oc];
                
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
    const data_ap_fixed_t in_data[IN_C*IN_ROWS*IN_COLS],
    data_ap_fixed_t out_data[IN_C * ((IN_ROWS - POOL_SIZE) / STRIDE + 1) * ((IN_COLS - POOL_SIZE) / STRIDE + 1)]
) {
    int OUT_ROWS = (IN_ROWS - POOL_SIZE) / STRIDE + 1;
    int OUT_COLS = (IN_COLS - POOL_SIZE) / STRIDE + 1;

    // Calculate reference output (ground truth)
    for(int ch=0; ch < IN_C; ch++) {
        for(int out_r=0; out_r < OUT_ROWS; out_r++) {
            for(int out_c=0; out_c < OUT_COLS; out_c++) {
                data_ap_fixed_t sum = 0.0f;
                for(int i=0; i < POOL_SIZE; i++) {
                    for(int j=0; j < POOL_SIZE; j++) {
                        int r = out_r * STRIDE + i;
                        int c = out_c * STRIDE + j;
                        sum += in_data[ch*IN_ROWS*IN_COLS + r*IN_COLS + c];
                    }
                }
                // Calculate average
                data_ap_fixed_t avg = sum / (POOL_SIZE * POOL_SIZE);
                out_data[ch*OUT_ROWS*OUT_COLS + out_r*OUT_COLS + out_c] = avg;
            }
        }
    }
}

template<int IN_DIM, int OUT_DIM>
void fc_golden(
    const data_ap_fixed_t in_data[IN_DIM],
    data_ap_fixed_t out_data[OUT_DIM],
    const data_ap_fixed_t weight[IN_DIM*OUT_DIM],
    const data_ap_fixed_t bias[OUT_DIM],
    bool use_relu
) {
    // Initialize with bias
    for(int j=0; j<OUT_DIM; j++) {
        out_data[j] = bias[j];
    }
    
    // Matrix multiplication
    for(int i=0; i<IN_DIM; i++) {
        for(int j=0; j<OUT_DIM; j++) {
            out_data[j] += in_data[i] * weight[i*OUT_DIM + j];
        }
    }
    
    // Apply ReLU if needed
    if(use_relu) {
        for(int j=0; j<OUT_DIM; j++) {
            if(out_data[j] < 0) {
                out_data[j] = 0;
            }
        }
    }
}

void forward_golden(
    const data_ap_fixed_t* in_data,
    const data_ap_fixed_t* weights,
    const data_ap_fixed_t* biases,
    data_ap_fixed_t* outs
) {
    data_ap_fixed_t conv1_out[NUM_CONV1_OUTS];
    data_ap_fixed_t pool1_out[NUM_POOL1_OUTS];
    data_ap_fixed_t conv2_out[NUM_CONV2_OUTS];
    data_ap_fixed_t pool2_out[NUM_POOL2_OUTS];
    data_ap_fixed_t fc1_out[NUM_FC1_OUTS];
    data_ap_fixed_t fc2_out[NUM_FC2_OUTS];
    data_ap_fixed_t fc3_out[NUM_FC3_OUTS];
    
    // Create local arrays for weights and biases
    data_ap_fixed_t conv1_weight[NUM_CONV1_WEIGHTS];
    data_ap_fixed_t conv1_bias[NUM_CONV1_BIASES];
    data_ap_fixed_t conv2_weight[NUM_CONV2_WEIGHTS];
    data_ap_fixed_t conv2_bias[NUM_CONV2_BIASES];
    data_ap_fixed_t fc1_weight[NUM_FC1_WEIGHTS];
    data_ap_fixed_t fc1_bias[NUM_FC1_BIASES];
    data_ap_fixed_t fc2_weight[NUM_FC2_WEIGHTS];
    data_ap_fixed_t fc2_bias[NUM_FC2_BIASES];
    data_ap_fixed_t fc3_weight[NUM_FC3_WEIGHTS];
    data_ap_fixed_t fc3_bias[NUM_FC3_BIASES];
    
    // Copy weights and biases from consolidated arrays
    for(int i = 0; i < NUM_CONV1_WEIGHTS; i++) {
        conv1_weight[i] = weights[CONV1_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < NUM_CONV1_BIASES; i++) {
        conv1_bias[i] = biases[CONV1_BIAS_OFFSET + i];
    }
    
    for(int i = 0; i < NUM_CONV2_WEIGHTS; i++) {
        conv2_weight[i] = weights[CONV2_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < NUM_CONV2_BIASES; i++) {
        conv2_bias[i] = biases[CONV2_BIAS_OFFSET + i];
    }
    
    for(int i = 0; i < NUM_FC1_WEIGHTS; i++) {
        fc1_weight[i] = weights[FC1_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < NUM_FC1_BIASES; i++) {
        fc1_bias[i] = biases[FC1_BIAS_OFFSET + i];
    }
    
    for(int i = 0; i < NUM_FC2_WEIGHTS; i++) {
        fc2_weight[i] = weights[FC2_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < NUM_FC2_BIASES; i++) {
        fc2_bias[i] = biases[FC2_BIAS_OFFSET + i];
    }
    
    for(int i = 0; i < NUM_FC3_WEIGHTS; i++) {
        fc3_weight[i] = weights[FC3_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < NUM_FC3_BIASES; i++) {
        fc3_bias[i] = biases[FC3_BIAS_OFFSET + i];
    }
    
    // Conv1
    conv_golden<CONV1_OUT_CH, CONV1_IN_CH, KERNEL_SIZE, CONV1_IN_ROWS, CONV1_IN_COLS>(in_data, conv1_out, conv1_weight, conv1_bias);

    // Pool1
    pool_golden<2, 2, CONV1_OUT_CH, CONV1_OUT_ROWS, CONV1_OUT_COLS>(conv1_out, pool1_out);

    // Conv2
    conv_golden<CONV2_OUT_CH, CONV2_IN_CH, KERNEL_SIZE, CONV2_IN_ROWS, CONV2_IN_COLS>(pool1_out, conv2_out, conv2_weight, conv2_bias);

    // Pool2
    pool_golden<2, 2, CONV2_OUT_CH, CONV2_OUT_ROWS, CONV2_OUT_COLS>(conv2_out, pool2_out);

    // FC1
    fc_golden<FC1_IN_DIM, FC1_OUT_DIM>(pool2_out, fc1_out, fc1_weight, fc1_bias, true);

    // FC2
    fc_golden<FC2_IN_DIM, FC2_OUT_DIM>(fc1_out, fc2_out, fc2_weight, fc2_bias, true);

    // FC3
    fc_golden<FC3_IN_DIM, FC3_OUT_DIM>(fc2_out, fc3_out, fc3_weight, fc3_bias, false);
    
    // Copy outputs to the consolidated output array
    for(int i = 0; i < NUM_CONV1_OUTS; i++) {
        outs[CONV1_OUT_OFFSET + i] = conv1_out[i];
    }
    
    for(int i = 0; i < NUM_POOL1_OUTS; i++) {
        outs[POOL1_OUT_OFFSET + i] = pool1_out[i];
    }
    
    for(int i = 0; i < NUM_CONV2_OUTS; i++) {
        outs[CONV2_OUT_OFFSET + i] = conv2_out[i];
    }
    
    for(int i = 0; i < NUM_POOL2_OUTS; i++) {
        outs[POOL2_OUT_OFFSET + i] = pool2_out[i];
    }
    
    for(int i = 0; i < NUM_FC1_OUTS; i++) {
        outs[FC1_OUT_OFFSET + i] = fc1_out[i];
    }
    
    for(int i = 0; i < NUM_FC2_OUTS; i++) {
        outs[FC2_OUT_OFFSET + i] = fc2_out[i];
    }
    
    for(int i = 0; i < NUM_FC3_OUTS; i++) {
        outs[FC3_OUT_OFFSET + i] = fc3_out[i];
    }
}

template<int N>
void mse_loss_golden(
        const data_ap_fixed_t y_pred[N],
        const data_ap_fixed_t y_true[N],
        data_ap_fixed_t &loss,
        data_ap_fixed_t grads[N]
        ) {
    data_ap_fixed_t acc_loss = 0.0f;

    for(int i=0; i<N; ++i) {
        data_ap_fixed_t p = y_pred[i];
        data_ap_fixed_t t = y_true[i];
        data_ap_fixed_t diff = t - p;
        acc_loss += diff * diff;
        grads[i] = (data_ap_fixed_t(2.0) * diff) / N;
    }

    loss = (acc_loss / N);
}

template<int IN_DIM, int OUT_DIM>
void fc_bwd_golden(
    const data_ap_fixed_t in_activation[IN_DIM],
    const data_ap_fixed_t grads[OUT_DIM],
    const data_ap_fixed_t in_weight[IN_DIM*OUT_DIM],
    data_ap_fixed_t dX[IN_DIM],
    data_ap_fixed_t dW[IN_DIM*OUT_DIM],
    data_ap_fixed_t dB[OUT_DIM],
    bool use_relu = true
) {
    // Reconstruct weights for easier access
    data_ap_fixed_t weight[IN_DIM][OUT_DIM];
    for(int i=0; i<IN_DIM; ++i) {
        for(int j=0; j<OUT_DIM; ++j) {
            int idx = i * OUT_DIM + j;
            weight[i][j] = in_weight[idx];
        }
    }
    
    // Bias gradients
    for(int j=0; j<OUT_DIM; ++j) {
        dB[j] = grads[j];
    }

    // Weight gradients
    for(int i=0; i<IN_DIM; ++i) {
        for(int j=0; j<OUT_DIM; ++j) {
            int idx = i * OUT_DIM + j;
            dW[idx] = in_activation[i] * grads[j];
        }
    }

    // Input gradients
    for(int i=0; i<IN_DIM; ++i) {
        data_ap_fixed_t acc = 0;
        for(int j=0; j<OUT_DIM; ++j) {
            acc += weight[i][j] * grads[j];
        }
        if(use_relu) {
            acc *= (in_activation[i] > 0 ? data_ap_fixed_t(1.0) : data_ap_fixed_t(0.0));
        }
        dX[i] = acc;
    }
}

template<int POOL_SIZE, int STRIDE, int IN_C, int IN_ROWS, int IN_COLS>
void pool_bwd_golden(
    const data_ap_fixed_t in_data[IN_C*((IN_ROWS-POOL_SIZE)/STRIDE+1)*((IN_COLS-POOL_SIZE)/STRIDE+1)],
    data_ap_fixed_t out_data[IN_C*IN_ROWS*IN_COLS]
) {
    int OUT_ROWS = (IN_ROWS - POOL_SIZE) / STRIDE + 1;
    int OUT_COLS = (IN_COLS - POOL_SIZE) / STRIDE + 1;
    
    // Initialize output to zeros
    for (int i = 0; i < IN_C*IN_ROWS*IN_COLS; i++) {
        out_data[i] = 0.0f;
    }
    
    const data_ap_fixed_t scale = 1.0f / (POOL_SIZE * POOL_SIZE);
    
    // For each input position
    for (int c = 0; c < IN_C; c++) {
        for (int h = 0; h < IN_ROWS; h++) {
            for (int w = 0; w < IN_COLS; w++) {
                // Find all output positions that this input contributes to
                int out_h_start = (h < POOL_SIZE) ? 0 : (h - POOL_SIZE + STRIDE) / STRIDE;
                int out_w_start = (w < POOL_SIZE) ? 0 : (w - POOL_SIZE + STRIDE) / STRIDE;
                int out_h_end = std::min(OUT_ROWS, (h / STRIDE) + 1);
                int out_w_end = std::min(OUT_COLS, (w / STRIDE) + 1);
                
                // Add contributions from all relevant output positions
                for (int oh = out_h_start; oh < out_h_end; oh++) {
                    for (int ow = out_w_start; ow < out_w_end; ow++) {
                        // Check if input position (h,w) is in the pooling window of output (oh,ow)
                        int h_start = oh * STRIDE;
                        int w_start = ow * STRIDE;
                        
                        if (h >= h_start && h < h_start + POOL_SIZE && 
                            w >= w_start && w < w_start + POOL_SIZE) {
                            int in_idx = c*OUT_ROWS*OUT_COLS + oh*OUT_COLS + ow;
                            int out_idx = c*IN_ROWS*IN_COLS + h*IN_COLS + w;
                            out_data[out_idx] += in_data[in_idx] * scale;
                        }
                    }
                }
            }
        }
    }
}

template<int OUT_C, int IN_C, int KERNEL, int IN_ROWS, int IN_COLS>
void conv_bwd_golden(
    const data_ap_fixed_t in_activation[IN_C * IN_ROWS * IN_COLS],
    const data_ap_fixed_t grads[OUT_C * (IN_ROWS - KERNEL + 1) * (IN_COLS - KERNEL + 1)],
    const data_ap_fixed_t in_weight[OUT_C * IN_C * KERNEL * KERNEL],
    data_ap_fixed_t out_grads[IN_C * IN_ROWS * IN_COLS],
    data_ap_fixed_t dW[OUT_C * IN_C * KERNEL * KERNEL],
    data_ap_fixed_t dB[OUT_C]
) {
    const int OUT_ROWS = IN_ROWS - KERNEL + 1;
    const int OUT_COLS = IN_COLS - KERNEL + 1;
    
    // Initialize output arrays to zero
    for (int i = 0; i < IN_C * IN_ROWS * IN_COLS; ++i) {
        out_grads[i] = 0.0f;
    }
    
    for (int i = 0; i < OUT_C * IN_C * KERNEL * KERNEL; ++i) {
        dW[i] = 0.0f;
    }
    
    for (int i = 0; i < OUT_C; ++i) {
        dB[i] = 0.0f;
    }
    
    // Part 1: Compute bias gradients (dB)
    // For each output channel, sum all gradient values across spatial dimensions
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int r = 0; r < OUT_ROWS; ++r) {
            for (int c = 0; c < OUT_COLS; ++c) {
                int grad_idx = oc * (OUT_ROWS * OUT_COLS) + r * OUT_COLS + c;
                dB[oc] += grads[grad_idx];
            }
        }
    }
    
    // Part 2: Compute weight gradients (dW)
    // For each position in the output gradient, update the corresponding weight gradients
    // using the input activation values within the kernel window
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int ic = 0; ic < IN_C; ++ic) {
            for (int kr = 0; kr < KERNEL; ++kr) {
                for (int kc = 0; kc < KERNEL; ++kc) {
                    data_ap_fixed_t gradient_sum = 0.0f;
                    
                    for (int r = 0; r < OUT_ROWS; ++r) {
                        for (int c = 0; c < OUT_COLS; ++c) {
                            int grad_idx = oc * (OUT_ROWS * OUT_COLS) + r * OUT_COLS + c;
                            int in_idx = ic * (IN_ROWS * IN_COLS) + (r + kr) * IN_COLS + (c + kc);
                            
                            gradient_sum += grads[grad_idx] * in_activation[in_idx];
                        }
                    }
                    
                    int weight_idx = oc * (IN_C * KERNEL * KERNEL) + ic * (KERNEL * KERNEL) + kr * KERNEL + kc;
                    dW[weight_idx] = gradient_sum;
                }
            }
        }
    }
    
    // Part 3: Compute input gradients (out_grads)
    // For each input position, compute the gradient by convolving the output gradients
    // with the flipped weights
    for (int ic = 0; ic < IN_C; ++ic) {
        for (int r = 0; r < IN_ROWS; ++r) {
            for (int c = 0; c < IN_COLS; ++c) {
                data_ap_fixed_t gradient_sum = 0.0f;
                
                // Iterate over all output channels
                for (int oc = 0; oc < OUT_C; ++oc) {
                    // Iterate over the kernel window that could affect this input position
                    for (int kr = 0; kr < KERNEL; ++kr) {
                        for (int kc = 0; kc < KERNEL; ++kc) {
                            // Calculate corresponding output position
                            int out_r = r - kr;
                            int out_c = c - kc;
                            
                            // Check if this output position is valid
                            if (out_r >= 0 && out_r < OUT_ROWS && out_c >= 0 && out_c < OUT_COLS) {
                                int grad_idx = oc * (OUT_ROWS * OUT_COLS) + out_r * OUT_COLS + out_c;
                                
                                // Use flipped kernel indices for backprop (rotate 180 degrees)
                                int weight_idx = oc * (IN_C * KERNEL * KERNEL) + 
                                                ic * (KERNEL * KERNEL) + 
                                                (KERNEL - 1 - kr) * KERNEL + 
                                                (KERNEL - 1 - kc);
                                
                                gradient_sum += grads[grad_idx] * in_weight[weight_idx];
                            }
                        }
                    }
                }
                
                int in_idx = ic * (IN_ROWS * IN_COLS) + r * IN_COLS + c;
                out_grads[in_idx] = gradient_sum;
            }
        }
    }
}

void backward_golden(
    const data_ap_fixed_t* in_data,
    const data_ap_fixed_t* weights,
    const data_ap_fixed_t* biases,
    const data_ap_fixed_t* outputs,
    const data_ap_fixed_t* label,
    data_ap_fixed_t* updated_weights,
    data_ap_fixed_t* updated_biases,
    data_ap_fixed_t& loss
) {
    // Extract layer outputs from consolidated outputs array
    const data_ap_fixed_t* conv1_out = &outputs[CONV1_OUT_OFFSET];
    const data_ap_fixed_t* pool1_out = &outputs[POOL1_OUT_OFFSET];
    const data_ap_fixed_t* conv2_out = &outputs[CONV2_OUT_OFFSET];
    const data_ap_fixed_t* pool2_out = &outputs[POOL2_OUT_OFFSET];
    const data_ap_fixed_t* fc1_out = &outputs[FC1_OUT_OFFSET];
    const data_ap_fixed_t* fc2_out = &outputs[FC2_OUT_OFFSET];
    const data_ap_fixed_t* fc3_out = &outputs[FC3_OUT_OFFSET];
    
    // Extract weights from consolidated weights array
    const data_ap_fixed_t* conv1_weight = &weights[CONV1_WEIGHT_OFFSET];
    const data_ap_fixed_t* conv2_weight = &weights[CONV2_WEIGHT_OFFSET];
    const data_ap_fixed_t* fc1_weight = &weights[FC1_WEIGHT_OFFSET];
    const data_ap_fixed_t* fc2_weight = &weights[FC2_WEIGHT_OFFSET];
    const data_ap_fixed_t* fc3_weight = &weights[FC3_WEIGHT_OFFSET];
    
    // Extract biases from consolidated biases array
    const data_ap_fixed_t* conv1_bias = &biases[CONV1_BIAS_OFFSET];
    const data_ap_fixed_t* conv2_bias = &biases[CONV2_BIAS_OFFSET];
    const data_ap_fixed_t* fc1_bias = &biases[FC1_BIAS_OFFSET];
    const data_ap_fixed_t* fc2_bias = &biases[FC2_BIAS_OFFSET];
    const data_ap_fixed_t* fc3_bias = &biases[FC3_BIAS_OFFSET];
    
    // Prepare pointers to updated weights and biases
    data_ap_fixed_t* conv1_updated_weight = &updated_weights[CONV1_WEIGHT_OFFSET];
    data_ap_fixed_t* conv2_updated_weight = &updated_weights[CONV2_WEIGHT_OFFSET];
    data_ap_fixed_t* fc1_updated_weight = &updated_weights[FC1_WEIGHT_OFFSET];
    data_ap_fixed_t* fc2_updated_weight = &updated_weights[FC2_WEIGHT_OFFSET];
    data_ap_fixed_t* fc3_updated_weight = &updated_weights[FC3_WEIGHT_OFFSET];
    
    data_ap_fixed_t* conv1_updated_bias = &updated_biases[CONV1_BIAS_OFFSET];
    data_ap_fixed_t* conv2_updated_bias = &updated_biases[CONV2_BIAS_OFFSET];
    data_ap_fixed_t* fc1_updated_bias = &updated_biases[FC1_BIAS_OFFSET];
    data_ap_fixed_t* fc2_updated_bias = &updated_biases[FC2_BIAS_OFFSET];
    data_ap_fixed_t* fc3_updated_bias = &updated_biases[FC3_BIAS_OFFSET];
    
    // Create temporary buffers for gradients
    data_ap_fixed_t out_grad[FC3_OUT_DIM];
    
    // Loss
    mse_loss_golden<FC3_OUT_DIM>(fc3_out, label, loss, out_grad);

    data_ap_fixed_t fc3_dX[FC3_IN_DIM];
    data_ap_fixed_t fc3_dW[NUM_FC3_WEIGHTS];
    data_ap_fixed_t fc3_dB[NUM_FC3_BIASES];
    // FC3 Bwd
    fc_bwd_golden<FC3_IN_DIM, FC3_OUT_DIM>(fc2_out, out_grad, fc3_weight, fc3_dX, fc3_dW, fc3_dB, false);
    // Update FC3 weights and biases
    for(int i = 0; i < NUM_FC3_WEIGHTS; i++) {
        fc3_updated_weight[i] = fc3_weight[i] - lr * fc3_dW[i];
    }
    for(int i = 0; i < NUM_FC3_BIASES; i++) {
        fc3_updated_bias[i] = fc3_bias[i] - lr * fc3_dB[i];
    }

    data_ap_fixed_t fc2_dX[FC2_IN_DIM];
    data_ap_fixed_t fc2_dW[NUM_FC2_WEIGHTS];
    data_ap_fixed_t fc2_dB[NUM_FC2_BIASES];
    // FC2 Bwd
    fc_bwd_golden<FC2_IN_DIM, FC2_OUT_DIM>(fc1_out, fc3_dX, fc2_weight, fc2_dX, fc2_dW, fc2_dB, true);
    // Update FC2 weights and biases
    for(int i = 0; i < NUM_FC2_WEIGHTS; i++) {
        fc2_updated_weight[i] = fc2_weight[i] - lr * fc2_dW[i];
    }
    for(int i = 0; i < NUM_FC2_BIASES; i++) {
        fc2_updated_bias[i] = fc2_bias[i] - lr * fc2_dB[i];
    }

    data_ap_fixed_t fc1_dX[FC1_IN_DIM];
    data_ap_fixed_t fc1_dW[NUM_FC1_WEIGHTS];
    data_ap_fixed_t fc1_dB[NUM_FC1_BIASES];
    // FC1 Bwd
    fc_bwd_golden<FC1_IN_DIM, FC1_OUT_DIM>(pool2_out, fc2_dX, fc1_weight, fc1_dX, fc1_dW, fc1_dB, true);
    // Update FC1 weights and biases
    for(int i = 0; i < NUM_FC1_WEIGHTS; i++) {
        fc1_updated_weight[i] = fc1_weight[i] - lr * fc1_dW[i];
    }
    for(int i = 0; i < NUM_FC1_BIASES; i++) {
        fc1_updated_bias[i] = fc1_bias[i] - lr * fc1_dB[i];
    }

    data_ap_fixed_t pool2_dX[CONV2_OUT_CH*CONV2_OUT_ROWS*CONV2_OUT_COLS];
    // Pool2 Bwd
    pool_bwd_golden<2, 2, CONV2_OUT_CH, CONV2_OUT_ROWS, CONV2_OUT_COLS>(fc1_dX, pool2_dX);

    data_ap_fixed_t conv2_dX[CONV2_IN_CH*CONV2_IN_ROWS*CONV2_IN_COLS];
    data_ap_fixed_t conv2_dW[NUM_CONV2_WEIGHTS];
    data_ap_fixed_t conv2_dB[NUM_CONV2_BIASES];
    // Conv2 Bwd
    conv_bwd_golden<CONV2_OUT_CH, CONV2_IN_CH, KERNEL_SIZE, CONV2_IN_ROWS, CONV2_OUT_ROWS>(pool1_out, pool2_dX, conv2_weight, conv2_dX, conv2_dW, conv2_dB);
    // Update Conv2 weights and biases
    for(int i = 0; i < NUM_CONV2_WEIGHTS; i++) {
        conv2_updated_weight[i] = conv2_weight[i] - lr * conv2_dW[i];
    }
    for(int i = 0; i < NUM_CONV2_BIASES; i++) {
        conv2_updated_bias[i] = conv2_bias[i] - lr * conv2_dB[i];
    }

    data_ap_fixed_t pool1_dX[CONV1_OUT_CH*CONV1_OUT_ROWS*CONV1_OUT_COLS];
    // Pool1 Bwd
    pool_bwd_golden<2, 2, CONV1_OUT_CH, CONV1_OUT_ROWS, CONV1_OUT_COLS>(conv2_dX, pool1_dX);

    data_ap_fixed_t conv1_dX[CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS];
    data_ap_fixed_t conv1_dW[NUM_CONV1_WEIGHTS];
    data_ap_fixed_t conv1_dB[NUM_CONV1_BIASES];
    // Conv1 Bwd
    conv_bwd_golden<CONV1_OUT_CH, CONV1_IN_CH, KERNEL_SIZE, CONV1_IN_ROWS, CONV1_OUT_ROWS>(in_data, pool1_dX, conv1_weight, conv1_dX, conv1_dW, conv1_dB);
    // Update Conv1 weights and biases
    for(int i = 0; i < NUM_CONV1_WEIGHTS; i++) {
        conv1_updated_weight[i] = conv1_weight[i] - lr * conv1_dW[i];
    }
    for(int i = 0; i < NUM_CONV1_BIASES; i++) {
        conv1_updated_bias[i] = conv1_bias[i] - lr * conv1_dB[i];
    }
}
#endif
