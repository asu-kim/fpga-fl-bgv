#include <iostream>
#include <cmath>
#include "lenet5/forward_path.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "test_utils.h"

#define CONV1_OUT_CH 6
#define CONV1_IN_CH 1
#define KERNEL_SIZE 5
#define CONV1_IN_ROWS 28
#define CONV1_IN_COLS 28

#define CONV2_OUT_CH 16
#define CONV2_IN_CH 6
#define CONV2_IN_ROWS 12
#define CONV2_IN_COLS 12

#define FC1_IN_DIM 256
#define FC1_OUT_DIM 120

#define FC2_IN_DIM 120
#define FC2_OUT_DIM 84

#define FC3_IN_DIM 84
#define FC3_OUT_DIM 10

int main() {
    // hls::stream<float> in_stream, out_stream;

    // float conv1_in_data[784] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01176471, 0.07058824, 0.07058824, 0.07058824, 0.49411765, 0.53333333, 0.68627451, 0.10196078,
    //     0.65098039, 1., 0.96862745, 0.49803922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11764706, 0.14117647, 0.36862745, 0.60392157,
    //     0.66666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.88235294, 0.6745098, 0.99215686, 0.94901961, 0.76470588, 0.25098039, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0.19215686, 0.93333333, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.98431373, 0.36470588, 0.32156863, 0.32156863, 0.21960784, 0.15294118, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07058824, 0.85882353, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.71372549,
    //     0.96862745, 0.94509804, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0.31372549, 0.61176471, 0.41960784, 0.99215686, 0.99215686, 0.80392157, 0.04313725, 0, 0.16862745, 0.60392157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05490196, 0.00392157, 0.60392157, 0.99215686, 0.35294118, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.54509804,
    //     0.99215686, 0.74509804, 0.00784314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0.04313725, 0.74509804, 0.99215686, 0.2745098, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1372549, 0.94509804, 0.88235294, 0.62745098,
    //     0.42352941, 0.00392157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0.31764706, 0.94117647, 0.99215686, 0.99215686, 0.46666667, 0.09803922, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17647059, 0.72941176, 0.99215686, 0.99215686, 0.58823529, 0.10588235,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0.0627451, 0.36470588, 0.98823529, 0.99215686, 0.73333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.97647059, 0.99215686, 0.97647059, 0.25098039, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18039216, 0.50980392,
    //     0.71764706, 0.99215686, 0.99215686, 0.81176471, 0.00784314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0.15294118, 0.58039216, 0.89803922, 0.99215686, 0.99215686, 0.99215686, 0.98039216, 0.71372549, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.09411765, 0.44705882, 0.86666667, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.78823529, 0.30588235, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.09019608, 0.25882353, 0.83529412, 0.99215686,
    //     0.99215686, 0.99215686, 0.99215686, 0.77647059, 0.31764706, 0.00784314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0.07058824, 0.67058824, 0.85882353, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.76470588, 0.31372549, 0.03529412, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0.21568627, 0.6745098, 0.88627451, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.95686275, 0.52156863, 0.04313725, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.53333333, 0.99215686, 0.99215686, 0.99215686,
    //     0.83137255, 0.52941176, 0.51764706, 0.0627451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 0, 0};
    float conv1_out_data[CONV1_OUT_CH * (CONV1_IN_ROWS - KERNEL_SIZE + 1) * (CONV1_IN_COLS - KERNEL_SIZE + 1)];
    float conv1_out_data_ref[CONV1_OUT_CH * (CONV1_IN_ROWS - KERNEL_SIZE + 1) * (CONV1_IN_COLS - KERNEL_SIZE + 1)];
    float conv1_weights[CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv1_bias[CONV1_OUT_CH];

    float conv1_in_data[784];

    for(int i = 0; i < CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS; i++) {
        conv1_in_data[i] = 0.001f * i;
    }

    for(int i = 0; i < CONV1_OUT_CH*CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV1_OUT_CH) {
            conv1_bias[i] = CONV1_BIAS_FP32_DATA[i];
            // conv1_bias[i] = CONV1_BIAS_INT8_DATA[i];
        }
        conv1_weights[i] = CONV1_WEIGHT_FP32_DATA[i];
        // conv1_weights[i] = CONV1_WEIGHT_INT8_DATA[i];
    }

    float avg_pool1_out_data[CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS];
    float avg_pool1_out_data_ref[CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS];

    float conv2_out_data[CONV2_OUT_CH * (CONV2_IN_ROWS - KERNEL_SIZE + 1) * (CONV2_IN_COLS - KERNEL_SIZE + 1)];
    float conv2_out_data_ref[CONV2_OUT_CH * (CONV2_IN_ROWS - KERNEL_SIZE + 1) * (CONV2_IN_COLS - KERNEL_SIZE + 1)];
    float conv2_weights[CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE];
    float conv2_bias[CONV2_OUT_CH];

    for(int i = 0; i < CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV2_OUT_CH) {
            conv2_bias[i] = CONV2_BIAS_FP32_DATA[i];
        }
        conv2_weights[i] = CONV2_WEIGHT_FP32_DATA[i];
    }

    float avg_pool2_out_data[FC1_IN_DIM];
    float avg_pool2_out_data_ref[FC1_IN_DIM];

    float fc1_out_data[FC1_OUT_DIM];
    float fc1_out_data_ref[FC1_OUT_DIM];
    float fc1_weights[FC1_IN_DIM*FC1_OUT_DIM];
    float fc1_bias[FC1_OUT_DIM];

    for(int i = 0; i < FC1_IN_DIM*FC1_OUT_DIM; i++) {
        if (i < FC1_OUT_DIM) {
            fc1_bias[i] = FC1_BIAS_FP32_DATA[i];
        }
        fc1_weights[i] = FC1_WEIGHT_FP32_DATA[i];
    }

    float fc2_out_data[FC2_OUT_DIM];
    float fc2_out_data_ref[FC2_OUT_DIM];
    float fc2_weights[FC2_IN_DIM*FC2_OUT_DIM];
    float fc2_bias[FC2_OUT_DIM];

    for(int i = 0; i < FC2_IN_DIM*FC2_OUT_DIM; i++) {
        if (i < FC2_OUT_DIM) {
            fc2_bias[i] = FC2_BIAS_FP32_DATA[i];
        }
        fc2_weights[i] = FC2_WEIGHT_FP32_DATA[i];
    }

    float fc3_out_data[FC3_OUT_DIM];
    float fc3_out_data_ref[FC3_OUT_DIM];
    float fc3_weights[FC3_IN_DIM*FC3_OUT_DIM];
    float fc3_bias[FC3_OUT_DIM];

    for(int i = 0; i < FC3_IN_DIM*FC3_OUT_DIM; i++) {
        if (i < FC3_OUT_DIM) {
            fc3_bias[i] = FC3_BIAS_FP32_DATA[i];
        }
        fc3_weights[i] = FC3_WEIGHT_FP32_DATA[i];
    }

    // run conv2d
    // conv1(in_stream, out_stream, flatten_weights, flatten_bias);
    forward_path(conv1_in_data, conv1_out_data, avg_pool1_out_data, conv2_out_data, avg_pool2_out_data, 
        fc1_out_data, fc2_out_data, fc3_out_data, conv1_weights, conv1_bias, conv2_weights, conv2_bias,
        fc1_weights, fc1_bias, fc2_weights, fc2_bias, fc3_weights, fc3_bias);
    conv_golden<6, 1, 5, 28, 28>(conv1_in_data, conv1_out_data_ref, conv1_weights, conv1_bias);

    int errors = 0;
    float max_diff = 0.0f;
    const int output_size = (CONV1_IN_ROWS - KERNEL_SIZE + 1) * (CONV1_IN_COLS - KERNEL_SIZE + 1) * CONV1_OUT_CH;
    std::cout << "conv1_out_data_ref = [";
    for(int i=0; i<output_size; ++i) {
        std::cout << conv1_out_data_ref[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    std::cout << "conv1_out_data = [";
    for(int i=0; i<output_size; ++i) {
        std::cout << conv1_out_data[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    std::cout << "Error indexes: ";
    for(int i=0; i<output_size; i++) {
        float diff = std::fabs(conv1_out_data[i] - conv1_out_data_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << conv1_out_data[i] 
                          << ", expected " << conv1_out_data_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << std::endl;

    return errors;
}
