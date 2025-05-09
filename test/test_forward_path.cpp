#include <iostream>
#include <cmath>
#include "lenet5/forward_path.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "test_utils.h"
#include "constants.hpp"
#include "hls_math.h"

int main() {
    data_ap_fixed_t in_data[784];
    
    // Consolidated arrays
    data_ap_fixed_t weights[TOTAL_WEIGHTS_SIZE];
    data_ap_fixed_t biases[TOTAL_BIASES_SIZE];
    data_ap_fixed_t outs[TOTAL_OUTS_SIZE]; // Consolidated output array for all layers
    
    // Reference outputs for each layer
    data_ap_fixed_t conv1_out_ref[NUM_CONV1_OUTS];
    data_ap_fixed_t pool1_out_ref[NUM_POOL1_OUTS];
    data_ap_fixed_t conv2_out_ref[NUM_CONV2_OUTS];
    data_ap_fixed_t pool2_out_ref[NUM_POOL2_OUTS];
    data_ap_fixed_t fc1_out_ref[NUM_FC1_OUTS];
    data_ap_fixed_t fc2_out_ref[NUM_FC2_OUTS];
    data_ap_fixed_t fc3_out_ref[NUM_FC3_OUTS];
    
    // Local separate arrays for golden functions
    data_ap_fixed_t conv1_weight[CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    data_ap_fixed_t conv1_bias[CONV1_OUT_CH];
    data_ap_fixed_t conv2_weight[CONV2_OUT_CH * CONV2_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    data_ap_fixed_t conv2_bias[CONV2_OUT_CH];
    data_ap_fixed_t fc1_weight[FC1_IN_DIM * FC1_OUT_DIM];
    data_ap_fixed_t fc1_bias[FC1_OUT_DIM];
    data_ap_fixed_t fc2_weight[FC2_IN_DIM * FC2_OUT_DIM];
    data_ap_fixed_t fc2_bias[FC2_OUT_DIM];
    data_ap_fixed_t fc3_weight[FC3_IN_DIM * FC3_OUT_DIM];
    data_ap_fixed_t fc3_bias[FC3_OUT_DIM];

    for(int i = 0; i < CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS; i++) {
        in_data[i] = SAMPLE_INPUT[i];
    }

    // Load Conv1 weights and biases
    for(int i = 0; i < CONV1_OUT_CH*CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV1_OUT_CH) {
            biases[CONV1_BIAS_OFFSET + i] = CONV1_BIAS_FP32_DATA[i];
            conv1_bias[i] = CONV1_BIAS_FP32_DATA[i];
        }
        weights[CONV1_WEIGHT_OFFSET + i] = CONV1_WEIGHT_FP32_DATA[i];
        conv1_weight[i] = CONV1_WEIGHT_FP32_DATA[i];
    }

    // Load Conv2 weights and biases
    for(int i = 0; i < CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV2_OUT_CH) {
            biases[CONV2_BIAS_OFFSET + i] = CONV2_BIAS_FP32_DATA[i];
            conv2_bias[i] = CONV2_BIAS_FP32_DATA[i];
        }
        weights[CONV2_WEIGHT_OFFSET + i] = CONV2_WEIGHT_FP32_DATA[i];
        conv2_weight[i] = CONV2_WEIGHT_FP32_DATA[i];
    }

    // Load FC1 weights and biases
    for(int i = 0; i < FC1_IN_DIM*FC1_OUT_DIM; i++) {
        if (i < FC1_OUT_DIM) {
            biases[FC1_BIAS_OFFSET + i] = FC1_BIAS_FP32_DATA[i];
            fc1_bias[i] = FC1_BIAS_FP32_DATA[i];
        }
        weights[FC1_WEIGHT_OFFSET + i] = FC1_WEIGHT_FP32_DATA[i];
        fc1_weight[i] = FC1_WEIGHT_FP32_DATA[i];
    }

    // Load FC2 weights and biases
    for(int i = 0; i < FC2_IN_DIM*FC2_OUT_DIM; i++) {
        if (i < FC2_OUT_DIM) {
            biases[FC2_BIAS_OFFSET + i] = FC2_BIAS_FP32_DATA[i];
            fc2_bias[i] = FC2_BIAS_FP32_DATA[i];
        }
        weights[FC2_WEIGHT_OFFSET + i] = FC2_WEIGHT_FP32_DATA[i];
        fc2_weight[i] = FC2_WEIGHT_FP32_DATA[i];
    }

    // Load FC3 weights and biases
    for(int i = 0; i < FC3_IN_DIM*FC3_OUT_DIM; i++) {
        if (i < FC3_OUT_DIM) {
            biases[FC3_BIAS_OFFSET + i] = FC3_BIAS_FP32_DATA[i];
            fc3_bias[i] = FC3_BIAS_FP32_DATA[i];
        }
        weights[FC3_WEIGHT_OFFSET + i] = FC3_WEIGHT_FP32_DATA[i];
        fc3_weight[i] = FC3_WEIGHT_FP32_DATA[i];
    }

    // Run forward path with consolidated arrays
    forward_path(
        in_data,
        weights,
        biases,
        outs  // Now using consolidated outs array
    );

    int global_errors = 0;

    // Run golden model and verify each layer

    // Conv1 Test
    conv_golden<CONV1_OUT_CH, CONV1_IN_CH, KERNEL_SIZE, CONV1_IN_ROWS, CONV1_IN_COLS>(
        in_data, conv1_out_ref, conv1_weight, conv1_bias);

    int errors = 0;
    data_ap_fixed_t max_diff = 0.0f;
    const int conv1_output_size = NUM_CONV1_OUTS;

    std::cout << "conv1_out = [" << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << outs[CONV1_OUT_OFFSET + i * (CONV1_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;
    
    std::cout << "Conv1 error indexes: ";
    for(int i=0; i<conv1_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(outs[CONV1_OUT_OFFSET + i] - conv1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << outs[CONV1_OUT_OFFSET + i] 
                          << ", expected " << conv1_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // Pool1 Test
    pool_golden<2, 2, CONV1_OUT_CH, CONV1_OUT_ROWS, CONV1_OUT_COLS>(conv1_out_ref, pool1_out_ref);
    errors = 0;
    max_diff = 0.0f;
    const int pool1_output_size = NUM_POOL1_OUTS;

    std::cout << "pool1_out = [" << std::endl;
    for(int i=0; i<3; i++) {
        for(int j = 0; j < 3; j++) {
            std::cout << outs[POOL1_OUT_OFFSET + i * CONV2_IN_COLS + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Pool1 error indexes: ";
    for(int i=0; i<pool1_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(outs[POOL1_OUT_OFFSET + i] - pool1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << outs[POOL1_OUT_OFFSET + i] 
                          << ", expected " << pool1_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // Conv2 Test
    conv_golden<CONV2_OUT_CH, CONV2_IN_CH, KERNEL_SIZE, CONV2_IN_ROWS, CONV2_IN_COLS>(
        pool1_out_ref, conv2_out_ref, conv2_weight, conv2_bias);
    errors = 0;
    max_diff = 0.0f;
    const int conv2_output_size = NUM_CONV2_OUTS;

    std::cout << "conv2_out = [" << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << outs[CONV2_OUT_OFFSET + i * (CONV2_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Conv2 error indexes: ";
    for(int i=0; i<conv2_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(outs[CONV2_OUT_OFFSET + i] - conv2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << outs[CONV2_OUT_OFFSET + i] 
                          << ", expected " << conv2_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // Pool2 Test
    pool_golden<2, 2, CONV2_OUT_CH, CONV2_OUT_ROWS, CONV2_OUT_COLS>(conv2_out_ref, pool2_out_ref);
    errors = 0;
    max_diff = 0.0f;
    const int pool2_output_size = NUM_POOL2_OUTS;

    std::cout << "pool2_out = [" << std::endl;
    for(int i=0; i<4; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << outs[POOL2_OUT_OFFSET + i * (CONV2_OUT_COLS/2) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Pool2 error indexes: ";
    for(int i=0; i<pool2_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(outs[POOL2_OUT_OFFSET + i] - pool2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << outs[POOL2_OUT_OFFSET + i] 
                          << ", expected " << pool2_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // FC1 Test
    fc_golden<FC1_IN_DIM, FC1_OUT_DIM>(pool2_out_ref, fc1_out_ref, fc1_weight, fc1_bias, true);
    errors = 0;
    max_diff = 0.0f;
    const int fc1_output_size = NUM_FC1_OUTS;

    std::cout << "fc1_out = [";
    for(int i=0; i<10; i++) {
        std::cout << outs[FC1_OUT_OFFSET + i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC1 error indexes: ";
    for(int i=0; i<fc1_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(outs[FC1_OUT_OFFSET + i] - fc1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << outs[FC1_OUT_OFFSET + i] 
                          << ", expected " << fc1_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // FC2 Test
    fc_golden<FC2_IN_DIM, FC2_OUT_DIM>(fc1_out_ref, fc2_out_ref, fc2_weight, fc2_bias, true);
    errors = 0;
    max_diff = 0.0f;
    const int fc2_output_size = NUM_FC2_OUTS;

    std::cout << "fc2_out = [";
    for(int i=0; i<10; i++) {
        std::cout << outs[FC2_OUT_OFFSET + i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC2 error indexes: ";
    for(int i=0; i<fc2_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(outs[FC2_OUT_OFFSET + i] - fc2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << outs[FC2_OUT_OFFSET + i] 
                          << ", expected " << fc2_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // FC3 Test
    fc_golden<FC3_IN_DIM, FC3_OUT_DIM>(fc2_out_ref, fc3_out_ref, fc3_weight, fc3_bias, false);
    errors = 0;
    max_diff = 0.0f;
    const int fc3_output_size = NUM_FC3_OUTS;

    std::cout << "fc3_out = [";
    for(int i=0; i<10; i++) {
        std::cout << outs[FC3_OUT_OFFSET + i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC3 error indexes: ";
    for(int i=0; i<fc3_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(outs[FC3_OUT_OFFSET + i] - fc3_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << outs[FC3_OUT_OFFSET + i] 
                          << ", expected " << fc3_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    std::cout << "Total errors: " << global_errors << std::endl;
    std::cout << "Test completed\n";
    return global_errors;
}