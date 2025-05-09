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

    for(int i = 0; i < CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS; i++) {
        in_data[i] = SAMPLE_INPUT[i];
    }

    // Load Conv1 weights and biases
    for(int i = 0; i < CONV1_OUT_CH*CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV1_OUT_CH) {
            biases[CONV1_BIAS_OFFSET + i] = CONV1_BIAS_FP32_DATA[i];
        }
        weights[CONV1_WEIGHT_OFFSET + i] = CONV1_WEIGHT_FP32_DATA[i];
    }

    // Load Conv2 weights and biases
    for(int i = 0; i < CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV2_OUT_CH) {
            biases[CONV2_BIAS_OFFSET + i] = CONV2_BIAS_FP32_DATA[i];
        }
        weights[CONV2_WEIGHT_OFFSET + i] = CONV2_WEIGHT_FP32_DATA[i];
    }

    // Load FC1 weights and biases
    for(int i = 0; i < FC1_IN_DIM*FC1_OUT_DIM; i++) {
        if (i < FC1_OUT_DIM) {
            biases[FC1_BIAS_OFFSET + i] = FC1_BIAS_FP32_DATA[i];
        }
        weights[FC1_WEIGHT_OFFSET + i] = FC1_WEIGHT_FP32_DATA[i];
    }

    // Load FC2 weights and biases
    for(int i = 0; i < FC2_IN_DIM*FC2_OUT_DIM; i++) {
        if (i < FC2_OUT_DIM) {
            biases[FC2_BIAS_OFFSET + i] = FC2_BIAS_FP32_DATA[i];
        }
        weights[FC2_WEIGHT_OFFSET + i] = FC2_WEIGHT_FP32_DATA[i];
    }

    // Load FC3 weights and biases
    for(int i = 0; i < FC3_IN_DIM*FC3_OUT_DIM; i++) {
        if (i < FC3_OUT_DIM) {
            biases[FC3_BIAS_OFFSET + i] = FC3_BIAS_FP32_DATA[i];
        }
        weights[FC3_WEIGHT_OFFSET + i] = FC3_WEIGHT_FP32_DATA[i];
    }

    // Run forward path with consolidated arrays
    forward_path(
        in_data,
        weights,
        biases,
        outs
    );

    data_ap_fixed_t outs_ref[TOTAL_OUTS_SIZE];
    forward_golden(
        in_data,
        weights,
        biases,
        outs_ref
    );

    data_ap_fixed_t* conv1_out = &outs[CONV1_OUT_OFFSET];
    data_ap_fixed_t* pool1_out = &outs[POOL1_OUT_OFFSET];
    data_ap_fixed_t* conv2_out = &outs[CONV2_OUT_OFFSET];
    data_ap_fixed_t* pool2_out = &outs[POOL2_OUT_OFFSET];
    data_ap_fixed_t* fc1_out = &outs[FC1_OUT_OFFSET];
    data_ap_fixed_t* fc2_out = &outs[FC2_OUT_OFFSET];
    data_ap_fixed_t* fc3_out = &outs[FC3_OUT_OFFSET];

    data_ap_fixed_t* conv1_out_ref = &outs_ref[CONV1_OUT_OFFSET];
    data_ap_fixed_t* pool1_out_ref = &outs_ref[POOL1_OUT_OFFSET];
    data_ap_fixed_t* conv2_out_ref = &outs_ref[CONV2_OUT_OFFSET];
    data_ap_fixed_t* pool2_out_ref = &outs_ref[POOL2_OUT_OFFSET];
    data_ap_fixed_t* fc1_out_ref = &outs_ref[FC1_OUT_OFFSET];
    data_ap_fixed_t* fc2_out_ref = &outs_ref[FC2_OUT_OFFSET];
    data_ap_fixed_t* fc3_out_ref = &outs_ref[FC3_OUT_OFFSET];

    int global_errors = 0;

    // Verify each layer

    int errors = 0;
    data_ap_fixed_t max_diff = 0.0f;
    const int conv1_output_size = NUM_CONV1_OUTS;

    std::cout << "conv1_out = [" << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << conv1_out[i * (CONV1_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;
    
    std::cout << "Conv1 error indexes: ";
    for(int i=0; i<conv1_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(conv1_out[i] - conv1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << conv1_out[i] 
                          << ", expected " << conv1_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // Pool1 Test
    errors = 0;
    max_diff = 0.0f;
    const int pool1_output_size = NUM_POOL1_OUTS;

    std::cout << "pool1_out = [" << std::endl;
    for(int i=0; i<3; i++) {
        for(int j = 0; j < 3; j++) {
            std::cout << pool1_out[i * CONV2_IN_COLS + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Pool1 error indexes: ";
    for(int i=0; i<pool1_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(pool1_out[i] - pool1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << pool1_out[i] 
                          << ", expected " << pool1_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // Conv2 Test
    errors = 0;
    max_diff = 0.0f;
    const int conv2_output_size = NUM_CONV2_OUTS;

    std::cout << "conv2_out = [" << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << conv2_out[i * (CONV2_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Conv2 error indexes: ";
    for(int i=0; i<conv2_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(conv2_out[i] - conv2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << conv2_out[i] 
                          << ", expected " << conv2_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // Pool2 Test
    errors = 0;
    max_diff = 0.0f;
    const int pool2_output_size = NUM_POOL2_OUTS;

    std::cout << "pool2_out = [" << std::endl;
    for(int i=0; i<4; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << pool2_out[i * (CONV2_OUT_COLS/2) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Pool2 error indexes: ";
    for(int i=0; i<pool2_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(pool2_out[i] - pool2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << pool2_out[i] 
                          << ", expected " << pool2_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // FC1 Test
    errors = 0;
    max_diff = 0.0f;
    const int fc1_output_size = NUM_FC1_OUTS;

    std::cout << "fc1_out = [";
    for(int i=0; i<10; i++) {
        std::cout << fc1_out[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC1 error indexes: ";
    for(int i=0; i<fc1_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(fc1_out[i] - fc1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << fc1_out[i] 
                          << ", expected " << fc1_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // FC2 Test
    errors = 0;
    max_diff = 0.0f;
    const int fc2_output_size = NUM_FC2_OUTS;

    std::cout << "fc2_out = [";
    for(int i=0; i<10; i++) {
        std::cout << fc2_out[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC2 error indexes: ";
    for(int i=0; i<fc2_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(fc2_out[i] - fc2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << fc2_out[i] 
                          << ", expected " << fc2_out_ref[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl;

    // FC3 Test
    errors = 0;
    max_diff = 0.0f;
    const int fc3_output_size = NUM_FC3_OUTS;

    std::cout << "fc3_out = [";
    for(int i=0; i<10; i++) {
        std::cout << fc3_out[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC3 error indexes: ";
    for(int i=0; i<fc3_output_size; i++) {
        data_ap_fixed_t diff = hls::fabs(fc3_out[i] - fc3_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << fc3_out[i] 
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