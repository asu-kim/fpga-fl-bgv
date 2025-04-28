#include <iostream>
#include <cmath>
#include "lenet5/forward_path.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "test_utils.h"

int main() {
    float in_data[784];
    float conv1_out[CONV1_OUT_CH * (CONV1_IN_ROWS - KERNEL_SIZE + 1) * (CONV1_IN_COLS - KERNEL_SIZE + 1)];
    float conv1_out_ref[CONV1_OUT_CH * (CONV1_IN_ROWS - KERNEL_SIZE + 1) * (CONV1_IN_COLS - KERNEL_SIZE + 1)];
    float conv1_weight[CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv1_bias[CONV1_OUT_CH];

    for(int i = 0; i < CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS; i++) {
        // in_data[i] = 0.001f * i;
        in_data[i] = SAMPLE_INPUT[i];
    }

    for(int i = 0; i < CONV1_OUT_CH*CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV1_OUT_CH) {
            conv1_bias[i] = CONV1_BIAS_FP32_DATA[i];
            // conv1_bias[i] = CONV1_BIAS_INT8_DATA[i];
        }
        conv1_weight[i] = CONV1_WEIGHT_FP32_DATA[i];
        // conv1_weight[i] = CONV1_WEIGHT_INT8_DATA[i];
    }

    float pool1_out[CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS];
    float pool1_out_ref[CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS];

    float conv2_out[CONV2_OUT_CH * (CONV2_IN_ROWS - KERNEL_SIZE + 1) * (CONV2_IN_COLS - KERNEL_SIZE + 1)];
    float conv2_out_ref[CONV2_OUT_CH * (CONV2_IN_ROWS - KERNEL_SIZE + 1) * (CONV2_IN_COLS - KERNEL_SIZE + 1)];
    float conv2_weight[CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE];
    float conv2_bias[CONV2_OUT_CH];

    for(int i = 0; i < CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV2_OUT_CH) {
            conv2_bias[i] = CONV2_BIAS_FP32_DATA[i];
        }
        conv2_weight[i] = CONV2_WEIGHT_FP32_DATA[i];
    }

    float pool2_out[FC1_IN_DIM];
    float pool2_out_ref[FC1_IN_DIM];

    float fc1_out[FC1_OUT_DIM];
    float fc1_out_ref[FC1_OUT_DIM];
    float fc1_weight[FC1_IN_DIM*FC1_OUT_DIM];
    float fc1_bias[FC1_OUT_DIM];

    for(int i = 0; i < FC1_IN_DIM*FC1_OUT_DIM; i++) {
        if (i < FC1_OUT_DIM) {
            fc1_bias[i] = FC1_BIAS_FP32_DATA[i];
        }
        fc1_weight[i] = FC1_WEIGHT_FP32_DATA[i];
    }

    float fc2_out[FC2_OUT_DIM];
    float fc2_out_ref[FC2_OUT_DIM];
    float fc2_weight[FC2_IN_DIM*FC2_OUT_DIM];
    float fc2_bias[FC2_OUT_DIM];

    for(int i = 0; i < FC2_IN_DIM*FC2_OUT_DIM; i++) {
        if (i < FC2_OUT_DIM) {
            fc2_bias[i] = FC2_BIAS_FP32_DATA[i];
        }
        fc2_weight[i] = FC2_WEIGHT_FP32_DATA[i];
    }

    float fc3_out[FC3_OUT_DIM];
    float fc3_out_ref[FC3_OUT_DIM];
    float fc3_weight[FC3_IN_DIM*FC3_OUT_DIM];
    float fc3_bias[FC3_OUT_DIM];

    for(int i = 0; i < FC3_IN_DIM*FC3_OUT_DIM; i++) {
        if (i < FC3_OUT_DIM) {
            fc3_bias[i] = FC3_BIAS_FP32_DATA[i];
        }
        fc3_weight[i] = FC3_WEIGHT_FP32_DATA[i];
    }

    // run forward path
    forward_path(
        in_data,
        conv1_weight,
        conv1_bias,
        conv1_out,
        pool1_out,
        conv2_weight,
        conv2_bias,
        conv2_out,   
        pool2_out,
        fc1_weight,
        fc1_bias,
        fc1_out,
        fc2_weight,
        fc2_bias,
        fc2_out,
        fc3_weight,
        fc3_bias,
        fc3_out
    );

    // Conv1 Test
    conv_golden<CONV1_OUT_CH, CONV1_IN_CH, KERNEL_SIZE, CONV1_IN_ROWS, CONV1_IN_COLS>(in_data, conv1_out_ref, conv1_weight, conv1_bias);

    int global_errors = 0;
    int errors = 0;
    float max_diff = 0.0f;
    const int conv1_output_size = (CONV1_IN_ROWS - KERNEL_SIZE + 1) * (CONV1_IN_COLS - KERNEL_SIZE + 1) * CONV1_OUT_CH;

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
        float diff = std::fabs(conv1_out[i] - conv1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
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
    pool_golden<2, 2, CONV1_OUT_CH, CONV1_OUT_ROWS, CONV1_OUT_COLS>(conv1_out_ref, pool1_out_ref);
    errors = 0;
    max_diff = 0.0f;
    const int pool1_output_size = CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS;

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
        float diff = std::fabs(pool1_out[i] - pool1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
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
    conv_golden<CONV2_OUT_CH, CONV2_IN_CH, KERNEL_SIZE, CONV2_IN_ROWS, CONV2_IN_COLS>(pool1_out_ref, conv2_out_ref, conv2_weight, conv2_bias);
    errors = 0;
    max_diff = 0.0f;
    const int conv2_output_size = (CONV2_IN_ROWS - KERNEL_SIZE + 1) * (CONV2_IN_COLS - KERNEL_SIZE + 1) * CONV2_OUT_CH;

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
        float diff = std::fabs(conv2_out[i] - conv2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
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
    pool_golden<2, 2, CONV2_OUT_CH, CONV2_OUT_ROWS, CONV2_OUT_COLS>(conv2_out_ref, pool2_out_ref);
    errors = 0;
    max_diff = 0.0f;
    const int pool2_output_size = FC1_IN_DIM;

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
        float diff = std::fabs(pool2_out[i] - pool2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
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
    fc_golden<FC1_IN_DIM, FC1_OUT_DIM>(pool2_out_ref, fc1_out_ref, fc1_weight, fc1_bias, true);
    errors = 0;
    max_diff = 0.0f;
    const int fc1_output_size = FC1_OUT_DIM;

    std::cout << "fc1_out = [";
    for(int i=0; i<10; i++) {
        std::cout << fc1_out[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC1 error indexes: ";
    for(int i=0; i<fc1_output_size; i++) {
        float diff = std::fabs(fc1_out[i] - fc1_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
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
    fc_golden<FC2_IN_DIM, FC2_OUT_DIM>(fc1_out_ref, fc2_out_ref, fc2_weight, fc2_bias, true);
    errors = 0;
    max_diff = 0.0f;
    const int fc2_output_size = FC2_OUT_DIM;

    std::cout << "fc2_out = [";
    for(int i=0; i<10; i++) {
        std::cout << fc2_out[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC2 error indexes: ";
    for(int i=0; i<fc2_output_size; i++) {
        float diff = std::fabs(fc2_out[i] - fc2_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
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
    fc_golden<FC3_IN_DIM, FC3_OUT_DIM>(fc2_out_ref, fc3_out_ref, fc3_weight, fc3_bias, false);
    errors = 0;
    max_diff = 0.0f;
    const int fc3_output_size = FC3_OUT_DIM;

    std::cout << "fc3_out = [";
    for(int i=0; i<10; i++) {
        std::cout << fc3_out[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC3 error indexes: ";
    for(int i=0; i<fc3_output_size; i++) {
        float diff = std::fabs(fc3_out[i] - fc3_out_ref[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
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