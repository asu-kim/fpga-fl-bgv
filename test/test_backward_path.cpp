#include <iostream>
#include <cmath>
#include "lenet5/backward_path.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "test_utils.h"
#include "constants.hpp"

#define lr 1e-3

int main() {
    // Initialize input data
    float in_data[CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS];
    
    // Consolidated arrays
    float weights[TOTAL_WEIGHTS_SIZE];
    float biases[TOTAL_BIASES_SIZE];
    float outputs[TOTAL_OUTS_SIZE];
    float updated_weights[TOTAL_WEIGHTS_SIZE];
    float updated_biases[TOTAL_BIASES_SIZE];
    
    float label[FC3_OUT_DIM];
    float loss = 0.0f;
    
    // Original format updated weights and biases
    float updated_weights_ref[TOTAL_WEIGHTS_SIZE];
    float updated_biases_ref[TOTAL_BIASES_SIZE];
    
    // Initialize input data
    for(int i = 0; i < CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS; i++) {
        in_data[i] = SAMPLE_INPUT[i];
    }
    
    // Initialize weights and biases in both consolidated and original formats
    // Conv1
    for(int i = 0; i < CONV1_OUT_CH*CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        weights[CONV1_WEIGHT_OFFSET + i] = CONV1_WEIGHT_FP32_DATA[i];
        if (i < CONV1_OUT_CH) {
            biases[CONV1_BIAS_OFFSET + i] = CONV1_BIAS_FP32_DATA[i];
        }
    }
    
    // Conv2
    for(int i = 0; i < CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        weights[CONV2_WEIGHT_OFFSET + i] = CONV2_WEIGHT_FP32_DATA[i];
        if (i < CONV2_OUT_CH) {
            biases[CONV2_BIAS_OFFSET + i] = CONV2_BIAS_FP32_DATA[i];
        }
    }
    
    // FC1
    for(int i = 0; i < FC1_IN_DIM*FC1_OUT_DIM; i++) {
        weights[FC1_WEIGHT_OFFSET + i] = FC1_WEIGHT_FP32_DATA[i];
        if (i < FC1_OUT_DIM) {
            biases[FC1_BIAS_OFFSET + i] = FC1_BIAS_FP32_DATA[i];
        }
    }
    
    // FC2
    for(int i = 0; i < FC2_IN_DIM*FC2_OUT_DIM; i++) {
        weights[FC2_WEIGHT_OFFSET + i] = FC2_WEIGHT_FP32_DATA[i];
        if (i < FC2_OUT_DIM) {
            biases[FC2_BIAS_OFFSET + i] = FC2_BIAS_FP32_DATA[i];
        }
    }
    
    // FC3
    for(int i = 0; i < FC3_IN_DIM*FC3_OUT_DIM; i++) {
        weights[FC3_WEIGHT_OFFSET + i] = FC3_WEIGHT_FP32_DATA[i];
        if (i < FC3_OUT_DIM) {
            biases[FC3_BIAS_OFFSET + i] = FC3_BIAS_FP32_DATA[i];
        }
    }
    
    // Initialize label (one-hot encoding for class 7)
    for(int i = 0; i < FC3_OUT_DIM; i++) {
        label[i] = (i == 7) ? 1.0f : 0.0f;
    }
    
    // Run forward pass with original arrays to get intermediate outputs
    forward_golden(
        in_data,
        weights,
        biases,
        outputs
    );
    
    std::cout << "Starting backward path test..." << std::endl;
    std::cout << "Loss: " << loss << std::endl;
    
    // Run backward path with consolidated arrays
    backward_path(
        in_data,
        weights,
        biases,
        outputs,
        label,
        updated_weights,
        updated_biases,
        loss
    );

    float loss_ref = 0.0f;
    // Run backward path with original format arrays for reference
    backward_golden(
        in_data,
        weights,
        biases,
        outputs,
        label,
        updated_weights_ref,
        updated_biases_ref,
        loss_ref
    );
    
    // Compare results and report errors
    int global_errors = 0;

    // Extract pointers to each layer's updated weights and biases from consolidated arrays
    // For the implementation
    const float* conv1_updated_weight = &updated_weights[CONV1_WEIGHT_OFFSET];
    const float* conv1_updated_bias = &updated_biases[CONV1_BIAS_OFFSET];
    const float* conv2_updated_weight = &updated_weights[CONV2_WEIGHT_OFFSET];
    const float* conv2_updated_bias = &updated_biases[CONV2_BIAS_OFFSET];
    const float* fc1_updated_weight = &updated_weights[FC1_WEIGHT_OFFSET];
    const float* fc1_updated_bias = &updated_biases[FC1_BIAS_OFFSET];
    const float* fc2_updated_weight = &updated_weights[FC2_WEIGHT_OFFSET];
    const float* fc2_updated_bias = &updated_biases[FC2_BIAS_OFFSET];
    const float* fc3_updated_weight = &updated_weights[FC3_WEIGHT_OFFSET];
    const float* fc3_updated_bias = &updated_biases[FC3_BIAS_OFFSET];

    // For the reference implementation
    const float* conv1_updated_weight_ref = &updated_weights_ref[CONV1_WEIGHT_OFFSET];
    const float* conv1_updated_bias_ref = &updated_biases_ref[CONV1_BIAS_OFFSET];
    const float* conv2_updated_weight_ref = &updated_weights_ref[CONV2_WEIGHT_OFFSET];
    const float* conv2_updated_bias_ref = &updated_biases_ref[CONV2_BIAS_OFFSET];
    const float* fc1_updated_weight_ref = &updated_weights_ref[FC1_WEIGHT_OFFSET];
    const float* fc1_updated_bias_ref = &updated_biases_ref[FC1_BIAS_OFFSET];
    const float* fc2_updated_weight_ref = &updated_weights_ref[FC2_WEIGHT_OFFSET];
    const float* fc2_updated_bias_ref = &updated_biases_ref[FC2_BIAS_OFFSET];
    const float* fc3_updated_weight_ref = &updated_weights_ref[FC3_WEIGHT_OFFSET];
    const float* fc3_updated_bias_ref = &updated_biases_ref[FC3_BIAS_OFFSET];

    // FC3 Weight and Bias Errors
    int errors = 0;
    float max_diff = 0.0f;

    std::cout << "FC3 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << fc3_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;

    std::cout << "FC3 Updated Biases = [";
    for(int i = 0; i < NUM_FC3_BIASES; i++) {
        std::cout << fc3_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC3 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < NUM_FC3_WEIGHTS; i++) {
        float diff = std::fabs(fc3_updated_weight[i] - fc3_updated_weight_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at FC3 weight " << i << ": got " << fc3_updated_weight[i]
                        << ", expected " << fc3_updated_weight_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    for(int i = 0; i < NUM_FC3_BIASES; i++) {
        float diff = std::fabs(fc3_updated_bias[i] - fc3_updated_bias_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at FC3 bias " << i << ": got " << fc3_updated_bias[i]
                        << ", expected " << fc3_updated_bias_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << "FC3 errors: " << errors << ", max diff: " << max_diff << std::endl << std::endl;

    // FC2 Weight and Bias Errors
    errors = 0;
    max_diff = 0.0f;

    std::cout << "FC2 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << fc2_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;

    std::cout << "FC2 Updated Biases = [";
    for(int i = 0; i < NUM_FC2_BIASES; i++) {
        std::cout << fc2_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "FC2 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < NUM_FC2_WEIGHTS; i++) {
        float diff = std::fabs(fc2_updated_weight[i] - fc2_updated_weight_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at FC2 weight " << i << ": got " << fc2_updated_weight[i]
                        << ", expected " << fc2_updated_weight_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    for(int i = 0; i < NUM_FC2_BIASES; i++) {
        float diff = std::fabs(fc2_updated_bias[i] - fc2_updated_bias_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at FC2 bias " << i << ": got " << fc2_updated_bias[i]
                        << ", expected " << fc2_updated_bias_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << "FC2 errors: " << errors << ", max diff: " << max_diff << std::endl << std::endl;

    // FC1 Weight and Bias Errors
    errors = 0;
    max_diff = 0.0f;

    std::cout << "FC1 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << fc1_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;

    std::cout << "FC1 Updated Biases sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << fc1_updated_bias[i] << ", ";
    }
    std::cout << "...]" << std::endl << std::endl;

    std::cout << "FC1 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < NUM_FC1_WEIGHTS; i++) {
        float diff = std::fabs(fc1_updated_weight[i] - fc1_updated_weight_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at FC1 weight " << i << ": got " << fc1_updated_weight[i]
                        << ", expected " << fc1_updated_weight_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    for(int i = 0; i < NUM_FC1_BIASES; i++) {
        float diff = std::fabs(fc1_updated_bias[i] - fc1_updated_bias_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at FC1 bias " << i << ": got " << fc1_updated_bias[i]
                        << ", expected " << fc1_updated_bias_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << "FC1 errors: " << errors << ", max diff: " << max_diff << std::endl << std::endl;

    // Conv2 Weight and Bias Errors
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Conv2 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << conv2_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;

    std::cout << "Conv2 Updated Biases = [";
    for(int i = 0; i < NUM_CONV2_BIASES; i++) {
        std::cout << conv2_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Conv2 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < NUM_CONV2_WEIGHTS; i++) {
        float diff = std::fabs(conv2_updated_weight[i] - conv2_updated_weight_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at Conv2 weight " << i << ": got " << conv2_updated_weight[i]
                        << ", expected " << conv2_updated_weight_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    for(int i = 0; i < NUM_CONV2_BIASES; i++) {
        float diff = std::fabs(conv2_updated_bias[i] - conv2_updated_bias_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at Conv2 bias " << i << ": got " << conv2_updated_bias[i]
                        << ", expected " << conv2_updated_bias_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << "Conv2 errors: " << errors << ", max diff: " << max_diff << std::endl << std::endl;

    // Conv1 Weight and Bias Errors
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Conv1 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << conv1_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;

    std::cout << "Conv1 Updated Biases = [";
    for(int i = 0; i < NUM_CONV1_BIASES; i++) {
        std::cout << conv1_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "Conv1 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < NUM_CONV1_WEIGHTS; i++) {
        float diff = std::fabs(conv1_updated_weight[i] - conv1_updated_weight_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at Conv1 weight " << i << ": got " << conv1_updated_weight[i]
                        << ", expected " << conv1_updated_weight_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    for(int i = 0; i < NUM_CONV1_BIASES; i++) {
        float diff = std::fabs(conv1_updated_bias[i] - conv1_updated_bias_ref[i]);
        max_diff = std::max(max_diff, diff);
        if(diff > 0.1f) {
            errors++;
            if(errors < 10) {
                std::cout << "Error at Conv1 bias " << i << ": got " << conv1_updated_bias[i]
                        << ", expected " << conv1_updated_bias_ref[i]
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << "Conv1 errors: " << errors << ", max diff: " << max_diff << std::endl << std::endl;

    std::cout << "Total errors: " << global_errors << std::endl;
    std::cout << "Test completed" << std::endl;

    return global_errors;
}