#include <iostream>
#include <cmath>
#include "lenet5/backward_path.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "test_utils.h"

#define lr 1e-3

int main() {
    // Initialize input data and forward pass outputs
    float in_data[CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS];
    float conv1_out[CONV1_OUT_CH * CONV1_OUT_ROWS * CONV1_OUT_COLS];
    float pool1_out[CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS];
    float conv2_out[CONV2_OUT_CH * CONV2_OUT_ROWS * CONV2_OUT_COLS];
    float pool2_out[FC1_IN_DIM];
    float fc1_out[FC1_OUT_DIM];
    float fc2_out[FC2_OUT_DIM];
    float fc3_out[FC3_OUT_DIM];
    float label[FC3_OUT_DIM];
    float loss = 0.0f;
    
    // Initialize weights and biases
    float conv1_weight[CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv1_bias[CONV1_OUT_CH];
    float conv2_weight[CONV2_OUT_CH * CONV2_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv2_bias[CONV2_OUT_CH];
    float fc1_weight[FC1_IN_DIM * FC1_OUT_DIM];
    float fc1_bias[FC1_OUT_DIM];
    float fc2_weight[FC2_IN_DIM * FC2_OUT_DIM];
    float fc2_bias[FC2_OUT_DIM];
    float fc3_weight[FC3_IN_DIM * FC3_OUT_DIM];
    float fc3_bias[FC3_OUT_DIM];
    
    // Initialize output updated weights and biases
    float conv1_updated_weight[CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv1_updated_bias[CONV1_OUT_CH];
    float conv2_updated_weight[CONV2_OUT_CH * CONV2_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv2_updated_bias[CONV2_OUT_CH];
    float fc1_updated_weight[FC1_IN_DIM * FC1_OUT_DIM];
    float fc1_updated_bias[FC1_OUT_DIM];
    float fc2_updated_weight[FC2_IN_DIM * FC2_OUT_DIM];
    float fc2_updated_bias[FC2_OUT_DIM];
    float fc3_updated_weight[FC3_IN_DIM * FC3_OUT_DIM];
    float fc3_updated_bias[FC3_OUT_DIM];
    
    // Initialize reference updated weights and biases
    float conv1_updated_weight_ref[CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv1_updated_bias_ref[CONV1_OUT_CH];
    float conv2_updated_weight_ref[CONV2_OUT_CH * CONV2_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv2_updated_bias_ref[CONV2_OUT_CH];
    float fc1_updated_weight_ref[FC1_IN_DIM * FC1_OUT_DIM];
    float fc1_updated_bias_ref[FC1_OUT_DIM];
    float fc2_updated_weight_ref[FC2_IN_DIM * FC2_OUT_DIM];
    float fc2_updated_bias_ref[FC2_OUT_DIM];
    float fc3_updated_weight_ref[FC3_IN_DIM * FC3_OUT_DIM];
    float fc3_updated_bias_ref[FC3_OUT_DIM];
    
    // Initialize temporary buffers for gradients
    float out_grad[FC3_OUT_DIM];
    float fc3_dX[FC3_IN_DIM];
    float fc3_dW[FC3_IN_DIM*FC3_OUT_DIM];
    float fc3_dB[FC3_OUT_DIM];
    float fc2_dX[FC2_IN_DIM];
    float fc2_dW[FC2_IN_DIM*FC2_OUT_DIM];
    float fc2_dB[FC2_OUT_DIM];
    float fc1_dX[FC1_IN_DIM];
    float fc1_dW[FC1_IN_DIM*FC1_OUT_DIM];
    float fc1_dB[FC1_OUT_DIM];
    float pool2_dX[CONV2_OUT_CH * CONV2_OUT_ROWS * CONV2_OUT_COLS];
    float conv2_dX[CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS];
    float conv2_dW[CONV2_OUT_CH * CONV2_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv2_dB[CONV2_OUT_CH];
    float pool1_dX[CONV1_OUT_CH * CONV1_OUT_ROWS * CONV1_OUT_COLS];
    float conv1_dX[CONV1_IN_CH * CONV1_IN_ROWS * CONV1_IN_COLS];
    float conv1_dW[CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE];
    float conv1_dB[CONV1_OUT_CH];
    
    // Initialize input data
    for(int i = 0; i < CONV1_IN_CH*CONV1_IN_ROWS*CONV1_IN_COLS; i++) {
        in_data[i] = SAMPLE_INPUT[i];
    }
    
    // Initialize weights and biases
    for(int i = 0; i < CONV1_OUT_CH*CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV1_OUT_CH) {
            conv1_bias[i] = CONV1_BIAS_FP32_DATA[i];
        }
        conv1_weight[i] = CONV1_WEIGHT_FP32_DATA[i];
    }
    
    for(int i = 0; i < CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE; i++) {
        if (i < CONV2_OUT_CH) {
            conv2_bias[i] = CONV2_BIAS_FP32_DATA[i];
        }
        conv2_weight[i] = CONV2_WEIGHT_FP32_DATA[i];
    }
    
    for(int i = 0; i < FC1_IN_DIM*FC1_OUT_DIM; i++) {
        if (i < FC1_OUT_DIM) {
            fc1_bias[i] = FC1_BIAS_FP32_DATA[i];
        }
        fc1_weight[i] = FC1_WEIGHT_FP32_DATA[i];
    }
    
    for(int i = 0; i < FC2_IN_DIM*FC2_OUT_DIM; i++) {
        if (i < FC2_OUT_DIM) {
            fc2_bias[i] = FC2_BIAS_FP32_DATA[i];
        }
        fc2_weight[i] = FC2_WEIGHT_FP32_DATA[i];
    }
    
    for(int i = 0; i < FC3_IN_DIM*FC3_OUT_DIM; i++) {
        if (i < FC3_OUT_DIM) {
            fc3_bias[i] = FC3_BIAS_FP32_DATA[i];
        }
        fc3_weight[i] = FC3_WEIGHT_FP32_DATA[i];
    }
    
    // Initialize label (one-hot encoding for class 7)
    for(int i = 0; i < FC3_OUT_DIM; i++) {
        label[i] = (i == 7) ? 1.0f : 0.0f;
    }
    
    // Run forward pass to get intermediate outputs
    forward_golden(
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
    
    std::cout << "Starting backward path test..." << std::endl;
    std::cout << "Loss: " << loss << std::endl;
    
    // Run backward path
    backward_path(
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
        fc3_out,
        label,
        conv1_updated_weight,
        conv1_updated_bias,
        conv2_updated_weight,
        conv2_updated_bias,
        fc1_updated_weight,
        fc1_updated_bias,
        fc2_updated_weight,
        fc2_updated_bias,
        fc3_updated_weight,
        fc3_updated_bias,
        loss
    );

    float loss_ref = 0.0f;
    // Run backward path
    backward_golden(
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
        fc3_out,
        label,
        conv1_updated_weight_ref,
        conv1_updated_bias_ref,
        conv2_updated_weight_ref,
        conv2_updated_bias_ref,
        fc1_updated_weight_ref,
        fc1_updated_bias_ref,
        fc2_updated_weight_ref,
        fc2_updated_bias_ref,
        fc3_updated_weight_ref,
        fc3_updated_bias_ref,
        loss_ref
    );
    
    // Compare results and report errors
    int global_errors = 0;
    
    // FC3 Weight and Bias Errors
    int errors = 0;
    float max_diff = 0.0f;
    const int fc3_weight_size = FC3_IN_DIM*FC3_OUT_DIM;
    const int fc3_bias_size = FC3_OUT_DIM;
    
    std::cout << "FC3 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << fc3_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;
    
    std::cout << "FC3 Updated Biases = [";
    for(int i = 0; i < FC3_OUT_DIM; i++) {
        std::cout << fc3_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    
    std::cout << "FC3 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < fc3_weight_size; i++) {
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
    for(int i = 0; i < fc3_bias_size; i++) {
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
    const int fc2_weight_size = FC2_IN_DIM*FC2_OUT_DIM;
    const int fc2_bias_size = FC2_OUT_DIM;
    
    std::cout << "FC2 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << fc2_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;
    
    std::cout << "FC2 Updated Biases = [";
    for(int i = 0; i < FC2_OUT_DIM; i++) {
        std::cout << fc2_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    
    std::cout << "FC2 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < fc2_weight_size; i++) {
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
    for(int i = 0; i < fc2_bias_size; i++) {
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
    const int fc1_weight_size = FC1_IN_DIM*FC1_OUT_DIM;
    const int fc1_bias_size = FC1_OUT_DIM;
    
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
    for(int i = 0; i < fc1_weight_size; i++) {
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
    for(int i = 0; i < fc1_bias_size; i++) {
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
    const int conv2_weight_size = CONV2_OUT_CH*CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE;
    const int conv2_bias_size = CONV2_OUT_CH;
    
    std::cout << "Conv2 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << conv2_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;
    
    std::cout << "Conv2 Updated Biases = [";
    for(int i = 0; i < CONV2_OUT_CH; i++) {
        std::cout << conv2_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    
    std::cout << "Conv2 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < conv2_weight_size; i++) {
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
    for(int i = 0; i < conv2_bias_size; i++) {
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
    const int conv1_weight_size = CONV1_OUT_CH*CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE;
    const int conv1_bias_size = CONV1_OUT_CH;
    
    std::cout << "Conv1 Updated Weights sample = [";
    for(int i = 0; i < 5; i++) {
        std::cout << conv1_updated_weight[i] << ", ";
    }
    std::cout << "...]" << std::endl;
    
    std::cout << "Conv1 Updated Biases = [";
    for(int i = 0; i < CONV1_OUT_CH; i++) {
        std::cout << conv1_updated_bias[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    
    std::cout << "Conv1 Weight and Bias Errors:" << std::endl;
    for(int i = 0; i < conv1_weight_size; i++) {
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
    for(int i = 0; i < conv1_bias_size; i++) {
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