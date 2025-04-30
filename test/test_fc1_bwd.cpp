#include <iostream>
#include <cmath>
#include <vector>
#include "lenet5/fc1_bwd.h"
#include "test_utils.h"

#define IN_DIM 256
#define OUT_DIM 120

void print_array(const float* arr, int size, const std::string& name) {
    std::cout << name << " (first 10 elements): ";
    for(int i=0; i < std::min(10, size); i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

#ifndef __SYNTHESIS__
int main() {
    // Allocate arrays
    float in_activation[IN_DIM];
    float grads[OUT_DIM];
    float in_weight[IN_DIM*OUT_DIM];
    float dX[IN_DIM];
    float dW[IN_DIM*OUT_DIM];
    float dB[OUT_DIM];

    // Initialize weights with a deterministic pattern
    for(int i=0; i<IN_DIM; i++) {
        in_activation[i] = 0.01f * i;
    }

    // Initialize bias values
    for(int j=0; j<OUT_DIM; j++) {
        grads[j] = j * 0.1f;
    }

    for(int i=0; i<IN_DIM; i++) {
        for(int j=0; j<OUT_DIM; j++) {
            in_weight[i*OUT_DIM + j] = 0.02f * (i+j);
        }
    }

    // Print some input values for verification
    print_array(in_activation, IN_DIM, "Input");
    print_array(grads, OUT_DIM, "Weights");
    print_array(in_weight, IN_DIM*OUT_DIM, "Bias");

    // Run the FC implementation being tested
    fc1_bwd(in_activation, grads, in_weight, dX, dW, dB);

    float dX_golden[IN_DIM];
    float dW_golden[IN_DIM*OUT_DIM];
    float dB_golden[OUT_DIM];

    // Run the golden reference implementation
    fc_bwd_golden<IN_DIM, OUT_DIM>(in_activation, grads, in_weight, dX_golden, dW_golden, dB_golden, false);

    // Compare results
    int total_errs = 0;
    int errs = 0;
    float max_diff = 0.0f;
    
    // Verify dX
    int dX_length = IN_DIM;
    for(int j=0; j<dX_length; j++) {
        float diff = std::fabs(dX[j] - dX_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << dX[j] 
                          << ", expected " << dX_golden[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "dX Output (first 10 elements): ";
    for(int j=0; j<std::min(10, dX_length); j++) {
        std::cout << dX[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "dX_golden (first 10 elements): ";
    for(int j=0; j<std::min(10, dX_length); j++) {
        std::cout << dX_golden[j] << " ";
    }
    std::cout << std::endl;
    std::cout << errs << " errors" << std::endl;
    std::cout << std::endl << std::endl;

    // Verify dW
    errs = 0;
    int dW_length = IN_DIM*OUT_DIM;
    for(int j=0; j<dW_length; j++) {
        float diff = std::fabs(dW[j] - dW_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << dW[j] 
                          << ", expected " << dW_golden[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "dW Output (first 10 elements): ";
    for(int j=0; j<std::min(10, dW_length); j++) {
        std::cout << dW[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "dW_golden (first 10 elements): ";
    for(int j=0; j<std::min(10, dW_length); j++) {
        std::cout << dW_golden[j] << " ";
    }
    std::cout << std::endl;
    std::cout << errs << " errors" << std::endl;
    std::cout << std::endl << std::endl;

    // Verify dB
    errs = 0;
    int dB_length = OUT_DIM;
    for(int j=0; j<dB_length; j++) {
        float diff = std::fabs(dB[j] - dB_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << dX[j] 
                          << ", expected " << dB_golden[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "dB Output (first 10 elements): ";
    for(int j=0; j<std::min(10, dB_length); j++) {
        std::cout << dB[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "dB_golden (first 10 elements): ";
    for(int j=0; j<std::min(10, dB_length); j++) {
        std::cout << dB_golden[j] << " ";
    }
    std::cout << std::endl;
    std::cout << errs << " errors" << std::endl;
    std::cout << std::endl << std::endl;

    std::cout << "Maximum difference: " << max_diff << std::endl;
    std::cout << "FC3 Backpropagation layer Test: " << (total_errs ? "FAIL" : "PASS") 
              << " (" << total_errs << " errors " << "out of " << dX_length + dW_length + dB_length << ")" << std::endl;

    return total_errs;
}
#endif