#include <iostream>
#include <cmath>
#include <vector>
#include "lenet5/fc1.h" // Make sure this matches your FC implementation header file

#define IN_DIM 256
#define OUT_DIM 120

// Golden reference implementation for FC layer
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

void print_array(const data_ap_fixed_t* arr, int size, const std::string& name) {
    std::cout << name << " (first 10 elements): ";
    for(int i=0; i < std::min(10, size); i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

#ifndef __SYNTHESIS__
int main() {
    // Allocate arrays
    data_ap_fixed_t in_data[IN_DIM];
    data_ap_fixed_t out_data[OUT_DIM];
    data_ap_fixed_t golden_output[OUT_DIM];
    data_ap_fixed_t weight[IN_DIM*OUT_DIM]; // 1D array as expected by fc function
    data_ap_fixed_t bias[OUT_DIM];
    bool use_relu = true;

    // Initialize weights with a deterministic pattern
    for(int i=0; i<IN_DIM; i++) {
        for(int j=0; j<OUT_DIM; j++) {
            weight[i*OUT_DIM + j] = 0.01f * (i+j);
        }
    }

    // Initialize bias values
    for(int j=0; j<OUT_DIM; j++) {
        bias[j] = j * 0.1f;
    }

    // Initialize input data
    for(int i=0; i<IN_DIM; i++) {
        in_data[i] = i * 0.5f;
    }

    // Print some input values for verification
    print_array(in_data, IN_DIM, "Input");
    print_array(weight, IN_DIM*OUT_DIM, "Weights");
    print_array(bias, OUT_DIM, "Bias");

    // Run the FC implementation being tested
    fc1(in_data, out_data, weight, bias, use_relu);

    // Run the golden reference implementation
    fc_golden(in_data, golden_output, weight, bias, use_relu);

    // Compare results
    int errs = 0;
    data_ap_fixed_t max_diff = 0.0f;
    
    for(int j=0; j<OUT_DIM; j++) {
        data_ap_fixed_t diff = std::fabs(out_data[j] - golden_output[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << out_data[j] 
                          << ", expected " << golden_output[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    // Print some output for verification
    std::cout << "FC Implementation Output (first 10 elements): ";
    for(int j=0; j<std::min(10, OUT_DIM); j++) {
        std::cout << out_data[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "Golden Reference Output (first 10 elements): ";
    for(int j=0; j<std::min(10, OUT_DIM); j++) {
        std::cout << golden_output[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "Maximum difference: " << max_diff << std::endl;
    std::cout << "FC Layer Test: " << (errs ? "FAIL" : "PASS") 
              << " (" << errs << " errors out of " << OUT_DIM << " outputs)" << std::endl;

    return errs;
}
#endif