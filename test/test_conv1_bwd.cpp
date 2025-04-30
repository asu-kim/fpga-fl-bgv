#include <iostream>
#include <cmath>
#include "lenet5/conv2d_bwd.h"
#include "lenet5/conv1_bwd.h"
#include "test_utils.h"

#define OUT_C 6
#define IN_C 1
#define KERNEL_SIZE 5
#define IN_ROW 28
#define IN_COL 28
#define OUT_ROW (IN_ROW-KERNEL_SIZE+1)
#define OUT_COL (IN_COL-KERNEL_SIZE+1)

void print_array(const float* arr, int size, const std::string& name) {
    std::cout << name << " (first 10 elements): ";
    for(int i=0; i < std::min(10, size); i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // hls::stream<float> in_stream, out_stream;

    float in_activation[IN_C * IN_ROW * IN_COL];
    float grads[OUT_C * OUT_ROW * OUT_COL];
    float in_weight[OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE];
    float out_grads[IN_C * IN_ROW * IN_COL];
    float dW[OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE];
    float dB[OUT_C];

    float out_grads_golden[IN_C * IN_ROW * IN_COL];
    float dW_golden[OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE];
    float dB_golden[OUT_C];

    for(int i = 0; i < IN_C * IN_ROW * IN_COL; i++) {
        in_activation[i] = 0.01f * i;
    }

    for(int i = 0; i < OUT_C; i++) {
        for(int j = 0; j < OUT_ROW; j++) {
            for(int k = 0; k < OUT_COL; k++) {
                int idx = i * OUT_ROW*OUT_COL + j*OUT_COL + k;
                grads[idx] = 0.01f * (i + j + k);
            }
        }
    }

    for(int i = 0; i < OUT_C; i++) {
        for(int j = 0; j < IN_C; j++) {
            for(int k = 0; k < KERNEL_SIZE*KERNEL_SIZE; k++) {
                int idx = i * OUT_ROW*OUT_COL + j*OUT_COL + k;
                grads[idx] = 0.02f * (i + j + k);
            }
        }
    }


    // Print some input values for verification
    print_array(in_activation, IN_C * IN_ROW * IN_COL, "Input");
    print_array(grads, OUT_C * OUT_ROW * OUT_COL, "Weights");
    print_array(in_weight, OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE, "Bias");

    // run conv2d_bwd
    conv1_bwd(in_activation, grads, in_weight, out_grads, dW, dB);
    conv_bwd_golden<OUT_C, IN_C, KERNEL_SIZE, IN_ROW, IN_COL>(in_activation, grads, in_weight, out_grads_golden, dW_golden, dB_golden);

    // Compare results
    int total_errs = 0;
    int errs = 0;
    float max_diff = 0.0f;
    
    // Verify dX
    int out_grads_length = IN_C * IN_ROW * IN_COL;
    for(int j=0; j<out_grads_length; j++) {
        float diff = std::fabs(out_grads[j] - out_grads_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << out_grads[j] 
                          << ", expected " << out_grads_golden[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "out_grads Output (first 10 elements): ";
    for(int j=0; j<std::min(10, out_grads_length); j++) {
        std::cout << out_grads[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "out_grads_golden (first 10 elements): ";
    for(int j=0; j<std::min(10, out_grads_length); j++) {
        std::cout << out_grads_golden[j] << " ";
    }
    std::cout << std::endl;
    std::cout << errs << " errors" << std::endl;
    std::cout << std::endl << std::endl;

    // Verify dW
    errs = 0;
    int dW_length = OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE;
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
    int dB_length = OUT_C;
    for(int j=0; j<dB_length; j++) {
        float diff = std::fabs(dB[j] - dB_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << dB[j] 
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
    std::cout << "Conv1 Backpropagation layer Test: " << (total_errs ? "FAIL" : "PASS") 
              << " (" << total_errs << " errors " << "out of " << out_grads_length + dW_length + dB_length << ")" << std::endl;

    return total_errs;
}
