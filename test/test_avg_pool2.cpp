#include <iostream>
#include <cmath>
#include "hls_math.h"
// #include "weights_bias.h"
// #include "data_type.hpp"
#include "lenet5/avg_pool2.h" // Using the updated avg_pool function

#define IN_CHANNELS 16
#define IN_ROWS 8
#define IN_COLS 8
#define POOL_SIZE 2
#define STRIDE 2
#define OUT_ROWS ((IN_ROWS - POOL_SIZE) / STRIDE + 1)
#define OUT_COLS ((IN_COLS - POOL_SIZE) / STRIDE + 1)

int main() {
    // Allocate memory for input and output data
    data_ap_fixed_t in_data[IN_CHANNELS * IN_ROWS * IN_COLS];
    data_ap_fixed_t out_data[IN_CHANNELS * OUT_ROWS * OUT_COLS];
    data_ap_fixed_t out_data_ref[IN_CHANNELS * OUT_ROWS * OUT_COLS];
    
    // Initialize input data with a pattern that makes it easy to verify results
    for(int ch=0; ch < IN_CHANNELS; ch++) {
        for(int i=0; i < IN_ROWS; i++) {
            for(int j=0; j < IN_COLS; j++) {
                in_data[ch*IN_ROWS*IN_COLS + i*IN_COLS + j] = ch*100 + i + j;
            }
        }
    }

    // Calculate reference output (ground truth)
    for(int ch=0; ch < IN_CHANNELS; ch++) {
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
                out_data_ref[ch*OUT_ROWS*OUT_COLS + out_r*OUT_COLS + out_c] = avg;
            }
        }
    }

    // Print sample of input data for verification
    std::cout << "Sample input data (first channel, first 5x5):" << std::endl;
    for(int i=0; i < 5 && i < IN_ROWS; i++) {
        for(int j=0; j < 5 && j < IN_COLS; j++) {
            std::cout << in_data[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Run the average pooling function
    avg_pool2(in_data, out_data);
    
    // Print sample of output data for verification
    std::cout << "Sample output data (first channel, first 3x3):" << std::endl;
    for(int i=0; i < OUT_ROWS; i++) {
        for(int j=0; j < OUT_COLS; j++) {
            std::cout << out_data[i*OUT_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Print expected output for comparison
    std::cout << "Expected output data (first channel, first 3x3):" << std::endl;
    for(int i=0; i < OUT_ROWS; i++) {
        for(int j=0; j < OUT_COLS; j++) {
            std::cout << out_data_ref[i*OUT_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    int errors = 0;
    const data_ap_fixed_t EPSILON = 1e-5f; // Allow for small floating-point differences
    
    // Check for errors
    std::cout << "\nChecking for errors..." << std::endl;
    for(int ch=0; ch < IN_CHANNELS; ch++) {
        for(int i=0; i < OUT_ROWS; i++) {
            for(int j=0; j < OUT_COLS; j++) {
                int idx = ch*OUT_ROWS*OUT_COLS + i*OUT_COLS + j;
                // Use epsilon comparison for floating point
                if(std::abs(out_data[idx] - out_data_ref[idx]) > EPSILON) {
                    errors++;
                    if(errors <= 10) { // Limit output to first 10 errors
                        std::cout << "Error at ch=" << ch << ", row=" << i << ", col=" << j 
                                  << ": expected " << out_data_ref[idx] 
                                  << ", got " << out_data[idx] << std::endl;
                    }
                }
            }
        }
    }
    
    if(errors == 0) {
        std::cout << "All tests passed successfully!" << std::endl;
    } else {
        std::cout << "Total errors: " << errors << std::endl;
    }

    return (errors > 0) ? 1 : 0;
}