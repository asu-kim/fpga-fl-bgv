#include <iostream>
#include <cmath>
#include <vector>
#include "lenet5/avg_pool1_bwd.h"
#include "test_utils.h"

#define IN_CHANNELS 6
#define IN_ROWS 24
#define IN_COLS 24
#define POOL_SIZE 2
#define STRIDE 2
#define OUT_ROWS ((IN_ROWS - POOL_SIZE) / STRIDE + 1)
#define OUT_COLS ((IN_COLS - POOL_SIZE) / STRIDE + 1)

void print_array(const float* arr, int size, const std::string& name) {
    std::cout << name << " (first 10 elements): ";
    for(int i=0; i < std::min(10, size); i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Allocate arrays
    float grads[IN_CHANNELS * OUT_ROWS * OUT_COLS];
    float dX[IN_CHANNELS * IN_ROWS * IN_COLS];
    float dX_ref[IN_CHANNELS * IN_ROWS * IN_COLS];

    // Initialize input data with a pattern that makes it easy to verify results
    for(int ch=0; ch < IN_CHANNELS; ch++) {
        for(int i=0; i < OUT_ROWS; i++) {
            for(int j=0; j < OUT_COLS; j++) {
                grads[ch*OUT_ROWS*OUT_COLS + i*OUT_COLS + j] = ch*100 + i + j;
            }
        }
    }

    // Print sample of input data for verification
    std::cout << "Sample grads data (first channel, first 5x5):" << std::endl;
    for(int i=0; i < 5 && i < IN_ROWS; i++) {
        for(int j=0; j < 5 && j < IN_COLS; j++) {
            std::cout << grads[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Run the average pooling function
    avg_pool1_bwd(grads, dX);
    pool_bwd_golden<POOL_SIZE, STRIDE, IN_CHANNELS, IN_ROWS, IN_COLS>(grads, dX_ref);

    // Print sample of output data for verification
    std::cout << "Sample dX data (first channel, first 6x6):" << std::endl;
    for(int i=0; i < 6; i++) {
        for(int j=0; j < 6; j++) {
            std::cout << dX[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Print expected output for comparison
    std::cout << "Expected dX data (first channel, first 6x6):" << std::endl;
    for(int i=0; i < 6; i++) {
        for(int j=0; j < 6; j++) {
            std::cout << dX_ref[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    int errors = 0;
    const float EPSILON = 1e-5f; // Allow for small floating-point differences
    
    // Check for errors
    std::cout << "\nChecking for errors..." << std::endl;
    for(int ch=0; ch < IN_CHANNELS; ch++) {
        for(int i=0; i < IN_ROWS; i++) {
            for(int j=0; j < IN_COLS; j++) {
                int idx = ch*IN_ROWS*IN_COLS + i*IN_COLS + j;
                // Use epsilon comparison for floating point
                if(std::abs(dX[idx] - dX_ref[idx]) > EPSILON) {
                    errors++;
                    if(errors <= 10) { // Limit output to first 10 errors
                        std::cout << "Error at ch=" << ch << ", row=" << i << ", col=" << j 
                                  << ": expected " << dX_ref[idx] 
                                  << ", got " << dX[idx] << std::endl;
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