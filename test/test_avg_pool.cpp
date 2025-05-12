#include <iostream>
#include <cmath>
#include "hls_math.h"
#include "weights_bias.h"
#include "data_type.hpp"
#include "lenet5/avg_pool1.h"

#define IN_CHANNELS 6
#define IN_ROWS 24
#define IN_COLS 24
#define OUT_ROWS (IN_ROWS / POOL_SIZE)
#define OUT_COLS (IN_COLS / POOL_SIZE)
#define POOL_SIZE 2

int main() {
    // Allocate memory for input and output data
    data_ap_fixed_t in_data[IN_CHANNELS * IN_ROWS * IN_COLS];
    data_ap_fixed_t out_data[IN_CHANNELS * OUT_ROWS * OUT_COLS];
    data_ap_fixed_t out_data_ref[IN_CHANNELS * OUT_ROWS * OUT_COLS];
    
    // Initialize input data with a pattern that makes it easy to verify results
    for(int ch=0; ch < IN_CHANNELS; ch++) {
        for(int i=0; i < IN_ROWS; i++) {
            for(int j=0; j < IN_COLS; j++) {
                // Use a different value for each channel to verify channel independence
                // Also use position-dependent values to verify proper pooling regions
                // in_data[ch + IN_CHANNELS*(i*IN_COLS + j)] = ch*100 + i + j;
                in_data[ch*IN_ROWS*IN_COLS + i*IN_COLS + j] = ch*100 + i + j;
            }
        }
    }

    // Calculate reference output (ground truth)
    for(int ch=0; ch < IN_CHANNELS; ch++) {
        for(int pr=0; pr < OUT_ROWS; pr++) {
            for(int pc=0; pc < OUT_COLS; pc++) {
                data_ap_fixed_t sum = 0;
                for(int i=0; i < POOL_SIZE; i++) {
                    for(int j=0; j < POOL_SIZE; j++) {
                        // sum += in_data[ch + IN_CHANNELS*((pr*POOL_SIZE+i)*IN_COLS + (pc*POOL_SIZE+j))];
                        sum += in_data[ch*IN_ROWS*IN_COLS + (pr*POOL_SIZE+i)*IN_COLS + (pc*POOL_SIZE+j)];
                    }
                }
                // Calculate average with rounding
                data_ap_fixed_t avg = sum / (POOL_SIZE*POOL_SIZE);
                out_data_ref[ch*OUT_ROWS*OUT_COLS + pr*OUT_COLS + pc] = avg;
            }
        }
    }

    // Print sample of input data for verification
    std::cout << "Sample input data (first channel, first 5x5):" << std::endl;
    for(int i=0; i < IN_ROWS; i++) {
        for(int j=0; j < IN_COLS; j++) {
            std::cout << in_data[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Run the average pooling function
    avg_pool1(in_data, out_data);
    
    // Print sample of input data for verification
    std::cout << "Sample output data (first channel, first 3x3):" << std::endl;
    for(int i=0; i < OUT_ROWS; i++) {
        for(int j=0; j < OUT_COLS; j++) {
            std::cout << out_data[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    int errors = 0;
    
    // Check for errors
    std::cout << "\nChecking for errors..." << std::endl;
    for(int ch=0; ch < IN_CHANNELS; ch++) {
        for(int i=0; i < OUT_ROWS; i++) {
            for(int j=0; j < OUT_COLS; j++) {
                int idx = ch*OUT_ROWS*OUT_COLS + i*OUT_COLS + j;
                if(out_data[idx] != out_data_ref[idx]) {
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
