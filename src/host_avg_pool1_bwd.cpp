/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "cmdlineparser.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <ctime>
#include <limits>

#include "hls_math.h"
#include "weights_bias.h"
#include "constants.hpp"
#include "../test/test_utils.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define IN_C 6
#define IN_ROWS 24
#define IN_COLS 24
#define OUT_ROWS (IN_ROWS / POOL_SIZE)
#define OUT_COLS (IN_COLS / POOL_SIZE)
#define POOL_SIZE 2

int main(int argc, char **argv)
{
    // Command Line Parser
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (binaryFile.empty()) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    // size_t vector_size_bytes = sizeof(float) * DATA_SIZE;
    size_t in_size_bytes = sizeof(float) * IN_C * OUT_ROWS * OUT_COLS;
    size_t out_size_bytes = sizeof(float) * IN_C * IN_ROWS * IN_COLS;

    // Create kernels
    auto avg_pool1_bwd_krnl = xrt::kernel(device, uuid, "avg_pool1_bwd");

    std::cout << "Allocate Buffer in Global Memory\n";

    // Allocate in and out for avg_pool1_bwd
    auto bo_in_data = xrt::bo(device, in_size_bytes, avg_pool1_bwd_krnl.group_id(0));
    auto bo_out_data = xrt::bo(device, out_size_bytes, avg_pool1_bwd_krnl.group_id(1));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<float *>();
    auto bo_out_data_map = bo_out_data.map<float *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_data_map, bo_in_data_map + IN_C * OUT_ROWS * OUT_COLS, 0);
    std::fill(bo_out_data_map, bo_out_data_map + IN_C * IN_ROWS * IN_COLS, 0);

    // Initialize input data with a pattern that makes it easy to verify results
    for(int ch=0; ch < IN_C; ch++) {
        for(int i=0; i < OUT_ROWS; i++) {
            for(int j=0; j < OUT_COLS; j++) {
                // Use a different value for each channel to verify channel independence
                // Also use position-dependent values to verify proper pooling regions
                bo_in_data_map[ch*OUT_ROWS*OUT_COLS + i*OUT_COLS + j] = ch*100 + i + j;
            }
        }
    }

    // Step 4: Sync inputs
    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Sample dX data (first channel, first 6x6):" << std::endl;
    for(int i=0; i < 6; i++) {
        for(int j=0; j < 6; j++) {
            std::cout << bo_in_data_map[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Run convolution
    std::cout << "Running avg pooling backward\n";
    auto run = avg_pool1_bwd_krnl(bo_in_data, bo_out_data);
    auto state = run.wait(std::chrono::seconds(20)); // Add timeout
    if (state != ERT_CMD_STATE_COMPLETED) {
        std::cout << "Kernel execution timed out or failed" << std::endl;
        // Handle error
    }
    // run.wait();
    std::cout << "Done avg pooling backward\n";

    // Read output from stream
    bo_out_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    float bo_out_data_golden[IN_C * IN_ROWS * IN_COLS];
    pool_bwd_golden<2, 2, 6, 24, 24>(bo_in_data_map, bo_out_data_golden);

    // Print sample of output data for verification
    std::cout << "Sample dX data (first channel, first 6x6):" << std::endl;
    for(int i=0; i < 6; i++) {
        for(int j=0; j < 6; j++) {
            std::cout << bo_out_data_map[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Print expected output for comparison
    std::cout << "Expected dX data (first channel, first 6x6):" << std::endl;
    for(int i=0; i < 6; i++) {
        for(int j=0; j < 6; j++) {
            std::cout << bo_out_data_golden[i*IN_COLS + j] << "\t";
        }
        std::cout << std::endl;
    }

    int errors = 0;
    const float EPSILON = 1e-5f; // Allow for small floating-point differences
    
    // Check for errors
    std::cout << "\nChecking for errors..." << std::endl;
    for(int ch=0; ch < IN_C; ch++) {
        for(int i=0; i < IN_ROWS; i++) {
            for(int j=0; j < IN_COLS; j++) {
                int idx = ch*IN_ROWS*IN_COLS + i*IN_COLS + j;
                // Use epsilon comparison for floating point
                if(std::abs(bo_out_data_map[idx] - bo_out_data_golden[idx]) > EPSILON) {
                    errors++;
                    if(errors <= 10) { // Limit output to first 10 errors
                        std::cout << "Error at ch=" << ch << ", row=" << i << ", col=" << j 
                                  << ": expected " << bo_out_data_golden[idx] 
                                  << ", got " << bo_out_data_map[idx] << std::endl;
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
