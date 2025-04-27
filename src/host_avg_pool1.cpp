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
#include "encrypted_weights_bias.h"
#include "keys.h"
#include "encryption.hpp"
#include "constants.hpp"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define DATA_SIZE POLYNOMIAL_DEGREE

std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));

#define IN_ROWS 24
#define IN_COLS 24
#define IN_C 6
#define OUT_ROWS (IN_ROWS / POOL_SIZE)
#define OUT_COLS (IN_COLS / POOL_SIZE)
#define POOL_SIZE 2
void avg_pool1_golden(
    const float in_data[IN_C*IN_ROWS*IN_COLS],
    float out_data[IN_C * OUT_ROWS * OUT_COLS]
) {
    for(int ch=0; ch < IN_C; ch++) {
        for(int pr=0; pr < OUT_ROWS; pr++) {
            for(int pc=0; pc < OUT_COLS; pc++) {
                float sum = 0;
                for(int i=0; i < POOL_SIZE; i++) {
                    for(int j=0; j < POOL_SIZE; j++) {
                        // sum += in_data[ch + IN_CHANNELS*((pr*POOL_SIZE+i)*IN_COLS + (pc*POOL_SIZE+j))];
                        sum += in_data[ch*IN_ROWS*IN_COLS + (pr*POOL_SIZE+i)*IN_COLS + (pc*POOL_SIZE+j)];
                    }
                }
                // Calculate average with rounding
                float avg = sum / (POOL_SIZE*POOL_SIZE);
                out_data[ch*OUT_ROWS*OUT_COLS + pr*OUT_COLS + pc] = avg;
            }
        }
    }
}
 
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

    std::vector<float> in_data(IN_C* IN_ROWS * IN_COLS);
    std::vector<float> out_data(IN_C * OUT_ROWS * OUT_COLS);

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    // size_t vector_size_bytes = sizeof(float) * DATA_SIZE;
    size_t in_size_bytes = sizeof(float) * IN_C * IN_ROWS * IN_COLS;
    size_t out_size_bytes = sizeof(float) * IN_C * OUT_ROWS * OUT_COLS;

    // Create kernels
    auto conv1_krnl = xrt::kernel(device, uuid, "avg_pool1");

    std::cout << "Allocate Buffer in Global Memory\n";

    // Allocate in and out for conv1
    auto bo_in_data = xrt::bo(device, in_size_bytes, conv1_krnl.group_id(0));
    auto bo_out_data = xrt::bo(device, out_size_bytes, conv1_krnl.group_id(1));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<float *>();
    auto bo_out_data_map = bo_out_data.map<float *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_data_map, bo_in_data_map + IN_C * IN_ROWS * IN_COLS, 0);
    std::fill(bo_out_data_map, bo_out_data_map + IN_C * OUT_ROWS * OUT_COLS, 0);

    // Initialize input data with a pattern that makes it easy to verify results
    for(int ch=0; ch < IN_C; ch++) {
        for(int i=0; i < IN_ROWS; i++) {
            for(int j=0; j < IN_COLS; j++) {
                // Use a different value for each channel to verify channel independence
                // Also use position-dependent values to verify proper pooling regions
                in_data[ch*IN_ROWS*IN_COLS + i*IN_COLS + j] = ch*100 + i + j;
            }
        }
    }

    // Step 4: Sync inputs
    for(int ch=0; ch < IN_C; ch++) {
        for(int i=0; i < IN_ROWS; i++) {
            for(int j=0; j < IN_COLS; j++) {
                bo_in_data_map[ch*IN_ROWS*IN_COLS + i*IN_COLS + j] = in_data[ch*IN_ROWS*IN_COLS + i*IN_COLS + j];
            }
        }
    }
    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "input = [";
    for(int i = 0; i < IN_C*IN_ROWS*IN_COLS; i++) {
        std::cout << bo_in_data_map[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    // Run convolution
    std::cout << "Running avg pooling\n";
    auto run = conv1_krnl(bo_in_data, bo_out_data);
    // auto state = run.wait(std::chrono::seconds(20)); // Add timeout
    // if (state != ERT_CMD_STATE_COMPLETED) {
    //     std::cout << "Kernel execution timed out or failed" << std::endl;
    //     // Handle error
    // }
    run.wait();
    std::cout << "Done avg pooling\n";

    // Read output from stream
    bo_out_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    float bo_out_data_golden[IN_C * OUT_ROWS * OUT_COLS];
    avg_pool1_golden(bo_in_data_map, bo_out_data_golden);
    // Print results
    std::cout << "Avg pooling results:\n";
    std::cout << "out_data = [";
    for(int i=0; i< IN_C * OUT_ROWS * OUT_COLS; i++) {
        std::cout << bo_out_data_map[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "out_data_golden = [";
    for(int i=0; i< IN_C * OUT_ROWS * OUT_COLS; ++i) {
        std::cout << bo_out_data_golden[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Test completed\n";
    return 0;
}
