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
#include <chrono>
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
#include "constants.hpp"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define DATA_SIZE POLYNOMIAL_DEGREE

std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));

#define IN_ROWS 28
#define IN_COLS 28
#define KERNEL_SIZE 5
#define IN_C 1
#define OUT_C 6
#define OUT_ROWS (IN_ROWS - KERNEL_SIZE + 1)
#define OUT_COLS (IN_COLS - KERNEL_SIZE + 1)
void conv1_golden(
    const data_ap_fixed_t in_flatten[IN_C*IN_ROWS*IN_COLS],
    data_ap_fixed_t out_data[OUT_C * OUT_ROWS * OUT_COLS],
    const data_ap_fixed_t weights_flatten[128],
    const data_ap_fixed_t bias_flatten[128]
) {
    data_ap_fixed_t in_data[IN_C][IN_ROWS][IN_COLS];
    data_ap_fixed_t weights[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE];
    data_ap_fixed_t bias[OUT_C];
    for(int i = 0; i < IN_C; i++) {
        for(int j = 0; j < IN_ROWS; j++) {
            for(int k = 0; k < IN_COLS; k++) {
                in_data[i][j][k] = in_flatten[i * IN_ROWS * IN_COLS + j * IN_COLS + k];
            }
        }
    }

    for(int i=0; i<OUT_C; i++) {
        bias[i] = bias_flatten[i];
        for(int j=0; j<IN_C; j++) {
            for(int k=0; k<KERNEL_SIZE; k++) {
                for(int l=0; l<KERNEL_SIZE; l++) {
                    weights[i][j][k][l] = weights_flatten[i * IN_C*KERNEL_SIZE*KERNEL_SIZE + j * 25 * KERNEL_SIZE * KERNEL_SIZE + k * KERNEL_SIZE + l];
                }
            }
        }
    }

    // Loop over each output channel
    for (int oc = 0; oc < OUT_C; oc++) {
        // Loop over each output row
        for (int oh = 0; oh < IN_ROWS - KERNEL_SIZE + 1; oh++) {
            // Loop over each output column
            for (int ow = 0; ow < IN_COLS - KERNEL_SIZE + 1; ow++) {
                int idx = oc * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)
                            + oh * (IN_COLS - KERNEL_SIZE + 1)
                            + ow;
                data_ap_fixed_t acc = bias[oc];
                
                // Calculate convolution for current output position
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            // Input position
                            int ih = oh + kh;
                            int iw = ow + kw;
                            
                            // Accumulate weighted input
                            data_ap_fixed_t in_val = in_data[ic][ih][iw];
                            data_ap_fixed_t w_val = weights[oc][ic][kh][kw];
                            acc += in_val * w_val;
                        }
                    }
                }

                // Calculate output index and store result
                int out_idx = oc * OUT_ROWS * OUT_COLS
                            + oh * OUT_COLS
                            + ow;
                out_data[out_idx] = acc;
                // printf("out[%d] = %d\n", out_idx, result);
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

    std::vector<data_ap_fixed_t> in_data(IN_ROWS*IN_COLS);
    std::vector<data_ap_fixed_t> out_data(OUT_C * OUT_ROWS * OUT_COLS);

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    size_t vector_size_bytes = sizeof(data_ap_fixed_t) * DATA_SIZE;
    size_t in_size_bytes = sizeof(data_ap_fixed_t) * IN_ROWS * IN_COLS;
    size_t out_size_bytes = sizeof(data_ap_fixed_t) * OUT_C * OUT_ROWS * OUT_COLS;

    // Create kernels
    auto conv1_krnl = xrt::kernel(device, uuid, "conv1");

    std::cout << "Allocate Buffer in Global Memory\n";

    // Allocate in and out for conv1
    auto bo_in_data = xrt::bo(device, in_size_bytes, conv1_krnl.group_id(0));
    auto bo_out_data = xrt::bo(device, out_size_bytes, conv1_krnl.group_id(1));

    // Allocate weights and biases for conv1
    auto bo_weights = xrt::bo(device, 2*vector_size_bytes, conv1_krnl.group_id(2));
    auto bo_bias = xrt::bo(device, vector_size_bytes, conv1_krnl.group_id(3));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<data_ap_fixed_t *>();
    auto bo_out_data_map = bo_out_data.map<data_ap_fixed_t *>();
    auto bo_weights_map = bo_weights.map<data_ap_fixed_t *>();
    auto bo_bias_map = bo_bias.map<data_ap_fixed_t *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_weights_map, bo_weights_map + 256, 0);
    std::fill(bo_bias_map, bo_bias_map + 128, 0);


    std::fill(bo_in_data_map, bo_in_data_map + IN_ROWS * IN_COLS, 0);
    std::fill(bo_out_data_map, bo_out_data_map + OUT_C * OUT_ROWS * OUT_COLS, 0);

    // Write input data (all 1s)
    for(int i = 0; i < IN_ROWS; i++) {
        for(int j = 0; j < IN_COLS; j++) {
            in_data[i * IN_COLS + j] = 1;
            // printf("index = %d, value = %d\n", i * IN_COLS + j, in_data[i * IN_COLS + j]);
        }
    }

    std::cout << "input = [";
    for(int i = 0; i < 784; i++) {
        std::cout << in_data[i] << ", ";
    }
    std::cout << "]" << std::endl;

    for(int i = 0; i < 256; i++) {
        if (i < OUT_C*KERNEL_SIZE*KERNEL_SIZE) {
            if (i < OUT_C) {
                bo_bias_map[i] = CONV1_BIAS_INT8_DATA[i];
                // bo_bias_map[i] = 1;
            }
            bo_weights_map[i] = CONV1_WEIGHT_INT8_DATA[i];
            // bo_weights_map[i] = 1;
        }
    }

    std::cout << "weight = [";
    for(int i = 0; i < 256; i++) {
        std::cout << bo_weights_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "bias = [";
    for(int i = 0; i < 128; i++) {
        std::cout << bo_bias_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    // Step 4: Sync inputs
    for(int i = 0; i < IN_ROWS; i++) {
        for(int j = 0; j < IN_COLS; j++) {
            bo_in_data_map[i * IN_COLS + j] = in_data[i * IN_COLS + j];
        }
    }

    std::cout << "input = [";
    for(int i = 0; i < 784; i++) {
        std::cout << bo_in_data_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    // Run convolution
    std::cout << "Running convolution\n";
    auto hw_start = std::chrono::high_resolution_clock::now();

    // Sync weights and bias to device
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = conv1_krnl(bo_in_data, bo_out_data, bo_weights, bo_bias);
    // auto state = run.wait(std::chrono::seconds(20)); // Add timeout
    // if (state != ERT_CMD_STATE_COMPLETED) {
    //     std::cout << "Kernel execution timed out or failed" << std::endl;
    //     // Handle error
    // }
    run.wait();

    // Read output from stream
    bo_out_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto hw_end = std::chrono::high_resolution_clock::now();
    auto hw_duration = std::chrono::duration<double, std::milli>(hw_end - hw_start).count();
    std::cout << "Hardware kernel execution time: " << hw_duration << " ms" << std::endl;
    std::cout << "Done convolution\n";

    data_ap_fixed_t bo_out_data_golden[OUT_C * OUT_ROWS * OUT_COLS];

    auto cpu_start = std::chrono::high_resolution_clock::now();
    conv1_golden(bo_in_data_map, bo_out_data_golden, bo_weights_map, bo_bias_map);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU reference execution time: " << cpu_duration << " ms" << std::endl;

    // Print results
    std::cout << "Convolution results:\n";
    std::cout << "out_data = [";
    for(int i=0; i< OUT_C * OUT_ROWS * OUT_COLS; i++) {
        std::cout << bo_out_data_map[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;

    std::cout << "out_data_golden = [";
    for(int i=0; i< OUT_C * OUT_ROWS * OUT_COLS; ++i) {
        std::cout << bo_out_data_golden[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Test completed\n";
    return 0;
}
