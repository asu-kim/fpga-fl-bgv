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

#define IN_ROWS 12
#define IN_COLS 12
#define KERNEL_SIZE 5
#define IN_C 6
#define OUT_C 16
#define OUT_ROWS (IN_ROWS - KERNEL_SIZE + 1)
#define OUT_COLS (IN_COLS - KERNEL_SIZE + 1)
void conv2_golden(
    const data_ap_fixed_t in_flatten[IN_C*IN_ROWS*IN_COLS],
    data_ap_fixed_t out_data[OUT_C * OUT_ROWS * OUT_COLS],
    const data_ap_fixed_t weights_flatten[IN_C*OUT_C*IN_ROWS*IN_COLS],
    const data_ap_fixed_t bias_flatten[OUT_C]
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
                    weights[i][j][k][l] = weights_flatten[i * IN_C*KERNEL_SIZE*KERNEL_SIZE + j * KERNEL_SIZE * KERNEL_SIZE + k * KERNEL_SIZE + l];
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
                // Initialize accumulator with bias
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
                            // printf("in_data[%d][%d][%d] = %d\n", ic, ih, iw, in_val);
                            // printf("w_data[%d][%d][%d][%d] = %d\n", oc, ic, kh, kw, w_val);
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

    std::vector<data_ap_fixed_t> in_data(IN_C*IN_ROWS*IN_COLS);
    std::vector<data_ap_fixed_t> out_data(OUT_C * OUT_ROWS * OUT_COLS);
    std::vector<data_ap_fixed_t> weight_data(OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE);
    std::vector<data_ap_fixed_t> bias_data(OUT_C);

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    size_t in_size_bytes = sizeof(data_ap_fixed_t) * IN_C * IN_ROWS * IN_COLS;
    size_t out_size_bytes = sizeof(data_ap_fixed_t) * OUT_C * OUT_ROWS * OUT_COLS;
    size_t weight_size_bytes = sizeof(data_ap_fixed_t) * OUT_C* IN_C * KERNEL_SIZE * KERNEL_SIZE;
    size_t bias_size_bytes = sizeof(data_ap_fixed_t) * OUT_C;

    // Create kernels
    auto conv2_krnl = xrt::kernel(device, uuid, "conv2");

    std::cout << "Allocate Buffer in Global Memory\n";

    // Allocate in and out for conv2
    auto bo_in_data = xrt::bo(device, in_size_bytes, conv2_krnl.group_id(0));
    auto bo_out_data = xrt::bo(device, out_size_bytes, conv2_krnl.group_id(1));

    // Allocate weights and biases for conv2
    auto bo_weights = xrt::bo(device, weight_size_bytes, conv2_krnl.group_id(2));
    auto bo_bias = xrt::bo(device, bias_size_bytes, conv2_krnl.group_id(3));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<data_ap_fixed_t *>();
    auto bo_out_data_map = bo_out_data.map<data_ap_fixed_t *>();
    auto bo_weights_map = bo_weights.map<data_ap_fixed_t *>();
    auto bo_bias_map = bo_bias.map<data_ap_fixed_t *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_data_map, bo_in_data_map + IN_C * IN_ROWS * IN_COLS, 0);
    std::fill(bo_out_data_map, bo_out_data_map + OUT_C * OUT_ROWS * OUT_COLS, 0);
    std::fill(bo_weights_map, bo_weights_map + OUT_C * IN_C * KERNEL_SIZE * KERNEL_SIZE, 0);
    std::fill(bo_bias_map, bo_bias_map + OUT_C, 0);

    // Write input data (all 1s)
    for(int i = 0; i < IN_C; i++) {
        for(int j=0; j<IN_ROWS; j++) {
            for(int k = 0; k < IN_COLS; k++) {
                in_data[i*IN_ROWS*IN_COLS + j*IN_COLS + k] = 1;
            }
        }
    }

    // Write weight data
    for(int channel=0; channel<OUT_C; channel++) {
        bias_data[channel] = CONV2_BIAS_INT8_DATA[channel];
        for(int ic=0; ic<IN_C; ic++) {
            // bias[channel] = 0.0;
            for(int i=0; i<KERNEL_SIZE; i++) {
                for(int j=0; j<KERNEL_SIZE; j++) {
                    data_ap_fixed_t weight = CONV2_WEIGHT_INT8_DATA[channel*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + ic*(KERNEL_SIZE*KERNEL_SIZE) + i*(KERNEL_SIZE) + j];
                    // data_ap_fixed_t weight = 1.0f;
                    weight_data[channel*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + ic*(KERNEL_SIZE*KERNEL_SIZE) + i*(KERNEL_SIZE) + j] = weight;
                }
            }
        }
    }

    // Step 4: Sync inputs
    for(int i = 0; i < IN_C; i++) {
        for(int j=0; j<IN_ROWS; j++) {
            for(int k = 0; k < IN_COLS; k++) {
                bo_in_data_map[i*IN_ROWS*IN_COLS + j*IN_COLS + k] = in_data[i*IN_ROWS*IN_COLS + j*IN_COLS + k];
            }
        }
    }

    // Sync weights and bias to device
    for(int i = 0; i < OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE; i++) {
        bo_weights_map[i] = weight_data[i];
    }
    for(int i = 0; i < OUT_C; i++) {
        bo_bias_map[i] = bias_data[i];
    }

    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "input = [";
    for(int i = 0; i < IN_C*IN_ROWS*IN_COLS; i++) {
        std::cout << bo_in_data_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "weight = [";
    for(int i = 0; i < OUT_C * IN_C * KERNEL_SIZE * KERNEL_SIZE; i++) {
        std::cout << bo_weights_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "bias = [";
    for(int i = 0; i < OUT_C; i++) {
        std::cout << bo_bias_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    // Run convolution
    std::cout << "Running convolution\n";
    auto run = conv2_krnl(bo_in_data, bo_out_data, bo_weights, bo_bias);
    auto state = run.wait(std::chrono::seconds(10)); // Add timeout
    if (state != ERT_CMD_STATE_COMPLETED) {
        std::cout << "Kernel execution timed out or failed" << std::endl;
        // Handle error
    }
    // run.wait();
    std::cout << "Done convolution\n";

    // Read output from stream
    bo_out_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    data_ap_fixed_t bo_out_data_golden[OUT_C * OUT_ROWS * OUT_COLS];
    conv2_golden(bo_in_data_map, bo_out_data_golden, bo_weights_map, bo_bias_map);
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
