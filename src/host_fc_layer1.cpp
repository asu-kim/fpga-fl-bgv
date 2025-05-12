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

    std::vector<data_ap_fixed_t> in_data(IN_DIM);
    std::vector<data_ap_fixed_t> out_data(OUT_DIM);
    std::vector<data_ap_fixed_t> weight_data(IN_DIM*OUT_DIM);
    std::vector<data_ap_fixed_t> bias_data(OUT_DIM);

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    // size_t vector_size_bytes = sizeof(data_ap_fixed_t) * DATA_SIZE;
    size_t in_size_bytes = sizeof(data_ap_fixed_t) * IN_DIM;
    size_t out_size_bytes = sizeof(data_ap_fixed_t) * OUT_DIM;
    size_t weight_size_bytes = sizeof(data_ap_fixed_t) * IN_DIM*OUT_DIM;
    size_t bias_size_bytes = sizeof(data_ap_fixed_t) * OUT_DIM;

    // Create kernels
    auto fc1_krnl = xrt::kernel(device, uuid, "fc1");

    std::cout << "Allocate Buffer in Global Memory\n";

    // Allocate in and out for conv1
    auto bo_in_data = xrt::bo(device, in_size_bytes, fc1_krnl.group_id(0));
    auto bo_out_data = xrt::bo(device, out_size_bytes, fc1_krnl.group_id(1));

    // Allocate weights and biases for conv1
    auto bo_weights = xrt::bo(device, weight_size_bytes, fc1_krnl.group_id(2));
    auto bo_bias = xrt::bo(device, bias_size_bytes, fc1_krnl.group_id(3));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<data_ap_fixed_t *>();
    auto bo_out_data_map = bo_out_data.map<data_ap_fixed_t *>();
    auto bo_weights_map = bo_weights.map<data_ap_fixed_t *>();
    auto bo_bias_map = bo_bias.map<data_ap_fixed_t *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_data_map, bo_in_data_map + IN_DIM, 0);
    std::fill(bo_out_data_map, bo_out_data_map + OUT_DIM, 0);
    std::fill(bo_weights_map, bo_weights_map + IN_DIM * OUT_DIM, 0);
    std::fill(bo_bias_map, bo_bias_map + OUT_DIM, 0);

    // Initialize input data with a pattern that makes it easy to verify results
    for(int i=0; i < IN_DIM; i++) {
        // Use a different value for each channel to verify channel independence
        // Also use position-dependent values to verify proper pooling regions
        // in_data[i] = i * 0.5f;
        in_data[i] = 1;
    }
    // Initialize weights with a deterministic pattern
    for(int i=0; i<IN_DIM; i++) {
        for(int j=0; j<OUT_DIM; j++) {
            // weight_data[i*OUT_DIM + j] = 0.01f * (i+j);
            weight_data[i*OUT_DIM + j] = 1;
        }
    }
    // Initialize bias values
    for(int i=0; i<OUT_DIM; i++) {
        // bias_data[i] = i * 0.1f;
        bias_data[i] = 1;
    }


    // Step 4: Sync inputs, weights, and biases.
    for(int i=0; i < IN_DIM; i++) {
        bo_in_data_map[i] = in_data[i];
    }
    for(int i=0; i<IN_DIM; i++) {
        for(int j=0; j<OUT_DIM; j++) {
            bo_weights_map[i*OUT_DIM + j] = weight_data[i*OUT_DIM + j];
        }
    }
    for(int i=0; i<OUT_DIM; i++) {
        bo_bias_map[i] = bias_data[i];
    }
    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // std::cout << "input = [";
    // for(int i = 0; i < IN_DIM; i++) {
    //     std::cout << bo_in_data_map[i] << ", ";
    // }
    // std::cout << "]" << std::endl << std::endl;

    // std::cout << "weight = [";
    // for(int i = 0; i < IN_DIM*OUT_DIM; i++) {
    //     std::cout << bo_weights_map[i] << ", ";
    // }
    // std::cout << "]" << std::endl << std::endl;

    // std::cout << "bias = [";
    // for(int i = 0; i < OUT_DIM; i++) {
    //     std::cout << bo_bias_map[i] << ", ";
    // }
    // std::cout << "]" << std::endl << std::endl;

    // Define the use_relu value
    bool use_relu = true;

    // Run FC layer with use_relu parameter
    std::cout << "Running fc\n";

    auto hw_start = std::chrono::high_resolution_clock::now();
    auto run = fc1_krnl(bo_in_data, bo_out_data, bo_weights, bo_bias, use_relu);
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
    std::cout << "Done FC1\n";

    data_ap_fixed_t bo_out_data_golden[OUT_DIM];
    auto cpu_start = std::chrono::high_resolution_clock::now();
    fc_golden(bo_in_data_map, bo_out_data_golden, bo_weights_map, bo_bias_map, use_relu);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU reference execution time: " << cpu_duration << " ms" << std::endl;

    // // Print results
    // std::cout << "FC results:\n";
    // std::cout << "out_data = [";
    // for(int i=0; i< OUT_DIM; i++) {
    //     std::cout << bo_out_data_map[i] << ", ";
    // }
    // std::cout << "]" << std::endl << std::endl;

    // std::cout << "out_data_golden = [";
    // for(int i=0; i< OUT_DIM; ++i) {
    //     std::cout << bo_out_data_golden[i] << ", ";
    // }
    // std::cout << "]" << std::endl;
    int errors = 0;
    data_ap_fixed_t max_error = 0.0f;

    for(int i=0; i < OUT_DIM; i++) {
        data_ap_fixed_t diff = hls::abs(bo_out_data_map[i] - bo_out_data_golden[i]);
        max_error = std::max(max_error, diff);
        if(diff > data_ap_fixed_t(0.001)) {
            errors++;
        }
    }

    std::cout << "Maximum error: " << max_error << std::endl;
    std::cout << "Number of errors: " << errors << std::endl;

    std::cout << "Test completed\n";
    return 0;
}
