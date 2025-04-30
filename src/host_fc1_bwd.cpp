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
#include <ctime>
#include <limits>

#include "../test/test_utils.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define IN_DIM 256
#define OUT_DIM 120
 
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
    size_t in_activation_size = IN_DIM;
    size_t grads_size = OUT_DIM;
    size_t in_weight_size = IN_DIM*OUT_DIM;
    size_t dX_size = IN_DIM;
    size_t dW_size = IN_DIM * OUT_DIM;
    size_t dB_size = OUT_DIM;

    // Create kernels
    auto fc1_bwd_krnl = xrt::kernel(device, uuid, "fc1_bwd");

    std::cout << "Allocate Buffer in Global Memory\n";

    // Allocate buffers
    auto bo_in_activation = xrt::bo(device, sizeof(float) * in_activation_size, fc1_bwd_krnl.group_id(0));
    auto bo_grads = xrt::bo(device, sizeof(float) * grads_size, fc1_bwd_krnl.group_id(1));
    auto bo_in_weight = xrt::bo(device, sizeof(float) * in_weight_size, fc1_bwd_krnl.group_id(2));
    auto bo_dX = xrt::bo(device, sizeof(float) * dX_size, fc1_bwd_krnl.group_id(3));
    auto bo_dW = xrt::bo(device, sizeof(float) * dW_size, fc1_bwd_krnl.group_id(4));
    auto bo_dB = xrt::bo(device, sizeof(float) * dB_size, fc1_bwd_krnl.group_id(5));

    // Map buffers to host memory
    auto bo_in_activation_map = bo_in_activation.map<float *>();
    auto bo_grads_map = bo_grads.map<float *>();
    auto bo_in_weight_map = bo_in_weight.map<float *>();
    auto bo_dX_map = bo_dX.map<float *>();
    auto bo_dW_map = bo_dW.map<float *>();
    auto bo_dB_map = bo_dB.map<float *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_activation_map, bo_in_activation_map + in_activation_size, 0);
    std::fill(bo_grads_map, bo_grads_map + grads_size, 0);
    std::fill(bo_in_weight_map, bo_in_weight_map + in_weight_size, 0);
    std::fill(bo_dX_map, bo_dX_map + dX_size, 0);
    std::fill(bo_dW_map, bo_dW_map + dW_size, 0);
    std::fill(bo_dB_map, bo_dB_map + dB_size, 0);

    // Initialize data with a pattern that makes it easy to verify results
    for(int i=0; i < in_activation_size; i++) {
        bo_in_activation_map[i] = 0.1f * i;
    }
    for(int i = 0; i < grads_size; i++) {
        bo_grads_map[i] = 0.2f * i;
    }
    for(int i = 0; i < in_weight_size; i++) {
        bo_in_weight_map[i] = 0.3f * i;
    }

    // Step 4: Sync inputs, weights, and biases.
    bo_in_activation.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_grads.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Define the use_relu value
    bool use_relu = false;

    // Run FC layer with use_relu parameter
    std::cout << "Running FC3 backward\n";
    auto run = fc1_bwd_krnl(bo_in_activation, bo_grads, bo_in_weight, bo_dX, bo_dW, bo_dB);
    auto state = run.wait(std::chrono::seconds(20)); // Add timeout
    if (state != ERT_CMD_STATE_COMPLETED) {
        std::cout << "Kernel execution timed out or failed" << std::endl;
        // Handle error
    }
    // run.wait();
    std::cout << "Done fc\n";

    // Read output from stream
    bo_dX.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_dW.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_dB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    float dX_golden[IN_DIM];
    float dW_golden[IN_DIM*OUT_DIM];
    float dB_golden[OUT_DIM];
    fc_bwd_golden<IN_DIM, OUT_DIM>(bo_in_activation_map, bo_grads_map, bo_in_weight_map, dX_golden, dW_golden, dB_golden, use_relu);

    // Compare results
    int total_errs = 0;
    int errs = 0;
    float max_diff = 0.0f;
    
    // Verify dX
    int dX_length = IN_DIM;
    for(int j=0; j<dX_length; j++) {
        float diff = std::fabs(bo_dX_map[j] - dX_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << bo_dX_map[j] 
                          << ", expected " << dX_golden[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "dX Output (first 10 elements): ";
    for(int j=0; j<std::min(10, dX_length); j++) {
        std::cout << bo_dX_map[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "dX_golden (first 10 elements): ";
    for(int j=0; j<std::min(10, dX_length); j++) {
        std::cout << dX_golden[j] << " ";
    }
    std::cout << std::endl;
    std::cout << errs << " errors" << std::endl;
    std::cout << std::endl << std::endl;

    // Verify dW
    errs = 0;
    int dW_length = IN_DIM*OUT_DIM;
    for(int j=0; j<dW_length; j++) {
        float diff = std::fabs(bo_dW_map[j] - dW_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << bo_dW_map[j] 
                          << ", expected " << dW_golden[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "dW Output (first 10 elements): ";
    for(int j=0; j<std::min(10, dW_length); j++) {
        std::cout << bo_dW_map[j] << " ";
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
    int dB_length = OUT_DIM;
    for(int j=0; j<dB_length; j++) {
        float diff = std::fabs(bo_dB_map[j] - dB_golden[j]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {
            errs++;
            total_errs++;
            if (errs < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << j << ": got " << bo_dB_map[j] 
                          << ", expected " << dB_golden[j] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }

    std::cout << "dB Output (first 10 elements): ";
    for(int j=0; j<std::min(10, dB_length); j++) {
        std::cout << bo_dB_map[j] << " ";
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
    std::cout << "FC3 Backpropagation layer Test: " << (total_errs ? "FAIL" : "PASS") 
              << " (" << total_errs << " errors) out of " << dX_length + dW_length + dB_length << " outputs)" << std::endl;

    return total_errs;
}
