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

#include "weights_bias.h"
#include "weights_bias_float.h"
#include "lenet5/forward_path.h"
#include "../test/test_utils.h"
#include "constants.hpp"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

 
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

    size_t in_size = CONV1_IN_CH * CONV1_IN_ROWS * CONV1_IN_COLS;
    size_t conv1_out_size = CONV1_OUT_CH * CONV1_OUT_ROWS * CONV1_OUT_COLS;
    size_t conv1_weight_size = CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE;
    size_t conv1_bias_size = CONV1_OUT_CH;

    size_t pool1_out_size = CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS;

    size_t conv2_out_size = CONV2_OUT_CH * CONV2_OUT_ROWS * CONV2_OUT_COLS;
    size_t conv2_weight_size = CONV2_OUT_CH * CONV2_IN_CH * KERNEL_SIZE * KERNEL_SIZE;
    size_t conv2_bias_size = CONV2_OUT_CH;

    size_t pool2_out_size = FC1_IN_DIM;

    size_t fc1_out_size = FC1_OUT_DIM;
    size_t fc1_weight_size = FC1_IN_DIM * FC1_OUT_DIM;
    size_t fc1_bias_size = FC1_OUT_DIM;

    size_t fc2_out_size = FC2_OUT_DIM;
    size_t fc2_weight_size = FC2_IN_DIM * FC2_OUT_DIM;
    size_t fc2_bias_size = FC2_OUT_DIM;

    size_t fc3_out_size = FC3_OUT_DIM;
    size_t fc3_weight_size = FC3_IN_DIM * FC3_OUT_DIM;
    size_t fc3_bias_size = FC3_OUT_DIM;

    // Create kernels
    auto forward_krnl = xrt::kernel(device, uuid, "forward_path");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_in_data = xrt::bo(device, sizeof(float)*in_size, forward_krnl.group_id(0));

    auto bo_conv1_weight = xrt::bo(device, sizeof(float)*conv1_weight_size, forward_krnl.group_id(1));
    auto bo_conv1_bias = xrt::bo(device, sizeof(float)*conv1_bias_size, forward_krnl.group_id(2));
    auto bo_conv1_out = xrt::bo(device, sizeof(float)*conv1_out_size, forward_krnl.group_id(3));

    auto bo_pool1_out = xrt::bo(device, sizeof(float)*pool1_out_size, forward_krnl.group_id(4));

    auto bo_conv2_weight = xrt::bo(device, sizeof(float)*conv2_weight_size, forward_krnl.group_id(5));
    auto bo_conv2_bias = xrt::bo(device, sizeof(float)*conv2_bias_size, forward_krnl.group_id(6));
    auto bo_conv2_out = xrt::bo(device, sizeof(float)*conv2_out_size, forward_krnl.group_id(7));

    auto bo_pool2_out = xrt::bo(device, sizeof(float)*pool2_out_size, forward_krnl.group_id(8));

    auto bo_fc1_weight = xrt::bo(device, sizeof(float)*fc1_weight_size, forward_krnl.group_id(9));
    auto bo_fc1_bias = xrt::bo(device, sizeof(float)*fc1_bias_size, forward_krnl.group_id(10));
    auto bo_fc1_out = xrt::bo(device, sizeof(float)*fc1_out_size, forward_krnl.group_id(11));

    auto bo_fc2_weight = xrt::bo(device, sizeof(float)*fc2_weight_size, forward_krnl.group_id(12));
    auto bo_fc2_bias = xrt::bo(device, sizeof(float)*fc2_bias_size, forward_krnl.group_id(13));
    auto bo_fc2_out = xrt::bo(device, sizeof(float)*fc2_out_size, forward_krnl.group_id(14));

    auto bo_fc3_weight = xrt::bo(device, sizeof(float)*fc3_weight_size, forward_krnl.group_id(15));
    auto bo_fc3_bias = xrt::bo(device, sizeof(float)*fc3_bias_size, forward_krnl.group_id(16));
    auto bo_fc3_out = xrt::bo(device, sizeof(float)*fc3_out_size, forward_krnl.group_id(17));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<float *>();
    auto bo_conv1_weight_map = bo_conv1_weight.map<float *>();
    auto bo_conv1_bias_map = bo_conv1_bias.map<float *>();
    auto bo_conv1_out_map = bo_conv1_out.map<float *>();

    auto bo_pool1_out_map = bo_pool1_out.map<float *>();

    auto bo_conv2_weight_map = bo_conv2_weight.map<float *>();
    auto bo_conv2_bias_map = bo_conv2_bias.map<float *>();
    auto bo_conv2_out_map = bo_conv2_out.map<float *>();

    auto bo_pool2_out_map = bo_pool2_out.map<float *>();

    auto bo_fc1_weight_map = bo_fc1_weight.map<float *>();
    auto bo_fc1_bias_map = bo_fc1_bias.map<float *>();
    auto bo_fc1_out_map = bo_fc1_out.map<float *>();

    auto bo_fc2_weight_map = bo_fc2_weight.map<float *>();
    auto bo_fc2_bias_map = bo_fc2_bias.map<float *>();
    auto bo_fc2_out_map = bo_fc2_out.map<float *>();

    auto bo_fc3_weight_map = bo_fc3_weight.map<float *>();
    auto bo_fc3_bias_map = bo_fc3_bias.map<float *>();
    auto bo_fc3_out_map = bo_fc3_out.map<float *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_data_map, bo_in_data_map + in_size, 0);
    std::fill(bo_conv1_weight_map, bo_conv1_weight_map + conv1_weight_size, 0);
    std::fill(bo_conv1_bias_map, bo_conv1_bias_map + conv1_bias_size, 0);
    std::fill(bo_conv1_out_map, bo_conv1_out_map + conv1_out_size, 0);

    std::fill(bo_pool1_out_map, bo_pool1_out_map + pool1_out_size, 0);

    std::fill(bo_conv2_weight_map, bo_conv2_weight_map + conv2_weight_size, 0);
    std::fill(bo_conv2_bias_map, bo_conv2_bias_map + conv2_bias_size, 0);
    std::fill(bo_conv2_out_map, bo_conv2_out_map + conv2_out_size, 0);

    std::fill(bo_pool2_out_map, bo_pool2_out_map + pool2_out_size, 0);
    
    std::fill(bo_fc1_weight_map, bo_fc1_weight_map + fc1_weight_size, 0);
    std::fill(bo_fc1_bias_map, bo_fc1_bias_map + fc1_bias_size, 0);
    std::fill(bo_fc1_out_map, bo_fc1_out_map + fc1_out_size, 0);

    std::fill(bo_fc2_weight_map, bo_fc2_weight_map + fc2_weight_size, 0);
    std::fill(bo_fc2_bias_map, bo_fc2_bias_map + fc2_bias_size, 0);
    std::fill(bo_fc2_out_map, bo_fc2_out_map + fc2_out_size, 0);

    std::fill(bo_fc3_weight_map, bo_fc3_weight_map + fc3_weight_size, 0);
    std::fill(bo_fc3_bias_map, bo_fc3_bias_map + fc3_bias_size, 0);
    std::fill(bo_fc3_out_map, bo_fc3_out_map + fc3_out_size, 0);

    // Write input data
    for(int i = 0; i < in_size; i++) {
        bo_in_data_map[i] = SAMPLE_INPUT[i];
    }

    // Write Conv1 weights and biases
    for(int i = 0; i < conv1_weight_size; i++) {
        bo_conv1_weight_map[i] = CONV1_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < conv1_bias_size; i++) {
        bo_conv1_bias_map[i] = CONV1_BIAS_FP32_DATA[i];
    }

    // Write Conv2 weights and biases
    for(int i = 0; i < conv2_weight_size; i++) {
        bo_conv2_weight_map[i] = CONV2_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < conv2_bias_size; i++) {
        bo_conv2_bias_map[i] = CONV2_BIAS_FP32_DATA[i];
    }

    // Write FC1 weights and biases
    for(int i = 0; i < fc1_weight_size; i++) {
        bo_fc1_weight_map[i] = FC1_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < fc1_bias_size; i++) {
        bo_fc1_bias_map[i] = FC1_BIAS_FP32_DATA[i];
    }

    // Write FC2 weights and biases
    for(int i = 0; i < fc2_weight_size; i++) {
        bo_fc2_weight_map[i] = FC2_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < fc2_bias_size; i++) {
        bo_fc2_bias_map[i] = FC2_BIAS_FP32_DATA[i];
    }

    // Write FC3 weights and biases
    for(int i = 0; i < fc3_weight_size; i++) {
        bo_fc3_weight_map[i] = FC3_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < fc3_bias_size; i++) {
        bo_fc3_bias_map[i] = FC3_BIAS_FP32_DATA[i];
    }

    // Sync all buffers to device
    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv1_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv1_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv2_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv2_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc1_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc1_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc2_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc2_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc3_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc3_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // std::cout << "input = [";
    // for(int i = 0; i < IN_C*IN_ROWS*IN_COLS; i++) {
    //     std::cout << bo_in_data_map[i] << ", ";
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "weight = [";
    // for(int i = 0; i < OUT_C * IN_C * KERNEL_SIZE * KERNEL_SIZE; i++) {
    //     std::cout << bo_weights_map[i] << ", ";
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "bias = [";
    // for(int i = 0; i < OUT_C; i++) {
    //     std::cout << bo_bias_map[i] << ", ";
    // }
    // std::cout << "]" << std::endl;

    // Run forward path
    std::cout << "Running the kernel\n";
    auto run = forward_krnl(
        bo_in_data,
        bo_conv1_weight,
        bo_conv1_bias,
        bo_conv1_out,
        bo_pool1_out,
        bo_conv2_weight,
        bo_conv2_bias,
        bo_conv2_out,   
        bo_pool2_out,
        bo_fc1_weight,
        bo_fc1_bias,
        bo_fc1_out,
        bo_fc2_weight,
        bo_fc2_bias,
        bo_fc2_out,
        bo_fc3_weight,
        bo_fc3_bias,
        bo_fc3_out
    );
    auto state = run.wait(std::chrono::seconds(10)); // Add timeout
    if (state != ERT_CMD_STATE_COMPLETED) {
        std::cout << "Kernel execution timed out or failed" << std::endl;
        // Handle error
    }
    // run.wait();
    std::cout << "Done convolution\n";

    // Read output from stream
    bo_conv1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_pool1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_conv2_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_pool2_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc2_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc3_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    float conv1_out_golden[conv1_out_size];
    conv_golden<CONV1_OUT_CH, CONV1_IN_CH, KERNEL_SIZE, CONV1_IN_ROWS, CONV1_IN_COLS>(bo_in_data_map, conv1_out_golden, bo_conv1_weight_map, bo_conv1_bias_map);
    std::cout << "Sample of conv1_out (6x6): " << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << bo_conv1_out_map[i * (CONV1_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Conv1 Test
    int global_errors = 0;
    int errors = 0;
    float max_diff = 0.0f;
    std::cout << "Conv1 error indexes: ";
    for(int i=0; i<conv1_out_size; i++) {
        float diff = std::fabs(bo_conv1_out_map[i] - conv1_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_conv1_out_map[i] 
                          << ", expected " << conv1_out_golden[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // Pool1 Test
    float pool1_out_golden[pool1_out_size];
    pool_golden<2, 2, CONV1_OUT_CH, CONV1_OUT_ROWS, CONV1_OUT_COLS>(conv1_out_golden, pool1_out_golden);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample pool1_out (6x6): " << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << bo_pool1_out_map[i * CONV2_IN_COLS + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Pool1 error indexes: ";
    for(int i=0; i<pool1_out_size; i++) {
        float diff = std::fabs(bo_pool1_out_map[i] - pool1_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_pool1_out_map[i] 
                          << ", expected " << pool1_out_golden[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // Conv2 Test
    float conv2_out_golden[conv2_out_size];
    conv_golden<CONV2_OUT_CH, CONV2_IN_CH, KERNEL_SIZE, CONV2_IN_ROWS, CONV2_IN_COLS>(bo_pool1_out_map, conv2_out_golden, bo_conv2_weight_map, bo_conv2_bias_map);
    std::cout << "Sample of conv2_out (6x6): " << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << bo_conv2_out_map[i * (CONV2_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    errors = 0;
    max_diff = 0.0f;
    std::cout << "Conv2 error indexes: ";
    for(int i=0; i<conv2_out_size; i++) {
        float diff = std::fabs(bo_conv2_out_map[i] - conv2_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_conv2_out_map[i] 
                          << ", expected " << conv2_out_golden[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // Pool2 Test
    float pool2_out_golden[pool2_out_size];
    pool_golden<2, 2, CONV2_OUT_CH, CONV2_OUT_ROWS, CONV2_OUT_COLS>(conv2_out_golden, pool2_out_golden);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample pool2_out (4x4): " << std::endl;
    for(int i=0; i<4; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << bo_pool2_out_map[i * (CONV2_OUT_COLS/2) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Pool2 error indexes: ";
    for(int i=0; i<pool2_out_size; i++) {
        float diff = std::fabs(bo_pool2_out_map[i] - pool2_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_pool2_out_map[i] 
                          << ", expected " << pool2_out_golden[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // FC1 Test
    float fc1_out_golden[fc1_out_size];
    fc_golden<FC1_IN_DIM, FC1_OUT_DIM>(pool2_out_golden, fc1_out_golden, bo_fc1_weight_map, bo_fc1_bias_map, true);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample fc1_out (10): " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << bo_fc1_out_map[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "FC1 error indexes: ";
    for(int i=0; i<fc1_out_size; i++) {
        float diff = std::fabs(bo_fc1_out_map[i] - fc1_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_fc1_out_map[i] 
                          << ", expected " << fc1_out_golden[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // FC2 Test
    float fc2_out_golden[fc2_out_size];
    fc_golden<FC2_IN_DIM, FC2_OUT_DIM>(fc1_out_golden, fc2_out_golden, bo_fc2_weight_map, bo_fc2_bias_map, true);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample fc2_out (10): " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << bo_fc2_out_map[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "FC2 error indexes: ";
    for(int i=0; i<fc2_out_size; i++) {
        float diff = std::fabs(bo_fc2_out_map[i] - fc2_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_fc2_out_map[i] 
                          << ", expected " << fc2_out_golden[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // FC3 Test
    float fc3_out_golden[fc3_out_size];
    fc_golden<FC3_IN_DIM, FC3_OUT_DIM>(fc2_out_golden, fc3_out_golden, bo_fc3_weight_map, bo_fc3_bias_map, false);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "fc3_out = [";
    for(int i=0; i<10; i++) {
        std::cout << bo_fc3_out_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "FC3 error indexes: ";
    for(int i=0; i<fc3_out_size; i++) {
        float diff = std::fabs(bo_fc3_out_map[i] - fc3_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_fc3_out_map[i] 
                          << ", expected " << fc3_out_golden[i] 
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    std::cout << "Total errors: " << global_errors << std::endl;

    std::cout << "Test completed\n";
    return 0;
}
