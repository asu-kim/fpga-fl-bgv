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

#include "weights_bias.h"
#include "weights_bias_float.h"
#include "lenet5/forward_path.h"
#include "../test/test_utils.h"
#include "constants.hpp"

#include "hls_math.h"

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
    size_t weights_size = TOTAL_WEIGHTS_SIZE;
    size_t biases_size = TOTAL_BIASES_SIZE;
    size_t outputs_size = TOTAL_OUTS_SIZE;

    // Create kernels
    auto forward_krnl = xrt::kernel(device, uuid, "forward_path");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_in_data = xrt::bo(device, sizeof(data_ap_fixed_t)*in_size, forward_krnl.group_id(0));
    auto bo_weights = xrt::bo(device, sizeof(data_ap_fixed_t)*weights_size, forward_krnl.group_id(1));
    auto bo_biases = xrt::bo(device, sizeof(data_ap_fixed_t)*biases_size, forward_krnl.group_id(2));
    auto bo_outs = xrt::bo(device, sizeof(data_ap_fixed_t)*outputs_size, forward_krnl.group_id(3));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<data_ap_fixed_t *>();
    auto bo_weights_map = bo_weights.map<data_ap_fixed_t *>();
    auto bo_biases_map = bo_biases.map<data_ap_fixed_t *>();
    auto bo_outs_map = bo_outs.map<data_ap_fixed_t *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_data_map, bo_in_data_map + in_size, 0);
    std::fill(bo_weights_map, bo_weights_map + weights_size, 0);
    std::fill(bo_biases_map, bo_biases_map + biases_size, 0);
    std::fill(bo_outs_map, bo_outs_map + outputs_size, 0);

    // Write input data
    for(int i = 0; i < in_size; i++) {
        bo_in_data_map[i] = SAMPLE_INPUT[i];
    }

    // Write Conv1 weights and biases
    for(int i = 0; i < NUM_CONV1_WEIGHTS; i++) {
        bo_weights_map[CONV1_WEIGHT_OFFSET + i] = CONV1_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < NUM_CONV1_BIASES; i++) {
        bo_biases_map[CONV1_BIAS_OFFSET + i] = CONV1_BIAS_FP32_DATA[i];
    }

    // Write Conv2 weights and biases
    for(int i = 0; i < NUM_CONV2_WEIGHTS; i++) {
        bo_weights_map[CONV2_WEIGHT_OFFSET + i] = CONV2_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < NUM_CONV2_BIASES; i++) {
        bo_biases_map[CONV2_BIAS_OFFSET + i] = CONV2_BIAS_FP32_DATA[i];
    }

    // Write FC1 weights and biases
    for(int i = 0; i < NUM_FC1_WEIGHTS; i++) {
        bo_weights_map[FC1_WEIGHT_OFFSET + i] = FC1_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < NUM_FC1_BIASES; i++) {
        bo_biases_map[FC1_BIAS_OFFSET + i] = FC1_BIAS_FP32_DATA[i];
    }

    // Write FC2 weights and biases
    for(int i = 0; i < NUM_FC2_WEIGHTS; i++) {
        bo_weights_map[FC2_WEIGHT_OFFSET + i] = FC2_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < NUM_FC2_BIASES; i++) {
        bo_biases_map[FC2_BIAS_OFFSET + i] = FC2_BIAS_FP32_DATA[i];
    }

    // Write FC3 weights and biases
    for(int i = 0; i < NUM_FC3_WEIGHTS; i++) {
        bo_weights_map[FC3_WEIGHT_OFFSET + i] = FC3_WEIGHT_FP32_DATA[i];
    }
    for(int i = 0; i < NUM_FC3_BIASES; i++) {
        bo_biases_map[FC3_BIAS_OFFSET + i] = FC3_BIAS_FP32_DATA[i];
    }

    // Run forward path
    std::cout << "Running the kernel\n";
    auto hw_start = std::chrono::high_resolution_clock::now();

    // Sync all buffers to device
    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_biases.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = forward_krnl(
        bo_in_data,
        bo_weights,
        bo_biases,
        bo_outs
    );
    run.wait();
    // auto state = run.wait(std::chrono::seconds(10)); // Add timeout
    // if (state != ERT_CMD_STATE_COMPLETED) {
    //     std::cout << "Kernel execution timed out or failed" << std::endl;
    //     // Handle error
    // }

    // Read output from stream
    bo_outs.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto hw_end = std::chrono::high_resolution_clock::now();
    auto hw_duration = std::chrono::duration<double, std::milli>(hw_end - hw_start).count();
    std::cout << "Hardware kernel execution time: " << hw_duration << " ms" << std::endl;
    std::cout << "Done forward\n";

    data_ap_fixed_t outs_ref[TOTAL_OUTS_SIZE];

    auto cpu_start = std::chrono::high_resolution_clock::now();
    forward_golden(
        bo_in_data_map,
        bo_weights_map,
        bo_biases_map,
        outs_ref
    );

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU reference execution time: " << cpu_duration << " ms" << std::endl;

    // Define pointers to each layer's outputs in the consolidated arrays
    const data_ap_fixed_t* bo_conv1_out = &bo_outs_map[CONV1_OUT_OFFSET];
    const data_ap_fixed_t* bo_pool1_out = &bo_outs_map[POOL1_OUT_OFFSET];
    const data_ap_fixed_t* bo_conv2_out = &bo_outs_map[CONV2_OUT_OFFSET];
    const data_ap_fixed_t* bo_pool2_out = &bo_outs_map[POOL2_OUT_OFFSET];
    const data_ap_fixed_t* bo_fc1_out = &bo_outs_map[FC1_OUT_OFFSET];
    const data_ap_fixed_t* bo_fc2_out = &bo_outs_map[FC2_OUT_OFFSET];
    const data_ap_fixed_t* bo_fc3_out = &bo_outs_map[FC3_OUT_OFFSET];

    // Define pointers to golden outputs in the consolidated reference array
    const data_ap_fixed_t* conv1_out_golden = &outs_ref[CONV1_OUT_OFFSET];
    const data_ap_fixed_t* pool1_out_golden = &outs_ref[POOL1_OUT_OFFSET];
    const data_ap_fixed_t* conv2_out_golden = &outs_ref[CONV2_OUT_OFFSET];
    const data_ap_fixed_t* pool2_out_golden = &outs_ref[POOL2_OUT_OFFSET];
    const data_ap_fixed_t* fc1_out_golden = &outs_ref[FC1_OUT_OFFSET];
    const data_ap_fixed_t* fc2_out_golden = &outs_ref[FC2_OUT_OFFSET];
    const data_ap_fixed_t* fc3_out_golden = &outs_ref[FC3_OUT_OFFSET];

    // data_ap_fixed_t conv1_out_golden[conv1_out_size];
    // conv_golden<CONV1_OUT_CH, CONV1_IN_CH, KERNEL_SIZE, CONV1_IN_ROWS, CONV1_IN_COLS>(in_data, &outs_ref[CONV1_OUT_OFFSET], weights + CONV1_WEIGHT_OFFSET, biases + CONV1_BIAS_OFFSET);
    std::cout << "Sample of conv1_out (6x6): " << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << bo_conv1_out[i * (CONV1_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Conv1 Test
    int global_errors = 0;
    int errors = 0;
    data_ap_fixed_t max_diff = 0.0f;
    std::cout << "Conv1 error indexes: ";
    for(int i=0; i<NUM_CONV1_OUTS; i++) {
        data_ap_fixed_t diff = hls::fabs(bo_conv1_out[i] - conv1_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1f)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_conv1_out[i] 
                        << ", expected " << conv1_out_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // Pool1 Test
    // data_ap_fixed_t pool1_out_golden[pool1_out_size];
    // pool_golden<2, 2, CONV1_OUT_CH, CONV1_OUT_ROWS, CONV1_OUT_COLS>(&outs_ref[CONV1_OUT_OFFSET], &outs_ref[POOL1_OUT_OFFSET]);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample pool1_out (6x6): " << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << bo_pool1_out[i * CONV2_IN_COLS + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Pool1 error indexes: ";
    for(int i=0; i<NUM_POOL1_OUTS; i++) {
        data_ap_fixed_t diff = hls::fabs(bo_pool1_out[i] - pool1_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1f)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_pool1_out[i] 
                        << ", expected " << pool1_out_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // Conv2 Test
    // data_ap_fixed_t conv2_out_golden[conv2_out_size];
    // conv_golden<CONV2_OUT_CH, CONV2_IN_CH, KERNEL_SIZE, CONV2_IN_ROWS, CONV2_IN_COLS>(&outs_ref[POOL1_OUT_OFFSET], &outs_ref[CONV2_OUT_OFFSET], weights + CONV2_WEIGHT_OFFSET, biases + CONV2_BIAS_OFFSET);
    std::cout << "Sample of conv2_out (6x6): " << std::endl;
    for(int i=0; i<6; i++) {
        for(int j = 0; j < 6; j++) {
            std::cout << bo_conv2_out[i * (CONV2_IN_COLS - KERNEL_SIZE + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    errors = 0;
    max_diff = 0.0f;
    std::cout << "Conv2 error indexes: ";
    for(int i=0; i<NUM_CONV2_OUTS; i++) {
        data_ap_fixed_t diff = hls::fabs(bo_conv2_out[i] - conv2_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1f)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_conv2_out[i] 
                        << ", expected " << conv2_out_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // Pool2 Test
    // data_ap_fixed_t pool2_out_golden[pool2_out_size];
    // pool_golden<2, 2, CONV2_OUT_CH, CONV2_OUT_ROWS, CONV2_OUT_COLS>(&outs_ref[CONV2_OUT_OFFSET], &outs_ref[POOL2_OUT_OFFSET]);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample pool2_out (4x4): " << std::endl;
    for(int i=0; i<4; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << bo_pool2_out[i * (CONV2_OUT_COLS/2) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Pool2 error indexes: ";
    for(int i=0; i<NUM_POOL2_OUTS; i++) {
        data_ap_fixed_t diff = hls::fabs(bo_pool2_out[i] - pool2_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1f)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_pool2_out[i] 
                        << ", expected " << pool2_out_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // FC1 Test
    // data_ap_fixed_t fc1_out_golden[fc1_out_size];
    // fc_golden<FC1_IN_DIM, FC1_OUT_DIM>(&outs_ref[POOL2_OUT_OFFSET], &outs_ref[FC1_OUT_OFFSET], weights + FC1_WEIGHT_OFFSET, biases + FC1_BIAS_OFFSET, true);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample fc1_out (10): " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << bo_fc1_out[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "FC1 error indexes: ";
    for(int i=0; i<NUM_FC1_OUTS; i++) {
        data_ap_fixed_t diff = hls::fabs(bo_fc1_out[i] - fc1_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1f)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_fc1_out[i] 
                        << ", expected " << fc1_out_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // FC2 Test
    // data_ap_fixed_t fc2_out_golden[fc2_out_size];
    // fc_golden<FC2_IN_DIM, FC2_OUT_DIM>(&outs_ref[FC1_OUT_OFFSET], &outs_ref[FC2_OUT_OFFSET], weights + FC2_WEIGHT_OFFSET, biases + FC2_BIAS_OFFSET, true);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "Sample fc2_out (10): " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << bo_fc2_out[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "FC2 error indexes: ";
    for(int i=0; i<NUM_FC2_OUTS; i++) {
        data_ap_fixed_t diff = hls::fabs(bo_fc2_out[i] - fc2_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1f)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_fc2_out[i] 
                        << ", expected " << fc2_out_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    global_errors += errors;
    std::cout << std::endl << std::endl;

    // FC3 Test
    // data_ap_fixed_t fc3_out_golden[fc3_out_size];
    // fc_golden<FC3_IN_DIM, FC3_OUT_DIM>(&outs_ref[FC2_OUT_OFFSET], &outs_ref[FC3_OUT_OFFSET], weights + FC3_WEIGHT_OFFSET, biases + FC3_BIAS_OFFSET, false);
    errors = 0;
    max_diff = 0.0f;

    std::cout << "fc3_out = [";
    for(int i=0; i<10; i++) {
        std::cout << bo_fc3_out[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "FC3 error indexes: ";
    for(int i=0; i<NUM_FC3_OUTS; i++) {
        data_ap_fixed_t diff = hls::fabs(bo_fc3_out[i] - fc3_out_golden[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > data_ap_fixed_t(0.1f)) {
            errors++;
            if (errors < 10) { // Limit error reporting to avoid flooding console
                std::cout << "Error at output " << i << ": got " << bo_fc3_out[i] 
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