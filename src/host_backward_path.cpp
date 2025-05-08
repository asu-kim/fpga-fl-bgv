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
#include "lenet5/backward_path.h"
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
    size_t weights_size = TOTAL_WEIGHTS_SIZE;
    size_t biases_size = TOTAL_BIASES_SIZE;
    size_t outputs_size = TOTAL_OUTS_SIZE;
    size_t labels_size = 10;

    size_t conv1_weight_size = CONV1_OUT_CH * CONV1_IN_CH * KERNEL_SIZE * KERNEL_SIZE;
    size_t conv1_bias_size = CONV1_OUT_CH;
    size_t conv1_out_size = CONV1_OUT_CH * CONV1_OUT_ROWS * CONV1_OUT_COLS;

    size_t pool1_out_size = CONV2_IN_CH * CONV2_IN_ROWS * CONV2_IN_COLS;

    size_t conv2_weight_size = CONV2_OUT_CH * CONV2_IN_CH * KERNEL_SIZE * KERNEL_SIZE;
    size_t conv2_bias_size = CONV2_OUT_CH;
    size_t conv2_out_size = CONV2_OUT_CH * CONV2_OUT_ROWS * CONV2_OUT_COLS;

    size_t pool2_out_size = FC1_IN_DIM;

    size_t fc1_weight_size = FC1_IN_DIM * FC1_OUT_DIM;
    size_t fc1_bias_size = FC1_OUT_DIM;
    size_t fc1_out_size = FC1_OUT_DIM;

    size_t fc2_weight_size = FC2_IN_DIM * FC2_OUT_DIM;
    size_t fc2_bias_size = FC2_OUT_DIM;
    size_t fc2_out_size = FC2_OUT_DIM;

    size_t fc3_weight_size = FC3_IN_DIM * FC3_OUT_DIM;
    size_t fc3_bias_size = FC3_OUT_DIM;
    size_t fc3_out_size = FC3_OUT_DIM;

    size_t label_size = FC3_OUT_DIM;

    // Create kernels
    auto backward_krnl = xrt::kernel(device, uuid, "backward_path");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_in_data = xrt::bo(device, sizeof(float)*in_size, backward_krnl.group_id(0));
    auto bo_weights = xrt::bo(device, sizeof(float)*weights_size, backward_krnl.group_id(1));
    auto bo_biases = xrt::bo(device, sizeof(float)*biases_size, backward_krnl.group_id(2));
    auto bo_outputs = xrt::bo(device, sizeof(float)*outputs_size, backward_krnl.group_id(3));
    auto bo_labels = xrt::bo(device, sizeof(float)*labels_size, backward_krnl.group_id(4));

    auto bo_updated_weights = xrt::bo(device, sizeof(float)*weights_size, backward_krnl.group_id(5));
    auto bo_updated_biases = xrt::bo(device, sizeof(float)*biases_size, backward_krnl.group_id(6));

    // Map buffers to host memory
    auto bo_in_data_map = bo_in_data.map<float *>();
    auto bo_weights_map = bo_weights.map<float *>();
    auto bo_biases_map = bo_biases.map<float *>();
    auto bo_outputs_map = bo_outputs.map<float *>();
    auto bo_labels_map = bo_labels.map<float *>();

    auto bo_updated_weights_map = bo_updated_weights.map<float *>();
    auto bo_updated_biases_map = bo_updated_biases.map<float *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_in_data_map, bo_in_data_map + in_size, 0);
    std::fill(bo_weights_map, bo_weights_map + weights_size, 0);
    std::fill(bo_biases_map, bo_biases_map + biases_size, 0);
    std::fill(bo_outputs_map, bo_outputs_map + outputs_size, 0);
    std::fill(bo_labels_map, bo_labels_map + labels_size, 0);

    std::fill(bo_updated_weights_map, bo_updated_weights_map + weights_size, 0);
    std::fill(bo_updated_biases_map, bo_updated_biases_map + biases_size, 0);

    std::cout << "Initialize weight and bias\n";
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

    float conv1_out_golden[conv1_out_size];
    float pool1_out_golden[pool1_out_size];
    float conv2_out_golden[conv2_out_size];
    float pool2_out_golden[pool2_out_size];
    float fc1_out_golden[fc1_out_size];
    float fc2_out_golden[fc2_out_size];
    float fc3_out_golden[fc3_out_size];

    std::cout << "Run forward path\n";
    // Run forward path on the CPU
    forward_golden(
        bo_in_data_map,
        bo_weights_map,
        bo_biases_map,
        conv1_out_golden,
        pool1_out_golden,
        bo_weights_map,
        bo_biases_map,
        conv2_out_golden,
        pool2_out_golden,
        bo_weights_map,
        bo_biases_map,
        fc1_out_golden,
        bo_weights_map,
        bo_biases_map,
        fc2_out_golden,
        bo_weights_map,
        bo_biases_map,
        fc3_out_golden
    );
    for(int i = 0; i < conv1_out_size; i++) {
        bo_conv1_out_map[i] = conv1_out_golden[i];
    }
    for(int i = 0; i < pool1_out_size; i++) {
        bo_pool1_out_map[i] = pool1_out_golden[i];
    }
    for(int i = 0; i < conv2_out_size; i++) {
        bo_conv2_out_map[i] = conv2_out_golden[i];
    }
    for(int i = 0; i < pool2_out_size; i++) {
        bo_pool2_out_map[i] = pool2_out_golden[i];
    }
    for(int i = 0; i < fc1_out_size; i++) {
        bo_fc1_out_map[i] = fc1_out_golden[i];
    }
    for(int i = 0; i < fc2_out_size; i++) {
        bo_fc2_out_map[i] = fc2_out_golden[i];
    }
    for(int i = 0; i < fc3_out_size; i++) {
        bo_fc3_out_map[i] = fc3_out_golden[i];
    }
    
    // Initialize label (one-hot encoding for class 7)
    for(int i = 0; i < FC3_OUT_DIM; i++) {
        bo_label_map[i] = (i == 7) ? 1.0f : 0.0f;
    }

    std::cout << "Sync buffers\n";
    // Sync all buffers to device

    float loss = 0;

    // Run backward path
    std::cout << "Running the kernel\n";
    auto hw_start = std::chrono::high_resolution_clock::now();

    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv1_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv1_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv1_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_pool1_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_conv2_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv2_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_conv2_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_pool2_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_fc1_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc1_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc1_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_fc2_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc2_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc2_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    bo_fc3_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc3_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fc3_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_label.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = backward_krnl(
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
        bo_fc3_out,
        bo_label,
        bo_conv1_updated_weight,
        bo_conv1_updated_bias,
        bo_conv2_updated_weight,
        bo_conv2_updated_bias,
        bo_fc1_updated_weight,
        bo_fc1_updated_bias,
        bo_fc2_updated_weight,
        bo_fc2_updated_bias,
        bo_fc3_updated_weight,
        bo_fc3_updated_bias,
        loss
    );
    auto state = run.wait(std::chrono::seconds(10)); // Add timeout
    if (state != ERT_CMD_STATE_COMPLETED) {
        std::cout << "Kernel execution timed out or failed" << std::endl;
        // Handle error
    }
    // run.wait();

    // Read output from stream
    bo_conv1_updated_weight.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_conv1_updated_bias.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_conv2_updated_weight.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_conv2_updated_bias.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc1_updated_weight.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc1_updated_bias.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc2_updated_weight.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc2_updated_bias.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc3_updated_weight.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_fc3_updated_bias.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto hw_end = std::chrono::high_resolution_clock::now();
    auto hw_duration = std::chrono::duration<double, std::milli>(hw_end - hw_start).count();
    std::cout << "Hardware kernel execution time: " << hw_duration << " ms" << std::endl;
    std::cout << "Done backward\n";

    float conv1_updated_weight_golden[conv1_weight_size];
    float conv1_updated_bias_golden[conv1_bias_size];
    float conv2_updated_weight_golden[conv2_weight_size];
    float conv2_updated_bias_golden[conv2_bias_size];
    float fc1_updated_weight_golden[fc1_weight_size];
    float fc1_updated_bias_golden[fc1_bias_size];
    float fc2_updated_weight_golden[fc2_weight_size];
    float fc2_updated_bias_golden[fc2_bias_size];
    float fc3_updated_weight_golden[fc3_weight_size];
    float fc3_updated_bias_golden[fc3_bias_size];
    float loss_golden = 0;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    // Run goldenerence
    backward_golden(
        bo_in_data_map,
        bo_conv1_weight_map,
        bo_conv1_bias_map,
        bo_conv1_out_map,
        bo_pool1_out_map,
        bo_conv2_weight_map,
        bo_conv2_bias_map,
        bo_conv2_out_map,
        bo_pool2_out_map,
        bo_fc1_weight_map,
        bo_fc1_bias_map,
        bo_fc1_out_map,
        bo_fc2_weight_map,
        bo_fc2_bias_map,
        bo_fc2_out_map,
        bo_fc3_weight_map,
        bo_fc3_bias_map,
        bo_fc3_out_map,
        bo_label_map,
        conv1_updated_weight_golden,
        conv1_updated_bias_golden,
        conv2_updated_weight_golden,
        conv2_updated_bias_golden,
        fc1_updated_weight_golden,
        fc1_updated_bias_golden,
        fc2_updated_weight_golden,
        fc2_updated_bias_golden,
        fc3_updated_weight_golden,
        fc3_updated_bias_golden,
        loss_golden
    );

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU reference execution time: " << cpu_duration << " ms" << std::endl;

    // Output loss value
    std::cout << "Loss from FPGA: " << loss << std::endl;
    std::cout << "Loss from golden: " << loss_golden << std::endl;
    std::cout << "Loss difference: " << std::fabs(loss - loss_golden) << std::endl;
    std::cout << std::endl;

    // Define tolerance threshold for float comparisons
    const float tolerance = 0.01f;

    // Verify Conv1 Updated Weights
    int conv1_weight_errors = 0;
    float conv1_weight_max_diff = 0.0f;
    std::cout << "Conv1 Updated Weights - checking " << conv1_weight_size << " values" << std::endl;
    for(int i = 0; i < conv1_weight_size; i++) {
        float diff = std::fabs(bo_conv1_updated_weight_map[i] - conv1_updated_weight_golden[i]);
        conv1_weight_max_diff = std::max(conv1_weight_max_diff, diff);
        
        if (diff > tolerance) {
            conv1_weight_errors++;
            if (conv1_weight_errors < 10) { // Limit error reporting
                std::cout << "Error at weight " << i << ": got " << bo_conv1_updated_weight_map[i] 
                        << ", expected " << conv1_updated_weight_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "Conv1 Updated Weights - errors: " << conv1_weight_errors 
            << ", max diff: " << conv1_weight_max_diff << std::endl << std::endl;

    // Verify Conv1 Updated Biases
    int conv1_bias_errors = 0;
    float conv1_bias_max_diff = 0.0f;
    std::cout << "Conv1 Updated Biases - checking " << conv1_bias_size << " values" << std::endl;
    for(int i = 0; i < conv1_bias_size; i++) {
        float diff = std::fabs(bo_conv1_updated_bias_map[i] - conv1_updated_bias_golden[i]);
        conv1_bias_max_diff = std::max(conv1_bias_max_diff, diff);
        
        if (diff > tolerance) {
            conv1_bias_errors++;
            if (conv1_bias_errors < 10) { // Limit error reporting
                std::cout << "Error at bias " << i << ": got " << bo_conv1_updated_bias_map[i] 
                        << ", expected " << conv1_updated_bias_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "Conv1 Updated Biases - errors: " << conv1_bias_errors 
            << ", max diff: " << conv1_bias_max_diff << std::endl << std::endl;

    // Verify Conv2 Updated Weights
    int conv2_weight_errors = 0;
    float conv2_weight_max_diff = 0.0f;
    std::cout << "Conv2 Updated Weights - checking " << conv2_weight_size << " values" << std::endl;
    for(int i = 0; i < conv2_weight_size; i++) {
        float diff = std::fabs(bo_conv2_updated_weight_map[i] - conv2_updated_weight_golden[i]);
        conv2_weight_max_diff = std::max(conv2_weight_max_diff, diff);
        
        if (diff > tolerance) {
            conv2_weight_errors++;
            if (conv2_weight_errors < 10) { // Limit error reporting
                std::cout << "Error at weight " << i << ": got " << bo_conv2_updated_weight_map[i] 
                        << ", expected " << conv2_updated_weight_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "Conv2 Updated Weights - errors: " << conv2_weight_errors 
            << ", max diff: " << conv2_weight_max_diff << std::endl << std::endl;

    // Verify Conv2 Updated Biases
    int conv2_bias_errors = 0;
    float conv2_bias_max_diff = 0.0f;
    std::cout << "Conv2 Updated Biases - checking " << conv2_bias_size << " values" << std::endl;
    for(int i = 0; i < conv2_bias_size; i++) {
        float diff = std::fabs(bo_conv2_updated_bias_map[i] - conv2_updated_bias_golden[i]);
        conv2_bias_max_diff = std::max(conv2_bias_max_diff, diff);
        
        if (diff > tolerance) {
            conv2_bias_errors++;
            if (conv2_bias_errors < 10) { // Limit error reporting
                std::cout << "Error at bias " << i << ": got " << bo_conv2_updated_bias_map[i] 
                        << ", expected " << conv2_updated_bias_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "Conv2 Updated Biases - errors: " << conv2_bias_errors 
            << ", max diff: " << conv2_bias_max_diff << std::endl << std::endl;

    // Verify FC1 Updated Weights
    int fc1_weight_errors = 0;
    float fc1_weight_max_diff = 0.0f;
    std::cout << "FC1 Updated Weights - checking " << fc1_weight_size << " values" << std::endl;
    for(int i = 0; i < fc1_weight_size; i++) {
        float diff = std::fabs(bo_fc1_updated_weight_map[i] - fc1_updated_weight_golden[i]);
        fc1_weight_max_diff = std::max(fc1_weight_max_diff, diff);
        
        if (diff > tolerance) {
            fc1_weight_errors++;
            if (fc1_weight_errors < 10) { // Limit error reporting
                std::cout << "Error at weight " << i << ": got " << bo_fc1_updated_weight_map[i] 
                        << ", expected " << fc1_updated_weight_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "FC1 Updated Weights - errors: " << fc1_weight_errors 
            << ", max diff: " << fc1_weight_max_diff << std::endl << std::endl;

    // Verify FC1 Updated Biases
    int fc1_bias_errors = 0;
    float fc1_bias_max_diff = 0.0f;
    std::cout << "FC1 Updated Biases - checking " << fc1_bias_size << " values" << std::endl;
    for(int i = 0; i < fc1_bias_size; i++) {
        float diff = std::fabs(bo_fc1_updated_bias_map[i] - fc1_updated_bias_golden[i]);
        fc1_bias_max_diff = std::max(fc1_bias_max_diff, diff);
        
        if (diff > tolerance) {
            fc1_bias_errors++;
            if (fc1_bias_errors < 10) { // Limit error reporting
                std::cout << "Error at bias " << i << ": got " << bo_fc1_updated_bias_map[i] 
                        << ", expected " << fc1_updated_bias_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "FC1 Updated Biases - errors: " << fc1_bias_errors 
            << ", max diff: " << fc1_bias_max_diff << std::endl << std::endl;

    // Verify FC2 Updated Weights
    int fc2_weight_errors = 0;
    float fc2_weight_max_diff = 0.0f;
    std::cout << "FC2 Updated Weights - checking " << fc2_weight_size << " values" << std::endl;
    for(int i = 0; i < fc2_weight_size; i++) {
        float diff = std::fabs(bo_fc2_updated_weight_map[i] - fc2_updated_weight_golden[i]);
        fc2_weight_max_diff = std::max(fc2_weight_max_diff, diff);
        
        if (diff > tolerance) {
            fc2_weight_errors++;
            if (fc2_weight_errors < 10) { // Limit error reporting
                std::cout << "Error at weight " << i << ": got " << bo_fc2_updated_weight_map[i] 
                        << ", expected " << fc2_updated_weight_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "FC2 Updated Weights - errors: " << fc2_weight_errors 
            << ", max diff: " << fc2_weight_max_diff << std::endl << std::endl;

    // Verify FC2 Updated Biases
    int fc2_bias_errors = 0;
    float fc2_bias_max_diff = 0.0f;
    std::cout << "FC2 Updated Biases - checking " << fc2_bias_size << " values" << std::endl;
    for(int i = 0; i < fc2_bias_size; i++) {
        float diff = std::fabs(bo_fc2_updated_bias_map[i] - fc2_updated_bias_golden[i]);
        fc2_bias_max_diff = std::max(fc2_bias_max_diff, diff);
        
        if (diff > tolerance) {
            fc2_bias_errors++;
            if (fc2_bias_errors < 10) { // Limit error reporting
                std::cout << "Error at bias " << i << ": got " << bo_fc2_updated_bias_map[i] 
                        << ", expected " << fc2_updated_bias_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "FC2 Updated Biases - errors: " << fc2_bias_errors 
            << ", max diff: " << fc2_bias_max_diff << std::endl << std::endl;

    // Verify FC3 Updated Weights
    int fc3_weight_errors = 0;
    float fc3_weight_max_diff = 0.0f;
    std::cout << "FC3 Updated Weights - checking " << fc3_weight_size << " values" << std::endl;
    for(int i = 0; i < fc3_weight_size; i++) {
        float diff = std::fabs(bo_fc3_updated_weight_map[i] - fc3_updated_weight_golden[i]);
        fc3_weight_max_diff = std::max(fc3_weight_max_diff, diff);
        
        if (diff > tolerance) {
            fc3_weight_errors++;
            if (fc3_weight_errors < 10) { // Limit error reporting
                std::cout << "Error at weight " << i << ": got " << bo_fc3_updated_weight_map[i] 
                        << ", expected " << fc3_updated_weight_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "FC3 Updated Weights - errors: " << fc3_weight_errors 
            << ", max diff: " << fc3_weight_max_diff << std::endl << std::endl;

    // Verify FC3 Updated Biases
    int fc3_bias_errors = 0;
    float fc3_bias_max_diff = 0.0f;
    std::cout << "FC3 Updated Biases - checking " << fc3_bias_size << " values" << std::endl;
    for(int i = 0; i < fc3_bias_size; i++) {
        float diff = std::fabs(bo_fc3_updated_bias_map[i] - fc3_updated_bias_golden[i]);
        fc3_bias_max_diff = std::max(fc3_bias_max_diff, diff);
        
        if (diff > tolerance) {
            fc3_bias_errors++;
            if (fc3_bias_errors < 10) { // Limit error reporting
                std::cout << "Error at bias " << i << ": got " << bo_fc3_updated_bias_map[i] 
                        << ", expected " << fc3_updated_bias_golden[i] 
                        << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "FC3 Updated Biases - errors: " << fc3_bias_errors 
            << ", max diff: " << fc3_bias_max_diff << std::endl << std::endl;

    // Calculate total errors for backward path
    int backward_total_errors = conv1_weight_errors + conv1_bias_errors +
                            conv2_weight_errors + conv2_bias_errors +
                            fc1_weight_errors + fc1_bias_errors +
                            fc2_weight_errors + fc2_bias_errors +
                            fc3_weight_errors + fc3_bias_errors;

    std::cout << "Backward Path Total Errors: " << backward_total_errors << std::endl;

    // Print summary of results
    std::cout << "\n=== TEST SUMMARY ===" << std::endl;
    std::cout << "Backward Path Errors: " << backward_total_errors << std::endl;
    std::cout << "Loss Difference: " << std::fabs(loss - loss_golden) << std::endl;

    // Determine overall test status
    if (backward_total_errors == 0 && std::fabs(loss - loss_golden) < tolerance) {
        std::cout << "TEST PASSED: All values matched within tolerance!" << std::endl;
    } else {
        std::cout << "TEST FAILED: " << backward_total_errors << " errors detected." << std::endl;
        
        if (backward_total_errors > 0) {
            std::cout << "Backward path has errors in: ";
            if (conv1_weight_errors > 0 || conv1_bias_errors > 0) std::cout << "Conv1, ";
            if (conv2_weight_errors > 0 || conv2_bias_errors > 0) std::cout << "Conv2, ";
            if (fc1_weight_errors > 0 || fc1_bias_errors > 0) std::cout << "FC1, ";
            if (fc2_weight_errors > 0 || fc2_bias_errors > 0) std::cout << "FC2, ";
            if (fc3_weight_errors > 0 || fc3_bias_errors > 0) std::cout << "FC3, ";
            std::cout << std::endl;
        }
    }
}
