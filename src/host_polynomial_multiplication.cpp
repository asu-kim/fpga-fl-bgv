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
#include "polynomial_multiplication.hpp"
#include "constants.hpp"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define DATA_SIZE POLYNOMIAL_DEGREE

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.addSwitch("--input_file", "-i", "input data file", "input.txt");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));
    std::string inputFile = parser.value("input_file");

    if (binaryFile.empty()) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    // Read input file
    std::ifstream infile(inputFile);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open input file '" << inputFile << "'" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<data_t> input1(DATA_SIZE);
    std::vector<data_t> input2(DATA_SIZE);

    data_t cpu_source_in[DATA_SIZE];

    std::cout << "Reading input from file: " << inputFile << std::endl;
    
    // Read up to DATA_SIZE pairs of numbers from the file
    for (int i = 0; i < DATA_SIZE; i++) {
        if (infile >> input1[i]) {
            input2[i] = input1[i];
            cpu_source_in[i] = input1[i];
        } else {
            if (i == 0) {
                std::cerr << "Error: Input file format is invalid. Expected pairs of numbers." << std::endl;
                return EXIT_FAILURE;
            }
            // If we reach end of file before reading DATA_SIZE pairs, fill the rest with zeros
            for (int j = i; j < DATA_SIZE; j++) {
                input1[j] = 0;
                input2[j] = 0;
                cpu_source_in[j] = 0;
            }
            break;
        }
    }

    // for (int i = 0; i < DATA_SIZE; i++) {
    //     printf("input1[%d] = %d.\n", i, input1[i]);
    //     printf("input2[%d] = %d.\n", i, input2[i]);
    // }

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    size_t vector_size_bytes = sizeof(data_t) * DATA_SIZE;

    auto krnl = xrt::kernel(device, uuid, "polynomial_multiplication");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_in1 = xrt::bo(device, vector_size_bytes, krnl.group_id(0));
    auto bo_in2 = xrt::bo(device, vector_size_bytes, krnl.group_id(1));
    auto bo_out = xrt::bo(device, vector_size_bytes, krnl.group_id(2));

    // Map the contents of the buffer object into host memory
    auto bo_in1_map = bo_in1.map<data_t*>();
    auto bo_in2_map = bo_in2.map<data_t*>();
    auto bo_out_map = bo_out.map<data_t*>();
    std::fill(bo_in1_map, bo_in1_map + DATA_SIZE, 0);
    std::fill(bo_in2_map, bo_in2_map + DATA_SIZE, 0);
    std::fill(bo_out_map, bo_out_map + DATA_SIZE, 0);

    // Create the test data
    // int bufReference[DATA_SIZE];
    // for (int i = 0; i < DATA_SIZE; ++i) {
    //     bo0_map[i] = i;
    //     bo1_map[i] = i;
    //     bufReference[i] = bo0_map[i] + bo1_map[i];
    // }
    
    // Copy input data to mapped buffers
    for (int i = 0; i < DATA_SIZE; i++) {
        bo_in1_map[i] = input1[i];
        bo_in2_map[i] = input2[i];
    }

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";

    bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel\n";
    auto run = krnl(bo_in1, bo_in2, bo_out);
    run.wait();

    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    data_t cpu_source_out[DATA_SIZE];
    // ntt_reference(cpu_source_in, cpu_source_out, DATA_SIZE, CIPHERTEXT_MODULUS, PRIMITIVE_N_TH_ROOT_OF_UNITY);
    for (int i = 0; i < DATA_SIZE; i++) {
        // std::cout << "i = " << i << " CPU result = " << fmod(cpu_source_in1[i], cpu_source_in2[i])
        // << " Device result = " << bo_out_map[i] << std::endl;
        // std::cout << "i = " << i << " CPU result = " << cpu_source_out[i]
        // << " Device result = " << bo_out_map[i] << std::endl;
        std::cout << "i = " << i << " CPU result = " << bo_out_map[i] << std::endl;
    }
    std::cout << "Test ended\n";
    return 0;
}
