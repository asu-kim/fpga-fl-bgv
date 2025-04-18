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
#include "encryption.hpp"
#include "constants.hpp"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define DATA_SIZE POLYNOMIAL_DEGREE

std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));

void polynomial_multiplication_reference(std::vector<data_t> input1, std::vector<data_t> input2, std::vector<data_t> output) {
    data_t n = POLYNOMIAL_DEGREE;
    data_t n_inv = INVERSE_POLYNOMIAL_DEGREE;
    data_t q = CIPHERTEXT_MODULUS;
    data_t w = PRIMITIVE_N_TH_ROOT_OF_UNITY;
    data_t w_inv = INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY;

    // data_t temp[POLYNOMIAL_DEGREE];
    for (int i = 0; i < n; i++) {
        int pos = 0;
        for (int j = 0; j <= i; j++) {
            pos += (input1[j] * input2[i - j]) % q;
        }
        int neg = 0;
        for (int j = i + 1; j < n; j++) {
            neg += (input1[j] * input2[n + i - j]) % q;
        }
        output[i] = (((pos - neg) % q) + q) % q;
    }
}

void generate_key(std::vector<data_t> private_key, std::vector<data_t> public_key1, std::vector<data_t> public_key2)
{
    int n = DATA_SIZE;
    int p = PLAINTEXT_MODULUS;
    int q = CIPHERTEXT_MODULUS;

    std::vector<data_t> a_prime(DATA_SIZE);
    std::vector<data_t> error(DATA_SIZE);

    std::uniform_int_distribution<int> dist1(-1, 1);

    for (int i = 0; i < n; i++)
    {
        private_key[i] = dist1(rng);
        error[i] = dist1(rng);
    }

    std::uniform_int_distribution<int> dist2(0, q - 1);

    for (int i = 0; i < n; i++)
    {
        a_prime[i] = dist2(rng);
    }

    std::vector<data_t> temp(DATA_SIZE);
    polynomial_multiplication_reference(a_prime, private_key, temp);

    for (int i = 0; i < n; i++)
    {
        public_key1[i] = temp[i] + p * error[i];
        public_key2[i] = -a_prime[i];
    }
}

void random_sampling(std::vector<data_t> array)
{
    std::uniform_int_distribution<int> dist1(-1, 1);
    for (int i = 0; i < DATA_SIZE; i++)
    {
        array[i] = dist1(rng);
    }
}

int main(int argc, char **argv)
{
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

    if (binaryFile.empty())
    {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    // Read input file
    std::ifstream infile(inputFile);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file '" << inputFile << "'" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<data_t> plaintext(DATA_SIZE);
    data_t cpu_source_in[DATA_SIZE];

    std::cout << "Reading input from file: " << inputFile << std::endl;

    // Read up to DATA_SIZE pairs of numbers from the file
    for (int i = 0; i < DATA_SIZE; i++)
    {
        if (infile >> plaintext[i])
        {
            cpu_source_in[i] = plaintext[i];
        }
        else
        {
            if (i == 0)
            {
                std::cerr << "Error: Input file format is invalid. Expected pairs of numbers." << std::endl;
                return EXIT_FAILURE;
            }
            // If we reach end of file before reading DATA_SIZE pairs, fill the rest with zeros
            for (int j = i; j < DATA_SIZE; j++)
            {
                plaintext[j] = 0;
                cpu_source_in[j] = 0;
            }
            break;
        }
    }

    std::vector<data_t> private_key(DATA_SIZE);
    std::vector<data_t> public_key1(DATA_SIZE);
    std::vector<data_t> public_key2(DATA_SIZE);

    generate_key(private_key, public_key1, public_key2);

    std::vector<data_t> error1(DATA_SIZE);
    std::vector<data_t> error2(DATA_SIZE);
    std::vector<data_t> r(DATA_SIZE);

    random_sampling(error1);
    random_sampling(error2);
    random_sampling(r);

    // for (int i = 0; i < DATA_SIZE; i++) {
    //     printf("plaintext[%d] = %d.\n", i, plaintext[i]);
    //     printf("input2[%d] = %d.\n", i, input2[i]);
    // }

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    size_t vector_size_bytes = sizeof(data_t) * DATA_SIZE;

    auto krnl = xrt::kernel(device, uuid, "top_encryption_decryption_test");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_plaintext = xrt::bo(device, vector_size_bytes, krnl.group_id(0));
    auto bo_private_key = xrt::bo(device, vector_size_bytes, krnl.group_id(1));
    auto bo_public_key1 = xrt::bo(device, vector_size_bytes, krnl.group_id(2));
    auto bo_public_key2 = xrt::bo(device, vector_size_bytes, krnl.group_id(2));
    auto bo_error1 = xrt::bo(device, vector_size_bytes, krnl.group_id(3));
    auto bo_error2 = xrt::bo(device, vector_size_bytes, krnl.group_id(3));
    auto bo_r = xrt::bo(device, vector_size_bytes, krnl.group_id(3));
    auto bo_out = xrt::bo(device, vector_size_bytes, krnl.group_id(4));

    // Map the contents of the buffer object into host memory
    auto bo_plaintext_map = bo_plaintext.map<data_t *>();
    auto bo_private_key_map = bo_private_key.map<data_t *>();
    auto bo_public_key1_map = bo_public_key1.map<data_t *>();
    auto bo_public_key2_map = bo_public_key2.map<data_t *>();
    auto bo_error1_map = bo_error1.map<data_t *>();
    auto bo_error2_map = bo_error2.map<data_t *>();
    auto bo_r_map = bo_r.map<data_t *>();
    auto bo_out_map = bo_out.map<data_t *>();
    std::fill(bo_plaintext_map, bo_plaintext_map + DATA_SIZE, 0);
    std::fill(bo_private_key_map, bo_private_key_map + DATA_SIZE, 0);
    std::fill(bo_public_key1_map, bo_public_key1_map + DATA_SIZE, 0);
    std::fill(bo_public_key2_map, bo_public_key2_map + DATA_SIZE, 0);
    std::fill(bo_error1_map, bo_error1_map + DATA_SIZE, 0);
    std::fill(bo_error2_map, bo_error2_map + DATA_SIZE, 0);
    std::fill(bo_r_map, bo_r_map + DATA_SIZE, 0);
    std::fill(bo_out_map, bo_out_map + DATA_SIZE, 0);

    // Create the test data
    // int bufReference[DATA_SIZE];
    // for (int i = 0; i < DATA_SIZE; ++i) {
    //     bo0_map[i] = i;
    //     bo1_map[i] = i;
    //     bufReference[i] = bo0_map[i] + bo1_map[i];
    // }

    // Copy input data to mapped buffers
    for (int i = 0; i < DATA_SIZE; i++)
    {
        bo_plaintext_map[i] = plaintext[i];
        bo_private_key_map[i] = private_key[i];
        bo_public_key1_map[i] = public_key1[i];
        bo_public_key2_map[i] = public_key2[i];
        bo_error1_map[i] = error1[i];
        bo_error2_map[i] = error2[i];
        bo_r_map[i] = r[i];
    }

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";

    bo_plaintext.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_private_key.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_public_key1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_public_key2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_error1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_error2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_r.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel\n";
    auto run = krnl(bo_error1, bo_error2, bo_r, bo_private_key, bo_public_key1, bo_public_key2, bo_plaintext, bo_out);
    run.wait();

    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    data_t cpu_source_out[DATA_SIZE];
    // ntt_reference(cpu_source_in, cpu_source_out, DATA_SIZE, CIPHERTEXT_MODULUS, PRIMITIVE_N_TH_ROOT_OF_UNITY);
    for (int i = 0; i < DATA_SIZE; i++)
    {
        // std::cout << "i = " << i << " CPU result = " << fmod(cpu_source_in1[i], cpu_source_in2[i])
        // << " Device result = " << bo_out_map[i] << std::endl;
        // std::cout << "i = " << i << " CPU result = " << cpu_source_out[i]
        // << " Device result = " << bo_out_map[i] << std::endl;
        std::cout << "i = " << i << " Device result = " << bo_out_map[i] << std::endl;
    }
    std::cout << "Test ended\n";
    return 0;
}
