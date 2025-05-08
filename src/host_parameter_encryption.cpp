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
#include "BGV/encryption.hpp"
#include "BGV/parameter_processing.hpp"

#include "keys.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "encrypted_weights_bias.h"
#include "constants.hpp"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define DATA_SIZE POLYNOMIAL_DEGREE

// std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
std::mt19937 rng(42);

void print_float_array(const float* arr, int size, const std::string& name) {
    std::cout << name << " (first 10 elements): ";
    for(int i=0; i < std::min(10, size); i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

void print_data_t_array(const data_t* arr, int size, const std::string& name) {
    std::cout << name << " (first 10 elements): ";
    for(int i=0; i < std::min(10, size); i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

void print_data_t_array_last(const data_t* arr, int size, const std::string& name) {
    std::cout << name << " (last 128 elements): ";
    int start_idx = size >= 128 ? size - 128 : 0;
    for(int i = 0; i < std::min(128, size); i++) {
        std::cout << arr[start_idx + i] << ", ";
    }
    std::cout << std::endl << std::endl;
}

int modulo(int a, int b) {
    int c = (a % b + b) % b;
    if (c < b /2) {
        return c;
    } else {
        return c - b;
    }
}

// Reference implementation for verification
void ntt_reference(data_t* input, data_t* output) {
    int n = POLYNOMIAL_DEGREE;
    int q = CIPHERTEXT_MODULUS;
    int w = PRIMITIVE_N_TH_ROOT_OF_UNITY;
    // Make a copy of input
    data_t temp[POLYNOMIAL_DEGREE];
    for (int i = 0; i < n; i++) {
        temp[i] = input[i];
    }

    // Perform NTT using the same algorithm
    for (int m = n / 2; m >= 1; m /= 2) {
        for (int j = 0; j < m; j++) {
            int exp = (j * n) / (2 * m);
            data_t w_m = 1;
            for (int k = 0; k < exp; k++) {
                w_m = (w_m * w) % q;
            }
            // printf("test w_m = %d\n", w_m);

            for (int i = j; i < n; i += 2 * m) {
                data_t t1 = temp[i];
                data_t t2 = temp[i + m];
                temp[i] = (t1 + t2) % q;
                temp[i + m] = (w_m * ((t1 - t2 + q) % q)) % q;
            }
        }
    }
    
    // Apply bit reversal
    for (int i = 0; i < n; i++) {
        output[BIT_REVERSE_LUT[i]] = temp[i];
    }
}

// Reference implementation for INNT verification
void intt_reference(data_t* input, data_t* output) {
    int n = POLYNOMIAL_DEGREE;
    int q = CIPHERTEXT_MODULUS;
    int w_inv = INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY;
    int n_inv = INVERSE_POLYNOMIAL_DEGREE;
    // Make a copy of input with bit-reversal
    data_t temp[n];
    for (int i = 0; i < n; i++) {
        temp[i] = input[i];
    }

    // INTT algorithm
    for (int m = n / 2; m >= 1; m /= 2) {
        for (int j = 0; j < m; j++) {
            int exp = (j * n) / (2 * m);
            data_t w_m = 1;
            for (int k = 0; k < exp; k++) {
                w_m = (w_m * w_inv) % q;
            }
            
            for (int i = j; i < n; i += 2 * m) {
                data_t t1 = temp[i];
                data_t t2 = temp[i + m];
                temp[i] = (t1 + t2) % q;
                
                // Correct computation for INTT
                data_t diff = (t1 - t2 + q) % q;  // Ensure positive result
                temp[i + m] = (diff * w_m) % q;
            }
        }
    }
    
    // Apply scaling by n_inv
    for (int i = 0; i < n; i++) {
        output[BIT_REVERSE_LUT[i]] = (temp[i] * n_inv) % q;
    }
}

void polynomial_multiplication_reference(data_t* input1, data_t* input2, data_t* output) {
    data_t n = POLYNOMIAL_DEGREE;
    data_t n_inv = INVERSE_POLYNOMIAL_DEGREE;
    data_t q = CIPHERTEXT_MODULUS;
    data_t w = PRIMITIVE_N_TH_ROOT_OF_UNITY;
    data_t w_inv = INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY;

    data_t in1_tilde[POLYNOMIAL_DEGREE];
    data_t in2_tilde[POLYNOMIAL_DEGREE];
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        in1_tilde[i] = (E_POWERS_LUT[i] * input1[i]) % q;
        in2_tilde[i] = (E_POWERS_LUT[i] * input2[i]) % q;
    }

    data_t transformed_in1_tilde[POLYNOMIAL_DEGREE];
    data_t transformed_in2_tilde[POLYNOMIAL_DEGREE];
    ntt_reference(in1_tilde, transformed_in1_tilde);
    ntt_reference(in2_tilde, transformed_in2_tilde);

    data_t transformed_out_tilde[POLYNOMIAL_DEGREE];
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        transformed_out_tilde[i] = (transformed_in1_tilde[i] * transformed_in2_tilde[i]) % q;
    }

    data_t out_tilde[POLYNOMIAL_DEGREE];
    intt_reference(transformed_out_tilde, out_tilde);

    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        output[i] = (E_INV_POWERS_LUT[i] * out_tilde[i]) % q;
    }
}

void encryption_reference(data_t* error1, data_t* error2, data_t* r, data_t* public_key0, data_t* public_key1, data_t* plaintext, data_t* ciphertext0, data_t* ciphertext1) {
    int n = POLYNOMIAL_DEGREE;
    int p = PLAINTEXT_MODULUS;
    int q = CIPHERTEXT_MODULUS;
    
    data_t temp1[n];
    data_t temp2[n];

    polynomial_multiplication_reference(public_key0, r, temp1);
    polynomial_multiplication_reference(public_key1, r, temp2);

    // print_data_t_array(temp1, 128, "temp1");
    // print_data_t_array(temp2, 128, "temp2");
    // std::cout << std::endl;

    for(int i = 0; i < n; i++) {
        ciphertext0[i] = (plaintext[i] + p * error1[i] + temp1[i] + q) % q;
        ciphertext1[i] = (p * error2[i] + temp2[i] + q) % q;
    }
}

void decryption_reference(data_t* private_key, data_t* ciphertext1, data_t* ciphertext2, data_t* plaintext){
    int n = POLYNOMIAL_DEGREE;
    int p = PLAINTEXT_MODULUS;
    int q = CIPHERTEXT_MODULUS;

    data_t temp[n];
    polynomial_multiplication_reference(ciphertext2, private_key, temp);

    for(int i = 0; i < n; i++) {
        data_t intermittent = modulo(ciphertext1[i] + temp[i], q);
        // printf("intermittent = %d\n", intermittent);
        plaintext[i] = modulo(intermittent, p);
    }
}

void parameter_encryption_reference(
    float pt[POLYNOMIAL_DEGREE],
    float scale,
    float zp,
    data_t errors[POLYNOMIAL_DEGREE*3],
    data_t pk0[POLYNOMIAL_DEGREE],
    data_t pk1[POLYNOMIAL_DEGREE],

    data_t ct0[POLYNOMIAL_DEGREE],
    data_t ct1[POLYNOMIAL_DEGREE]
) {
    // Quantize
    data_t quantized_pt[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        float quantized = pt[i] / scale+ zp;
        quantized = (quantized > 127) ? 127 : ((quantized < -128) ? -128 : quantized);
        quantized_pt[i] = (data_t) quantized;
    }

    data_t error0[POLYNOMIAL_DEGREE];
    data_t ciphertext0[POLYNOMIAL_DEGREE];
    data_t r[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        error0[i] = errors[i];
    }

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        ciphertext0[i] = errors[POLYNOMIAL_DEGREE + i];
    }

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        r[i] = errors[2*POLYNOMIAL_DEGREE + i];
    }

    encryption_reference(
        error0,
        ciphertext0,
        r,
        pk0,
        pk1,
        quantized_pt, 
        ct0,
        ct1
    );
}

void parameter_decryption_reference(
    data_t sk[POLYNOMIAL_DEGREE*3],
    data_t ct0[POLYNOMIAL_DEGREE],
    data_t ct1[POLYNOMIAL_DEGREE],
    float scale,
    float zp,

    float pt[POLYNOMIAL_DEGREE]
) {
    // Dequantize
    data_t quantized_pt[POLYNOMIAL_DEGREE];
    decryption_reference(
        sk, 
        ct0, 
        ct1, 
        quantized_pt
    );

    print_data_t_array(sk, POLYNOMIAL_DEGREE, "sk");
    print_data_t_array(ct0, POLYNOMIAL_DEGREE, "ct0");
    print_data_t_array(ct1, POLYNOMIAL_DEGREE, "ct1");
    print_data_t_array(quantized_pt, POLYNOMIAL_DEGREE, "quantized_pt");

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        pt[i] = (quantized_pt[i] - zp) * scale;
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

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto krnl = xrt::kernel(device, uuid, "parameter_encryption");

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_plaintext = xrt::bo(device, sizeof(float)*POLYNOMIAL_DEGREE, krnl.group_id(0));
    auto bo_errors = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE*3, krnl.group_id(3));
    auto bo_public_key0 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl.group_id(4));
    auto bo_public_key1 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl.group_id(5));
    auto bo_ciphertext0 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl.group_id(6));
    auto bo_ciphertext1 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl.group_id(7));

    std::cout << "Create maps\n";
    // Map the contents of the buffer object into host memory
    auto bo_plaintext_map = bo_plaintext.map<float *>();
    auto bo_errors_map = bo_errors.map<data_t *>();
    auto bo_public_key0_map = bo_public_key0.map<data_t *>();
    auto bo_public_key1_map = bo_public_key1.map<data_t *>();
    auto bo_ciphertext0_map = bo_ciphertext0.map<data_t *>();
    auto bo_ciphertext1_map = bo_ciphertext1.map<data_t *>();

    std::fill(bo_plaintext_map, bo_plaintext_map + POLYNOMIAL_DEGREE, 0);
    std::fill(bo_errors_map, bo_errors_map + POLYNOMIAL_DEGREE*3, 0);
    std::fill(bo_public_key0_map, bo_public_key0_map + POLYNOMIAL_DEGREE, 0);
    std::fill(bo_public_key1_map, bo_public_key1_map + POLYNOMIAL_DEGREE, 0);
    std::fill(bo_ciphertext0_map, bo_ciphertext0_map + POLYNOMIAL_DEGREE, 0);
    std::fill(bo_ciphertext1_map, bo_ciphertext1_map + POLYNOMIAL_DEGREE, 0);

    // Initialize plaintext
    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        bo_plaintext_map[i] = CONV1_WEIGHT_FP32_DATA[i];
    }

    // Sample error
    std::uniform_int_distribution<int> dist1(-1, 1);
    for(int i = 0; i < POLYNOMIAL_DEGREE * 3; i++) {
        bo_errors_map[i] = dist1(rng);
    }

    // Initialize public keys
    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        bo_public_key0_map[i] = PUBLIC_KEY0[i];
        bo_public_key1_map[i] = PUBLIC_KEY1[i];
    }

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";

    float scale = CONV1_ACT_SCALE_DATA[0];
    float zp = CONV1_ACT_ZP_DATA[0];

    std::cout << "Execution of the kernel\n";
    auto hw_start = std::chrono::high_resolution_clock::now();

    bo_plaintext.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_errors.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_public_key0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_public_key1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = krnl(bo_plaintext, scale, zp, bo_errors, bo_public_key0, bo_public_key1, bo_ciphertext0, bo_ciphertext1);
    run.wait();

    // Get the output data from the device
    bo_ciphertext0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_ciphertext1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto hw_end = std::chrono::high_resolution_clock::now();
    auto hw_duration = std::chrono::duration<double, std::milli>(hw_end - hw_start).count();
    std::cout << "Hardware kernel execution time: " << hw_duration << " ms" << std::endl;

    data_t encrypted_weights_ref0[POLYNOMIAL_DEGREE];
    data_t encrypted_weights_ref1[POLYNOMIAL_DEGREE];

    auto cpu_start = std::chrono::high_resolution_clock::now();
    parameter_encryption_reference(
        bo_plaintext_map,
        scale,
        zp,
        bo_errors_map,
        bo_public_key0_map,
        bo_public_key1_map,
        encrypted_weights_ref0,
        encrypted_weights_ref1
    );

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU reference execution time: " << cpu_duration << " ms" << std::endl;

    data_t cpu_source_out[DATA_SIZE];

    int num_err = 0;
    std::cout << "Enc0 test" << std::endl;
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++)
    {
        if(bo_ciphertext0_map[i] != encrypted_weights_ref0[i]) {
            std::cout << "Error at index" << i << std::endl;
            std::cout << "Device result = (" << bo_ciphertext0_map[i] << ", " << bo_ciphertext1_map[i] << ")" <<std::endl;
            std::cout << "Ref result = (" << encrypted_weights_ref0[i] << ", " << encrypted_weights_ref1[i] << ")" <<std::endl;
            num_err++;
        }
    }
    std::cout << "Enc1 test" << std::endl;
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++)
    {
        if(bo_ciphertext1_map[i] != encrypted_weights_ref1[i]) {
            std::cout << "Error at index" << i << std::endl;
            std::cout << "i = " << i << " Device result = " << bo_ciphertext1_map[i] << std::endl;
            std::cout << "i = " << i << " Ref result = (" << encrypted_weights_ref1[i] << std::endl;
            num_err++;
        }
    }
    std::cout << "Test ended with " << num_err << " errors" << std::endl;
    return 0;
}
