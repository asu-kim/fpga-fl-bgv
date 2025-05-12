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
// #include "BGV/encryption.hpp"
// #include "BGV/parameter_processing.hpp"

#include "keys.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "encrypted_weights_bias.h"
#include "../test/test_utils.h"
#include "constants.hpp"

#include "hls_math.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define DATA_SIZE POLYNOMIAL_DEGREE

// std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
std::mt19937 rng(42);

void print_float_array(const data_ap_fixed_t* arr, int size, const std::string& name) {
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
    data_ap_fixed_t pt[POLYNOMIAL_DEGREE],
    data_ap_fixed_t scale,
    data_ap_fixed_t zp,
    data_t errors[POLYNOMIAL_DEGREE*3],
    data_t pk0[POLYNOMIAL_DEGREE],
    data_t pk1[POLYNOMIAL_DEGREE],

    data_t ct0[POLYNOMIAL_DEGREE],
    data_t ct1[POLYNOMIAL_DEGREE]
) {
    // Quantize
    data_t quantized_pt[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        data_ap_fixed_t quantized = pt[i] / scale+ zp;
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
    data_ap_fixed_t scale,
    data_ap_fixed_t zp,

    data_ap_fixed_t pt[POLYNOMIAL_DEGREE]
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

void parameter_encryption_test(const xrt::device& device, const xrt::uuid& uuid, xrt::kernel& krnl_enc) {
    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_plaintext = xrt::bo(device, sizeof(data_ap_fixed_t)*POLYNOMIAL_DEGREE, krnl_enc.group_id(0));
    auto bo_errors = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE*3, krnl_enc.group_id(3));
    auto bo_public_key0 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl_enc.group_id(4));
    auto bo_public_key1 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl_enc.group_id(5));
    auto bo_ciphertext0 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl_enc.group_id(6));
    auto bo_ciphertext1 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl_enc.group_id(7));

    std::cout << "Create maps\n";
    // Map the contents of the buffer object into host memory
    auto bo_plaintext_map = bo_plaintext.map<data_ap_fixed_t *>();
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

    data_ap_fixed_t scale = CONV1_ACT_SCALE_DATA[0];
    data_ap_fixed_t zp = CONV1_ACT_ZP_DATA[0];

    std::cout << "Execution of the kernel\n";
    auto hw_start = std::chrono::high_resolution_clock::now();

    bo_plaintext.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_errors.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_public_key0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_public_key1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = krnl_enc(bo_plaintext, scale, zp, bo_errors, bo_public_key0, bo_public_key1, bo_ciphertext0, bo_ciphertext1);
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
    std::cout << "Encryption test ended with " << num_err << " errors" << std::endl << std::endl;
}

void parameter_decryption_test(const xrt::device& device, const xrt::uuid& uuid, xrt::kernel& krnl_dec) {
    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_private_key = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl_dec.group_id(0));
    auto bo_ciphertext0 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl_dec.group_id(1));
    auto bo_ciphertext1 = xrt::bo(device, sizeof(data_t)*POLYNOMIAL_DEGREE, krnl_dec.group_id(2));
    auto bo_plaintext = xrt::bo(device, sizeof(data_ap_fixed_t)*POLYNOMIAL_DEGREE, krnl_dec.group_id(5));

    std::cout << "Create maps\n";
    // Map the contents of the buffer object into host memory
    auto bo_private_key_map = bo_private_key.map<data_t *>();
    auto bo_ciphertext0_map = bo_ciphertext0.map<data_t *>();
    auto bo_ciphertext1_map = bo_ciphertext1.map<data_t *>();
    auto bo_plaintext_map = bo_plaintext.map<data_ap_fixed_t *>();

    std::fill(bo_private_key_map, bo_private_key_map + POLYNOMIAL_DEGREE, 0);
    std::fill(bo_ciphertext0_map, bo_ciphertext0_map + POLYNOMIAL_DEGREE, 0);
    std::fill(bo_ciphertext1_map, bo_ciphertext1_map + POLYNOMIAL_DEGREE, 0);
    std::fill(bo_plaintext_map, bo_plaintext_map + POLYNOMIAL_DEGREE, 0);

    // Initialize private key
    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        bo_private_key_map[i] = PRIVATE_KEY[i];
    }

    // Initialize ciphertexts
    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        bo_ciphertext0_map[i] = CONV1_WEIGHT_INT8_DATA_ENC0[i];
        bo_ciphertext1_map[i] = CONV1_WEIGHT_INT8_DATA_ENC1[i];
    }

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";

    data_ap_fixed_t scale = CONV1_ACT_SCALE_DATA[0];
    data_ap_fixed_t zp = CONV1_ACT_ZP_DATA[0];

    std::cout << "Execution of the kernel\n";
    auto hw_start = std::chrono::high_resolution_clock::now();
    bo_private_key.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ciphertext0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ciphertext1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = krnl_dec(bo_private_key, bo_ciphertext0, bo_ciphertext1, scale, zp, bo_plaintext);
    run.wait();
    bo_plaintext.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto hw_end = std::chrono::high_resolution_clock::now();
    auto hw_duration = std::chrono::duration<double, std::milli>(hw_end - hw_start).count();
    std::cout << "Hardware kernel execution time: " << hw_duration << " ms" << std::endl;

    data_ap_fixed_t paintext_ref[POLYNOMIAL_DEGREE];

    auto cpu_start = std::chrono::high_resolution_clock::now();
    parameter_decryption_reference(
        bo_private_key_map,
        bo_ciphertext0_map,
        bo_ciphertext1_map,
        scale,
        zp,
        paintext_ref
    );

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU reference execution time: " << cpu_duration << " ms" << std::endl;

    int num_err = 0;
    data_ap_fixed_t max_err = 0;
    std::cout << "Test" << std::endl;
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++)
    {
        data_ap_fixed_t diff = std::abs(bo_plaintext_map[i] - paintext_ref[i]);
        if(diff > max_err) {
            max_err = diff;
        }
        if(diff > 0.01f) {
            std::cout << "Error at index " << i << std::endl;
            std::cout << "i = " << i << " Device result = " << bo_plaintext_map[i] << std::endl;
            std::cout << "i = " << i << " Ref result = " << paintext_ref[i] << std::endl;
            num_err++;
        }
    }
    std::cout << "Decryption test ended with " << num_err << " errors" << std::endl;
    std::cout << "Maximum absolute error: " << max_err << std::endl << std::endl;
}

void forward_path_test(const xrt::device& device, const xrt::uuid& uuid, xrt::kernel& krnl_forward) {
    size_t in_size = CONV1_IN_CH * CONV1_IN_ROWS * CONV1_IN_COLS;
    size_t weights_size = TOTAL_WEIGHTS_SIZE;
    size_t biases_size = TOTAL_BIASES_SIZE;
    size_t outputs_size = TOTAL_OUTS_SIZE;

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_in_data = xrt::bo(device, sizeof(data_ap_fixed_t)*in_size, krnl_forward.group_id(0));
    auto bo_weights = xrt::bo(device, sizeof(data_ap_fixed_t)*weights_size, krnl_forward.group_id(1));
    auto bo_biases = xrt::bo(device, sizeof(data_ap_fixed_t)*biases_size, krnl_forward.group_id(2));
    auto bo_outs = xrt::bo(device, sizeof(data_ap_fixed_t)*outputs_size, krnl_forward.group_id(3));

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

    auto run = krnl_forward(
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

    std::cout << "Forward path test completed" << std::endl << std::endl;
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

    auto krnl_enc = xrt::kernel(device, uuid, "parameter_encryption");
    auto krnl_dec = xrt::kernel(device, uuid, "parameter_decryption");

    parameter_encryption_test(device, uuid, krnl_enc);
    parameter_decryption_test(device, uuid, krnl_dec);
    forward_path_test(device, uuid, krnl_dec);

    return 0;
}
