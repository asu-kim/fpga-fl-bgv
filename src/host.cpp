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

#define IN_ROWS 28
#define IN_COLS 28
#define KERNEL_SIZE 5
#define IN_C 1
#define OUT_C 6
void conv1_golden(
    const data_t in_flatten[IN_C*IN_ROWS*IN_COLS],
    data_t out_data[OUT_C * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)],
    const data_t weights_flatten[256],
    const data_t bias_flatten[128],
    data_ap_fixed_t act_out_scale = 1.0f, 
    int act_out_zp = 0
) {
    data_t in_data[IN_C][IN_ROWS][IN_COLS];
    data_t weights[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE];
    data_t bias[OUT_C];
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
                    weights[i][j][k][l] = weights_flatten[i * IN_C*KERNEL_SIZE*KERNEL_SIZE + j * 25 * KERNEL_SIZE * KERNEL_SIZE + k * KERNEL_SIZE + l];
                }
            }
        }
    }

    // std::cout << "weights_golden = [";
    // for(int i=0; i<6; i++) {
    //     for(int j=0; j<1; j++) {
    //         for(int k=0; k<5; k++) {
    //             for(int l=0; l<5; l++) {
    //                 std::cout << weights[i][j][k][l] << ", ";
    //             }
    //         }
    //     }
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "bias_golden = [";
    // for(int i=0; i<6; i++) {
    //     std::cout << bias[i] << ", ";
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "input_golden = [";
    // for(int i = 0; i < IN_C; i++) {
    //     for(int j = 0; j < IN_ROWS; j++) {
    //         for(int k = 0; k < IN_COLS; k++) {
    //             std::cout << in_data[i][j][k] << ", ";
    //         }
    //     }
    // }
    // std::cout << "]" << std::endl;

    // Loop over each output channel
    for (int oc = 0; oc < OUT_C; oc++) {
        // Loop over each output row
        for (int oh = 0; oh < IN_ROWS - KERNEL_SIZE + 1; oh++) {
            // Loop over each output column
            for (int ow = 0; ow < IN_COLS - KERNEL_SIZE + 1; ow++) {
                int idx = oc * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)
                            + oh * (IN_COLS - KERNEL_SIZE + 1)
                            + ow;
                // printf("--------------------------------\n");
                // printf("output index = %d\n", idx);
                // Initialize accumulator with bias
                ap_int<128> acc = bias[oc];
                
                // Calculate convolution for current output position
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            // Input position
                            int ih = oh + kh;
                            int iw = ow + kw;
                            
                            // Accumulate weighted input
                            data_t in_val = in_data[ic][ih][iw];
                            data_t w_val = weights[oc][ic][kh][kw];
                            acc += (ap_int<64>)in_val * (ap_int<64>)w_val;
                            // printf("in_data[%d][%d][%d] = %d\n", ic, ih, iw, in_val);
                            // printf("w_data[%d][%d][%d][%d] = %d\n", oc, ic, kh, kw, w_val);
                        }
                    }
                }

                // Quantize output
                data_ap_fixed_t acc_float = data_ap_fixed_t(acc);
                data_ap_fixed_t scaled = acc_float * act_out_scale + (data_ap_fixed_t)act_out_zp;
                data_ap_fixed_t rounded = floor(scaled + 0.5f);
                
                // Clip to data type range
                data_t result = (data_t)rounded;
                result = hls::max(hls::numeric_limits<data_t>::min(), 
                            hls::min(hls::numeric_limits<data_t>::max(), result));
                
                // Calculate output index and store result
                int out_idx = oc * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)
                            + oh * (IN_COLS - KERNEL_SIZE + 1)
                            + ow;
                out_data[out_idx] = result;
                // printf("out[%d] = %d\n", out_idx, result);
            }
        }
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
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (binaryFile.empty()) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    // Load ciphertext data for weights and bias
    std::vector<data_t> weight_ciphertext1_1(CONV1_WEIGHT_INT8_DATA0_ENC1, CONV1_WEIGHT_INT8_DATA0_ENC1 + DATA_SIZE);
    std::vector<data_t> weight_ciphertext1_2(CONV1_WEIGHT_INT8_DATA0_ENC2, CONV1_WEIGHT_INT8_DATA0_ENC2 + DATA_SIZE);
    std::vector<data_t> weight_ciphertext2_1(CONV1_WEIGHT_INT8_DATA1_ENC1, CONV1_WEIGHT_INT8_DATA1_ENC1 + DATA_SIZE);
    std::vector<data_t> weight_ciphertext2_2(CONV1_WEIGHT_INT8_DATA1_ENC2, CONV1_WEIGHT_INT8_DATA1_ENC2 + DATA_SIZE);
    std::vector<data_t> bias_ciphertext1(CONV1_BIAS_INT8_DATA_ENC1, CONV1_BIAS_INT8_DATA_ENC1 + DATA_SIZE);
    std::vector<data_t> bias_ciphertext2(CONV1_BIAS_INT8_DATA_ENC2, CONV1_BIAS_INT8_DATA_ENC2 + DATA_SIZE);
    std::vector<data_t> private_key(PRIVATE_KEY, PRIVATE_KEY + DATA_SIZE);

    std::vector<data_t> in_data(784);
    std::vector<data_t> out_data(3456);

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    size_t vector_size_bytes = sizeof(data_t) * DATA_SIZE;
    size_t in_size_bytes = sizeof(data_t) * 784;
    size_t out_size_bytes = sizeof(data_t) * 3456;

    // Create kernels
    auto decrypt_krnl = xrt::kernel(device, uuid, "decryption");
    auto conv1_krnl = xrt::kernel(device, uuid, "conv1");

    std::cout << "Allocate Buffer in Global Memory\n";
    // Allocate buffer objects for decryption
    auto bo_private_key = xrt::bo(device, vector_size_bytes, decrypt_krnl.group_id(0));
    auto bo_ciphertext1 = xrt::bo(device, vector_size_bytes, decrypt_krnl.group_id(1));
    auto bo_ciphertext2 = xrt::bo(device, vector_size_bytes, decrypt_krnl.group_id(2));
    auto bo_decrypted_out = xrt::bo(device, vector_size_bytes, decrypt_krnl.group_id(3));

    // Allocate in and out for conv1
    auto bo_in_data = xrt::bo(device, in_size_bytes, conv1_krnl.group_id(0));
    auto bo_out_data = xrt::bo(device, out_size_bytes, conv1_krnl.group_id(1));

    // Allocate HBM buffers for conv1
    auto bo_weights = xrt::bo(device, sizeof(data_t) * 256, conv1_krnl.group_id(2));
    auto bo_bias = xrt::bo(device, sizeof(data_t) * 128, conv1_krnl.group_id(3));

    // Map buffers to host memory
    auto bo_private_key_map = bo_private_key.map<data_t *>();
    auto bo_ciphertext1_map = bo_ciphertext1.map<data_t *>();
    auto bo_ciphertext2_map = bo_ciphertext2.map<data_t *>();
    auto bo_decrypted_out_map = bo_decrypted_out.map<data_t *>();
    auto bo_weights_map = bo_weights.map<data_t *>();
    auto bo_bias_map = bo_bias.map<data_t *>();

    auto bo_in_data_map = bo_in_data.map<data_t *>();
    auto bo_out_data_map = bo_out_data.map<data_t *>();

    std::cout << "Initialize buffers\n";
    // Initialize buffers
    std::fill(bo_private_key_map, bo_private_key_map + DATA_SIZE, 0);
    std::fill(bo_weights_map, bo_weights_map + 256, 0);
    std::fill(bo_bias_map, bo_bias_map + 128, 0);


    std::fill(bo_in_data_map, bo_in_data_map + DATA_SIZE, 0);
    std::fill(bo_out_data_map, bo_out_data_map + DATA_SIZE, 0);

    // Write input data (all 1s)
    for(int i = 0; i < IN_ROWS; i++) {
        for(int j = 0; j < IN_COLS; j++) {
            in_data[i * IN_COLS + j] = 1;
        }
    }

    // Copy private key to buffer
    for (int i = 0; i < DATA_SIZE; i++) {
        bo_private_key_map[i] = private_key[i];
    }
    bo_private_key.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Step 1: Decrypt first part of weights
    std::cout << "Decrypting weights part 1\n";
    for(int i = 0; i < DATA_SIZE; i++) {
        bo_ciphertext1_map[i] = weight_ciphertext1_1[i];
        bo_ciphertext2_map[i] = weight_ciphertext1_2[i];
    }
    bo_ciphertext1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ciphertext2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run1 = decrypt_krnl(bo_private_key, bo_ciphertext1, bo_ciphertext2, bo_decrypted_out);
    run1.wait();
    
    bo_decrypted_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < DATA_SIZE; i++) {
        if (i < 128) {
            bo_weights_map[i] = bo_decrypted_out_map[i];
        }
    }

    std::cout << "decrypted_weight1 = [";
    for(int i = 0; i < 128; i++) {
        std::cout << bo_decrypted_out_map[i] << ", ";
    }
    std::cout << "]\n";

    // Step 2: Decrypt second part of weights
    std::cout << "Decrypting weights part 2\n";
    for (int i = 0; i < DATA_SIZE; i++) {
        bo_ciphertext1_map[i] = weight_ciphertext2_1[i];
        bo_ciphertext2_map[i] = weight_ciphertext2_2[i];
    }
    bo_ciphertext1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ciphertext2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run2 = decrypt_krnl(bo_private_key, bo_ciphertext1, bo_ciphertext2, bo_decrypted_out);
    run2.wait();
    
    bo_decrypted_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < DATA_SIZE; i++) {
        if (i < 128) {
            bo_weights_map[i + 128] = bo_decrypted_out_map[i];
        }
    }

    std::cout << "decrypted_weight2 = [";
    for(int i = 0; i < 128; i++) {
        std::cout << bo_decrypted_out_map[i] << ", ";
    }
    std::cout << "]\n";

    // Step 3: Decrypt bias
    std::cout << "Decrypting bias\n";
    for (int i = 0; i < DATA_SIZE; i++) {
        bo_ciphertext1_map[i] = bias_ciphertext1[i];
        bo_ciphertext2_map[i] = bias_ciphertext2[i];
    }
    bo_ciphertext1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ciphertext2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run3 = decrypt_krnl(bo_private_key, bo_ciphertext1, bo_ciphertext2, bo_decrypted_out);
    run3.wait();
    
    bo_decrypted_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < DATA_SIZE; i++) {
        if (i < 128) {
            bo_bias_map[i] = bo_decrypted_out_map[i];
        }
    }

    // Sync weights and bias to device
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_bias.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "weight = [";
    for(int i = 0; i < 256; i++) {
        std::cout << bo_weights_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "bias = [";
    for(int i = 0; i < 128; i++) {
        std::cout << bo_bias_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    // Step 4: Sync inputs
    for(int i = 0; i < IN_ROWS; i++) {
        for(int j = 0; j < IN_COLS; j++) {
            bo_in_data_map[i * IN_COLS + j] = in_data[i * IN_COLS + j];
        }
    }
    bo_in_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "input = [";
    for(int i = 0; i < 784; i++) {
        std::cout << bo_in_data_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    // Step 5: Run convolution
    std::cout << "Running convolution\n";
    auto run5 = conv1_krnl(bo_in_data, bo_out_data, bo_weights, bo_bias);
    auto state = run5.wait(std::chrono::seconds(5)); // Add timeout
    if (state != ERT_CMD_STATE_COMPLETED) {
        std::cout << "Kernel execution timed out or failed" << std::endl;
        // Handle error
    }
    // run5.wait();
    std::cout << "Done convolution\n";

    // Step 6: Read output from stream
    bo_out_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    data_t bo_out_data_golden[3456];
    conv1_golden(bo_in_data_map, bo_out_data_golden, bo_weights_map, bo_bias_map);

    // Print results
    std::cout << "Convolution results:\n";
    std::cout << "out_data = [";
    for(int i=0; i<3456; i++) {
        std::cout << bo_out_data_map[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "out_data_golden = [";
    for(int i=0; i<3456; ++i) {
        std::cout << bo_out_data_golden[i] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Test completed\n";
    return 0;
}
