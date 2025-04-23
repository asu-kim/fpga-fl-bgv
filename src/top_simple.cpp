#include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "encryption.hpp"
#include <iostream>

extern "C" {
void top(
        data_t* private_key,
        data_t* encrypted_conv1_weight0_0,
        data_t* encrypted_conv1_weight0_1,
        data_t* encrypted_conv1_weight1_0,
        data_t* encrypted_conv1_weight1_1,
        data_t* encrypted_conv1_bias0,
        data_t* encrypted_conv1_bias1,
        data_t* input,
        data_t* output
    ) {
    #pragma HLS INTERFACE m_axi port=private_key bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=encrypted_conv1_weight0_0 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=encrypted_conv1_weight0_1 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=encrypted_conv1_weight1_0 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=encrypted_conv1_weight1_1 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=encrypted_conv1_bias0 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=encrypted_conv1_bias1 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=input bundle=gmem1 depth=784
    #pragma HLS INTERFACE m_axi port=output bundle=gmem1 depth=3456

    data_t decrypted[POLYNOMIAL_DEGREE];
    data_t decrypted_conv1_weight[150];
    data_t decrypted_conv1_bias[6];

    std::cout << "Decrypted weight = [";
    decryption(private_key, encrypted_conv1_weight0_0, encrypted_conv1_weight0_1, decrypted);
    for(int i = 0; i < 128; i++) {
        decrypted_conv1_weight[i] = decrypted[i];
        std::cout << decrypted[i] << ", ";
    }
    decryption(private_key, encrypted_conv1_weight1_0, encrypted_conv1_weight1_1, decrypted);
    for(int i = 128; i < 150; i++) {
        decrypted_conv1_weight[i] = decrypted[i - 128];
        std::cout << decrypted[i - 128] << ", ";
    }
    std::cout << "]\n ";
    std::cout << "Decrypted bias = [";
    decryption(private_key, encrypted_conv1_bias0, encrypted_conv1_bias1, decrypted);
    for(int i = 0; i < 6; i++) {
        decrypted_conv1_bias[i] = decrypted[i];
        std::cout << decrypted[i] << ", ";
    }
    std::cout << "]\n ";
    data_t local_weight[6][1][5][5];
    // Copy data
    for(int i=0; i<6; i++) {
        for(int j=0; j<1; j++) {
            for(int k=0; k<5; k++) {
                for(int l=0; l<5; l++) {
                    local_weight[i][j][k][l] = decrypted_conv1_weight[i * 25 + j * 25 + k * 5 + l];
                }
            }
        }
    }

    conv2d<6, 1, 5, 28, 28>(input, output, local_weight, decrypted_conv1_bias);
}
}