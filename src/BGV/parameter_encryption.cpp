#include "BGV/encryption.hpp"
#include "BGV/parameter_encryption.hpp"
// #include "keys.h"
#include <iostream>

#ifndef __SYNTHESIS__
void print_data_t_array_last_hls(const data_t* arr, int size, const char* name) {
    std::cout << name << " (last 128 elements): ";
    int start_idx = size >= 128 ? size - 128 : 0;
    for(int i = 0; i < std::min(128, size); i++) {
        std::cout << arr[start_idx + i] << ", ";
    }
    std::cout << std::endl << std::endl;
}
void print_data_t_array_hls(const data_t* arr, int size, const char* name) {
    std::cout << name << " (first 10 elements): ";
    for(int i = 0; i < std::min(10, size); i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl << std::endl;
}
#else
// Empty function with C-style string
void print_data_t_array_last_hls(const data_t* arr, int size, const char* name) {}
void print_data_t_array_hls(const data_t* arr, int size, const char* name) {}
#endif

extern "C" {
void parameter_encryption(
    float pt[POLYNOMIAL_DEGREE],
    float scale,
    float zp,
    data_t errors[POLYNOMIAL_DEGREE*3],
    data_t pk0[POLYNOMIAL_DEGREE],
    data_t pk1[POLYNOMIAL_DEGREE],

    data_t ct0[POLYNOMIAL_DEGREE],
    data_t ct1[POLYNOMIAL_DEGREE]
) {
    #pragma HLS INTERFACE m_axi port=pt     bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=errors bundle=gmem1 depth=POLYNOMIAL_DEGREE*3
    #pragma HLS INTERFACE m_axi port=pk0    bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pk1    bundle=gmem3 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct0    bundle=gmem4 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct1    bundle=gmem5 depth=POLYNOMIAL_DEGREE

    // Control interface
    #pragma HLS INTERFACE s_axilite port=pt     bundle=control
    #pragma HLS INTERFACE s_axilite port=scale  bundle=control
    #pragma HLS INTERFACE s_axilite port=zp     bundle=control
    #pragma HLS INTERFACE s_axilite port=errors bundle=control
    #pragma HLS INTERFACE s_axilite port=pk0    bundle=control
    #pragma HLS INTERFACE s_axilite port=pk1    bundle=control
    #pragma HLS INTERFACE s_axilite port=ct0    bundle=control
    #pragma HLS INTERFACE s_axilite port=ct1    bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    float local_pt[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        #pragma HLS PIPELINE II=1
        local_pt[i] = pt[i];
    }

    data_t local_error0[POLYNOMIAL_DEGREE];
    data_t local_error1[POLYNOMIAL_DEGREE];
    data_t local_r[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        #pragma HLS PIPELINE II=1
        local_error0[i] = errors[i];
    }

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        #pragma HLS PIPELINE II=1
        local_error1[i] = errors[POLYNOMIAL_DEGREE + i];
    }

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        #pragma HLS PIPELINE II=1
        local_r[i] = errors[2*POLYNOMIAL_DEGREE + i];
    }

    data_t local_pk0[POLYNOMIAL_DEGREE];
    data_t local_pk1[POLYNOMIAL_DEGREE];
    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        #pragma HLS PIPELINE II=1
        local_pk0[i] = pk0[i];
        local_pk1[i] = pk1[i];
    }

    data_t quantized_pt[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        #pragma HLS PIPELINE II=1
        float quantized = local_pt[i] / scale + zp;
        quantized = (quantized > 127) ? 127 : ((quantized < -128) ? -128 : quantized);
        quantized_pt[i] = (data_t) quantized;
    }

    data_t local_ct0[POLYNOMIAL_DEGREE];
    data_t local_ct1[POLYNOMIAL_DEGREE];

    encryption(
        local_error0, 
        local_error1, 
        local_r, 
        local_pk0, 
        local_pk1, 
        quantized_pt, 
        local_ct0, 
        local_ct1
    );

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        #pragma HLS PIPELINE II=1
        ct0[i] = local_ct0[i];
        ct1[i] = local_ct1[i];
    }
}
}