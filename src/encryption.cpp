#include <stdio.h>

#include "hls_stream.h"
#include "constants.hpp"
#include "modulo_reduction.hpp"
#include "polynomial_multiplication.hpp"
#include "encryption.hpp"

extern "C" {
void encryption(data_t* e1, data_t* e2, data_t* r, data_t* pk1, data_t* pk2, data_t* pt, data_t* ct1, data_t* ct2) {
    #pragma HLS INTERFACE m_axi port=e1 bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=e2 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=r bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pk1 bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pk2 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pt bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct1 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct2 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    // #pragma HLS INTERFACE s_axilite port=size bundle=control
    // #pragma HLS INTERFACE s_axilite port=return bundle=control

    int n = POLYNOMIAL_DEGREE;
    int q = CIPHERTEXT_MODULUS;
    int p = PLAINTEXT_MODULUS;
    
    data_t temp1[n];
    data_t temp2[n];

    polynomial_multiplication(pk1, r, temp1);
    polynomial_multiplication(pk2, r, temp2);

    for(int i = 0; i < n; i++) {
        // ct1[i] = hls::remainder(pt[i] + p * e1[i] + temp1[i], q);
        // ct2[i] = hls::remainder(p * e2[i] + temp2[i], q);
        ct1[i] = modulo_reduction(pt[i] + p * e1[i] + temp1[i], q);
        ct2[i] = modulo_reduction(p * e2[i] + temp2[i], q);
    }
}

void decryption(data_t* sk, data_t* ct1, data_t* ct2, data_t* pt) {
    #pragma HLS INTERFACE m_axi port=sk bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct1 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct2 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pt bundle=gmem3 depth=POLYNOMIAL_DEGREE
    // #pragma HLS INTERFACE s_axilite port=size bundle=control
    // #pragma HLS INTERFACE s_axilite port=return bundle=control

    int n = POLYNOMIAL_DEGREE;
    int q = CIPHERTEXT_MODULUS;
    int p = PLAINTEXT_MODULUS;
    data_t half_p = PLAINTEXT_MODULUS/2;

    data_t temp[n];

    polynomial_multiplication(ct2, sk, temp);

    for(int i = 0; i < n; i++) {
        data_t intermittent = hls::remainder(ct1[i] + temp[i], q);
        // pt[i] = hls::remainder(intermittent, p);
        intermittent = hls::remainder(intermittent, p);
        if(intermittent == half_p) {
            pt[i] = -half_p;
        } else {
            pt[i] = intermittent;
        }
    }
}
}