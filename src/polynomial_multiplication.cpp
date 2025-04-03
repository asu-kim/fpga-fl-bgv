#include "constants.hpp"
#include "hls_math.h"
#include "hls_stream.h"
#include "ntt_transform.hpp"
#include "polynomial_multiplication.hpp"

void polynomial_multiplication(data_t* in1, data_t* in2, data_t* out) {
    #pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=out bundle=gmem2 depth=POLYNOMIAL_DEGREE

    data_t n = POLYNOMIAL_DEGREE;
    data_t q = CIPHERTEXT_MODULUS;
    data_t w = PRIMITIVE_N_TH_ROOT_OF_UNITY;

    data_t in1_tilde[POLYNOMIAL_DEGREE];
    data_t in2_tilde[POLYNOMIAL_DEGREE];
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        in1_tilde[i] = hls::remainder((data_t) (E_POWERS_LUT[i] * in1[i]), (data_t) q);
        in2_tilde[i] = hls::remainder((data_t) (E_POWERS_LUT[i] * in2[i]), (data_t) q);
    }
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        printf("in1_tilde[%d] = %d\n", i, in1_tilde[i]);
    }

    data_t transformed_in1_tilde[POLYNOMIAL_DEGREE];
    data_t transformed_in2_tilde[POLYNOMIAL_DEGREE];
    ntt_transform(in1_tilde, transformed_in1_tilde);
    ntt_transform(in2_tilde, transformed_in2_tilde);
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        printf("transformed_in1_tilde[%d] = %d\n", i, transformed_in1_tilde[i]);
    }

    data_t transformed_out_tilde[POLYNOMIAL_DEGREE];
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        transformed_out_tilde[i] = hls::remainder((data_t) (transformed_in1_tilde[i] * transformed_in2_tilde[i]), (data_t) q);
    }
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        printf("transformed_out_tilde[%d] = %d\n", i, transformed_out_tilde[i]);
    }

    data_t out_tilde[POLYNOMIAL_DEGREE];
    intt_transform(transformed_out_tilde, out_tilde);
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        printf("out_tilde[%d] = %d\n", i, out_tilde[i]);
    }

    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        out[i] = hls::remainder((data_t) (E_INV_POWERS_LUT[i] * out_tilde[i]), (data_t) q);
    }
}