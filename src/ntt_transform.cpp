#include "hls_math.h"
#include "hls_stream.h"
#include "ntt_transform.hpp"
#include "constants.hpp"
#include <stdio.h>

// Forward NTT for polynomial multiplication
void ntt_transform(data_t* coeffs, data_t* result) {
    #pragma HLS INTERFACE m_axi port=coeffs bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=result bundle=gmem1 depth=POLYNOMIAL_DEGREE

    data_t n = POLYNOMIAL_DEGREE;
    data_t q = CIPHERTEXT_MODULUS;
    data_t w = PRIMITIVE_N_TH_ROOT_OF_UNITY;
    
    // Create local buffers for better performance
    data_t local_coeffs[POLYNOMIAL_DEGREE];
    #pragma HLS ARRAY_PARTITION variable=local_coeffs cyclic factor=16
    
    // Copy input to local memory
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        local_coeffs[i] = coeffs[i];
    }

    for (int m = POLYNOMIAL_DEGREE / 2; m >= 1; m /= 2) {
        for (int j = 0; j < m; j++) {
            data_t exponent = j * n / (2 * m);
            // data_t intermittent = hls::pow((data_t) w, ((data_t) exponent));
            // data_t w_m = hls::remainder((data_t) intermittent, (data_t) q);
            // data_t w_m = 1;
            // for (int e = 0; e < exponent; e++){
            //     w_m = hls::remainder((data_t) (w_m * w), (data_t) q);
            // }
            data_t w_m = W_POWERS_HALF_LUT[exponent];
            // printf("w_m = %d\n", w_m);
            for (int i = j; i < n; i += 2 * m) {
                data_t butterfly1 = local_coeffs[i] + local_coeffs[i + m];
                data_t butterfly2 = w_m * (local_coeffs[i] - local_coeffs[i + m]);
                local_coeffs[i] = hls::remainder((data_t) butterfly1, q);
                local_coeffs[i + m] = hls::remainder((data_t) butterfly2, q);
            }
        }
    }
    
    // Copy result back
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        result[BIT_REVERSE_LUT[i]] = local_coeffs[i];
    }
}



// Inverse NTT
void intt_transform(data_t* coeffs, data_t* result) {
    #pragma HLS INTERFACE m_axi port=coeffs bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=result bundle=gmem1 depth=POLYNOMIAL_DEGREE

    data_t n = POLYNOMIAL_DEGREE;
    data_t n_inv = INVERSE_POLYNOMIAL_DEGREE;
    data_t q = CIPHERTEXT_MODULUS;
    data_t w_inv = INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY;
    
    // Create local buffers for better performance
    data_t local_coeffs[POLYNOMIAL_DEGREE];
    #pragma HLS ARRAY_PARTITION variable=local_coeffs cyclic factor=16
    
    // Copy input to local memory with bit reversal
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        local_coeffs[i] = coeffs[i];
    }

    // INTT algorithm (note: starting with m=1 and doubling)
    for (int m = POLYNOMIAL_DEGREE / 2; m >= 1; m /= 2) {
        for (int j = 0; j < m; j++) {
            data_t exponent = j * n / (2 * m);
            // data_t w_m = 1;
            // for (int e = 0; e < exponent; e++) {
            //     w_m = hls::remainder((data_t)(w_m * w_inv), (data_t)q);
            // }
            data_t w_m_inv = W_INV_POWERS_HALF_LUT[exponent];
            
            for (int i = j; i < n; i += 2 * m) {
                // data_t u = local_coeffs[i];
                // data_t v = local_coeffs[i + m];
                // local_coeffs[i] = hls::remainder((data_t)(u + v), q);
                // // For INTT, we compute (u - v) * w_m_inv
                // // Ensure the subtraction doesn't go negative
                // // data_t diff = (u >= v) ? (u - v) : (u + q - v);
                // local_coeffs[i + m] = hls::remainder((data_t)((u - v) * w_m_inv), q);

                data_t butterfly1 = local_coeffs[i] + local_coeffs[i + m];
                data_t butterfly2 = w_m_inv * (local_coeffs[i] - local_coeffs[i + m]);
                local_coeffs[i] = hls::remainder((data_t) butterfly1, q);
                local_coeffs[i + m] = hls::remainder((data_t) butterfly2, q);
            }
        }
    }
    
    // Multiply by n^(-1) and copy result back
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        result[BIT_REVERSE_LUT[i]] = hls::remainder((data_t)(local_coeffs[i] * n_inv), q);
    }
}