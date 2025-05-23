#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "hls_math.h"
#include "BGV/encryption.hpp"
#include "BGV/parameter_decryption.hpp"

#include "keys.h"
#include "weights_bias.h"
#include "weights_bias_float.h"
#include "encrypted_weights_bias.h"
#include "constants.hpp"

int n = POLYNOMIAL_DEGREE;
int p = PLAINTEXT_MODULUS;
int q = CIPHERTEXT_MODULUS;

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
    std::cout << name << " (first 128 elements): ";
    for(int i=0; i < std::min(128, size); i++) {
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

// // Reference implementation for verification
// void ntt_reference(data_t* input, data_t* output) {
//     int n = POLYNOMIAL_DEGREE;
//     int q = CIPHERTEXT_MODULUS;
//     int w = PRIMITIVE_N_TH_ROOT_OF_UNITY;
//     // Make a copy of input
//     data_t temp[POLYNOMIAL_DEGREE];
//     for (int i = 0; i < n; i++) {
//         temp[i] = input[i];
//     }

//     // Perform NTT using the same algorithm
//     for (int m = n / 2; m >= 1; m /= 2) {
//         for (int j = 0; j < m; j++) {
//             int exp = (j * n) / (2 * m);
//             data_t w_m = 1;
//             for (int k = 0; k < exp; k++) {
//                 w_m = (w_m * w) % q;
//             }
//             printf("test w_m = %d\n", w_m);

//             for (int i = j; i < n; i += 2 * m) {
//                 data_t t1 = temp[i];
//                 data_t t2 = temp[i + m];
//                 temp[i] = (t1 + t2) % q;
//                 temp[i + m] = (w_m * ((t1 - t2 + q) % q)) % q;
//             }
//         }
//     }
    
//     // Apply bit reversal
//     for (int i = 0; i < n; i++) {
//         output[BIT_REVERSE_LUT[i]] = temp[i];
//     }
// }

// // Reference implementation for INNT verification
// void intt_reference(data_t* input, data_t* output) {
//     int n = POLYNOMIAL_DEGREE;
//     int q = CIPHERTEXT_MODULUS;
//     int w_inv = INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY;
//     int n_inv = INVERSE_POLYNOMIAL_DEGREE;
//     // Make a copy of input with bit-reversal
//     data_t temp[n];
//     for (int i = 0; i < n; i++) {
//         temp[i] = input[i];
//     }

//     // INTT algorithm
//     for (int m = n / 2; m >= 1; m /= 2) {
//         for (int j = 0; j < m; j++) {
//             int exp = (j * n) / (2 * m);
//             data_t w_m = 1;
//             for (int k = 0; k < exp; k++) {
//                 w_m = (w_m * w_inv) % q;
//             }
            
//             for (int i = j; i < n; i += 2 * m) {
//                 data_t t1 = temp[i];
//                 data_t t2 = temp[i + m];
//                 temp[i] = (t1 + t2) % q;
                
//                 // Correct computation for INTT
//                 data_t diff = (t1 - t2 + q) % q;  // Ensure positive result
//                 temp[i + m] = (diff * w_m) % q;
//             }
//         }
//     }
    
//     // Apply scaling by n_inv
//     for (int i = 0; i < n; i++) {
//         output[BIT_REVERSE_LUT[i]] = (temp[i] * n_inv) % q;
//     }
// }

// void polynomial_multiplication_reference(data_t* input1, data_t* input2, data_t* output) {
//     data_t n = POLYNOMIAL_DEGREE;
//     data_t n_inv = INVERSE_POLYNOMIAL_DEGREE;
//     data_t q = CIPHERTEXT_MODULUS;
//     data_t w = PRIMITIVE_N_TH_ROOT_OF_UNITY;
//     data_t w_inv = INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY;

//     data_t in1_tilde[POLYNOMIAL_DEGREE];
//     data_t in2_tilde[POLYNOMIAL_DEGREE];
//     for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
//         in1_tilde[i] = (E_POWERS_LUT[i] * input1[i]) % q;
//         in2_tilde[i] = (E_POWERS_LUT[i] * input2[i]) % q;
//     }

//     data_t transformed_in1_tilde[POLYNOMIAL_DEGREE];
//     data_t transformed_in2_tilde[POLYNOMIAL_DEGREE];
//     ntt_reference(in1_tilde, transformed_in1_tilde);
//     ntt_reference(in2_tilde, transformed_in2_tilde);

//     data_t transformed_out_tilde[POLYNOMIAL_DEGREE];
//     for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
//         transformed_out_tilde[i] = (transformed_in1_tilde[i] * transformed_in2_tilde[i]) % q;
//     }

//     data_t out_tilde[POLYNOMIAL_DEGREE];
//     intt_reference(transformed_out_tilde, out_tilde);

//     for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
//         output[i] = (E_INV_POWERS_LUT[i] * out_tilde[i]) % q;
//     }
// }

void polynomial_multiplication_reference(data_t* input1, data_t* input2, data_t* output) {
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

void encryption_reference(data_t* error1, data_t* error2, data_t* r, data_t* public_key1, data_t* public_key2, data_t* plaintext, data_t* ciphertext1, data_t* ciphertext2) {
    data_t temp1[n];
    data_t temp2[n];

    polynomial_multiplication_reference(public_key1, r, temp1);
    polynomial_multiplication_reference(public_key2, r, temp2);

    for(int i = 0; i < n; i++) {
        ciphertext1[i] = (plaintext[i] + p * error1[i] + temp1[i] + q) % q;
        ciphertext2[i] = (p * error2[i] + temp2[i] + q) % q;
    }
}

void decryption_reference(data_t* private_key, data_t* ciphertext1, data_t* ciphertext2, data_t* plaintext){
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
        quantized = (quantized > data_ap_fixed_t(127)) ? data_ap_fixed_t(127) : 
        ((quantized < data_ap_fixed_t(-128)) ? data_ap_fixed_t(-128) : quantized);
        quantized_pt[i] = (data_t) quantized;
    }

    data_t error0[POLYNOMIAL_DEGREE];
    data_t error1[POLYNOMIAL_DEGREE];
    data_t r[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        error0[i] = errors[i];
    }

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        error1[i] = errors[POLYNOMIAL_DEGREE + i];
    }

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        r[i] = errors[2*POLYNOMIAL_DEGREE + i];
    }

    encryption_reference(
        error0,
        error1,
        r,
        pk0,
        pk1,
        quantized_pt, 
        ct0,
        ct1
    );
}

void parameter_decryption_reference(
    data_t sk[POLYNOMIAL_DEGREE],
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

int main() {
    std::cout << "Testing parameter processing with parameters:" << std::endl;
    std::cout << "POLYNOMIAL_DEGREE: " << POLYNOMIAL_DEGREE << std::endl;
    std::cout << "CIPHERTEXT_MODULUS: " << CIPHERTEXT_MODULUS << std::endl;
    std::cout << "PRIMITIVE_N_TH_ROOT_OF_UNITY: " << PRIMITIVE_N_TH_ROOT_OF_UNITY << std::endl;

    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Decryption test" << std::endl;

    // Global flag for overall test success
    bool all_tests_passed = true;

    data_t pk0[POLYNOMIAL_DEGREE];
    data_t pk1[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        pk0[i] = PUBLIC_KEY0[i];
        pk1[i] = PUBLIC_KEY1[i];
    }

    // CONV1
    int num_conv1_weights_slices = NUM_ENCRYPTED_CONV1_WEIGHTS/POLYNOMIAL_DEGREE;
    data_t conv1_encrypted_weights0[num_conv1_weights_slices][POLYNOMIAL_DEGREE];
    data_t conv1_encrypted_weights1[num_conv1_weights_slices][POLYNOMIAL_DEGREE];

    for(int i = 0; i < num_conv1_weights_slices; i++) {
        for(int j = 0; j < POLYNOMIAL_DEGREE; j++) {
            int idx = i * POLYNOMIAL_DEGREE + j;
            conv1_encrypted_weights0[i][j] = CONV1_WEIGHT_INT8_DATA_ENC0[idx];
            conv1_encrypted_weights1[i][j] = CONV1_WEIGHT_INT8_DATA_ENC1[idx];
        }
    }

    data_ap_fixed_t scale = CONV1_ACT_SCALE_DATA[0];
    data_ap_fixed_t zp = CONV1_ACT_ZP_DATA[0];

    // Decryption Test
    data_t sk[POLYNOMIAL_DEGREE];
    for(int i = 0;i < POLYNOMIAL_DEGREE; i++) {
        sk[i] = PRIVATE_KEY[i];
    }

    data_ap_fixed_t decrypted_conv1_weights[num_conv1_weights_slices][POLYNOMIAL_DEGREE];

    for(int i = 0; i < num_conv1_weights_slices; i++) {
        parameter_decryption(
            sk,
            conv1_encrypted_weights0[i],
            conv1_encrypted_weights1[i],
            scale,
            zp,

            decrypted_conv1_weights[i]
        );
    }

    data_ap_fixed_t decrypted_conv1_weights_ref[num_conv1_weights_slices][POLYNOMIAL_DEGREE];

    for(int i = 0; i < num_conv1_weights_slices; i++) {
        parameter_decryption_reference(
            sk,
            conv1_encrypted_weights0[i],
            conv1_encrypted_weights1[i],
            scale,
            zp,

            decrypted_conv1_weights_ref[i]
        );
    }

    // Test 1: Compare reference and HLS decryption results
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Decryption test results" << std::endl;

    // Compare HLS and reference decryption results
    int decryption_result_errors = 0;
    data_ap_fixed_t max_result_diff = data_ap_fixed_t(0.0);
    data_ap_fixed_t epsilon = data_ap_fixed_t(0.00001);

    for(int i = 0; i < num_conv1_weights_slices; i++) {
        for(int j = 0; j < POLYNOMIAL_DEGREE; j++) {
            data_ap_fixed_t diff = hls::fabs(decrypted_conv1_weights[i][j] - decrypted_conv1_weights_ref[i][j]);
            
            if(diff > max_result_diff) {
                max_result_diff = diff;
            }
            
            // Consider any difference as an error
            if(diff > epsilon) {  // Using a small epsilon for floating point comparison
                decryption_result_errors++;
                if (decryption_result_errors < 10) {
                    std::cout << "Decryption result error at index " << i*POLYNOMIAL_DEGREE+j 
                            << ": Expected " << decrypted_conv1_weights_ref[i][j]
                            << ", Got " << decrypted_conv1_weights[i][j]
                            << ", Diff = " << diff << std::endl;
                }
            }
        }
    }

    if (decryption_result_errors == 0) {
        std::cout << "Decryption HLS vs reference test passed" << std::endl;
    } else {
        std::cout << "Decryption HLS vs reference test failed with " << decryption_result_errors 
                << " errors, max diff = " << max_result_diff << std::endl;
        all_tests_passed = false;  // Update overall test status
    }

    if(all_tests_passed) {
        std::cout << "All tests passed" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed" << std::endl;
        return 1;
    }
    // return all_tests_passed ? 0 : 1;
}