#include "constants.hpp"
#include <iostream>
#include <iomanip>
#include "hls_math.h"
#include "BGV/polynomial_multiplication.hpp"

// Helper function to print array
void print_array(const char* name, data_t* arr, int size) {
    std::cout << name << ": [";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
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
//     print_array("in1_tilde", in1_tilde, POLYNOMIAL_DEGREE);

//     data_t transformed_in1_tilde[POLYNOMIAL_DEGREE];
//     data_t transformed_in2_tilde[POLYNOMIAL_DEGREE];
//     ntt_reference(in1_tilde, transformed_in1_tilde);
//     ntt_reference(in2_tilde, transformed_in2_tilde);
//     print_array("transformed_in1_tilde", transformed_in1_tilde, POLYNOMIAL_DEGREE);

//     data_t transformed_out_tilde[POLYNOMIAL_DEGREE];
//     for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
//         transformed_out_tilde[i] = (transformed_in1_tilde[i] * transformed_in2_tilde[i]) % q;
//     }
//     print_array("transformed_out_tilde", transformed_out_tilde, POLYNOMIAL_DEGREE);

//     data_t out_tilde[POLYNOMIAL_DEGREE];
//     intt_reference(transformed_out_tilde, out_tilde);
//     print_array("out_tilde", out_tilde, POLYNOMIAL_DEGREE);

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

// Function to print the lookup table
void printBitReverseLUT() {
    std::cout << "BIT_REVERSE_LUT contents:" << std::endl;
    
    // Print header
    std::cout << "Index | Bit-Reversed Value" << std::endl;
    std::cout << "------+------------------" << std::endl;
    
    // Print each element in the table
    for (size_t i = 0; i < POLYNOMIAL_DEGREE; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(18) << BIT_REVERSE_LUT[i] << std::endl;
    }
    
    // Print header
    std::cout << "Index | W_POWERS_LUT Value" << std::endl;
    std::cout << "------+------------------" << std::endl;
    
    // Print each element in the table
    for (size_t i = 0; i < POLYNOMIAL_DEGREE; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(18) << W_POWERS_LUT[i] << std::endl;
    }
    
    // Print header
    std::cout << "Index | W_INV_POWERS_LUT Value" << std::endl;
    std::cout << "------+------------------" << std::endl;
    
    // Print each element in the table
    for (size_t i = 0; i < POLYNOMIAL_DEGREE; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(18) << W_INV_POWERS_LUT[i] << std::endl;
    }
    
    // Print header
    std::cout << "Index | W_INV_POWERS_HALF_LUT Value" << std::endl;
    std::cout << "------+------------------" << std::endl;
    
    // Print each element in the table
    for (size_t i = 0; i < POLYNOMIAL_DEGREE; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(18) << W_INV_POWERS_HALF_LUT[i] << std::endl;
    }
    
    // Print header
    std::cout << "Index | E_POWERS_LUT Value" << std::endl;
    std::cout << "------+------------------" << std::endl;
    
    // Print each element in the table
    for (size_t i = 0; i < POLYNOMIAL_DEGREE; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(18) << E_POWERS_LUT[i] << std::endl;
    }
    
    // Print header
    std::cout << "Index | E_INV_POWERS_LUT Value" << std::endl;
    std::cout << "------+------------------" << std::endl;
    
    // Print each element in the table
    for (size_t i = 0; i < POLYNOMIAL_DEGREE; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(18) << E_INV_POWERS_LUT[i] << std::endl;
    }
}

// Test function
int main() {
    // Initialize test data
    data_t input1[POLYNOMIAL_DEGREE];
    data_t input2[POLYNOMIAL_DEGREE];
    // data_t input1[POLYNOMIAL_DEGREE] = {0, 0, 0, 0, 0, 0, 0, 1};
    // data_t input2[POLYNOMIAL_DEGREE] = {0, 0, 0, 0, 0, 0, 0, 1};
    data_t output_hls[POLYNOMIAL_DEGREE];
    data_t output_ref[POLYNOMIAL_DEGREE];

    data_t output_ntt[POLYNOMIAL_DEGREE];
    data_t output_intt[POLYNOMIAL_DEGREE];
    data_t output_mult[POLYNOMIAL_DEGREE];
    
    // Fill input with sample values
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        input1[i] = i % CIPHERTEXT_MODULUS + 1; // Simple pattern
        input2[i] = i % CIPHERTEXT_MODULUS + 1; // Simple pattern
    }
    
    std::cout << "Testing NTT transform with parameters:" << std::endl;
    std::cout << "POLYNOMIAL_DEGREE: " << POLYNOMIAL_DEGREE << std::endl;
    std::cout << "CIPHERTEXT_MODULUS: " << CIPHERTEXT_MODULUS << std::endl;
    std::cout << "PRIMITIVE_N_TH_ROOT_OF_UNITY: " << PRIMITIVE_N_TH_ROOT_OF_UNITY << std::endl;
    
    // Print input data
    print_array("Input", input1, POLYNOMIAL_DEGREE);

    // data_t temp[POLYNOMIAL_DEGREE] = {1, 1, 1, 1, 1, 1, 1, 1};
    // ntt_reference(temp, output_ntt);

    // // data_t temp2[POLYNOMIAL_DEGREE] = {8, 4, 0, 13, 2, 16, 16, 2};
    // intt_reference(output_ntt, output_intt);
    
    polynomial_multiplication(input1, input2, output_hls);

    polynomial_multiplication_reference(input1, input2, output_ref);
    
    // printBitReverseLUT();

    // Print both outputs
    // print_array("NTT Output", output_ntt, POLYNOMIAL_DEGREE);
    // print_array("INTT Output", output_intt, POLYNOMIAL_DEGREE);

    print_array("HLS Output", output_hls, POLYNOMIAL_DEGREE);
    print_array("Reference Output", output_ref, POLYNOMIAL_DEGREE);
    
    // Verify results
    bool match = true;
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        if (output_hls[i] != output_ref[i]) {
            if (hls::abs(output_hls[i]) + hls::abs(output_ref[i]) != CIPHERTEXT_MODULUS) {
                match = false;
            }
            std::cout << "Mismatch at index " << i << ": HLS=" << output_hls[i] 
                      << ", Ref=" << output_ref[i] << std::endl;
        }
    }
    
    if (match) {
        std::cout << "PASS: HLS implementation matches reference!" << std::endl;
    } else {
        std::cout << "FAIL: Results don't match!" << std::endl;
    }
    
    return match ? 0 : 1;
}