#include <iostream>
#include <iomanip>
#include "hls_math.h"
#include "ap_int.h"
#include "ntt_transform.hpp"
#include "constants.hpp"

// Reference implementation for verification
void ntt_reference(data_t* input, data_t* output, int n, int q, int w) {
    /* Different Implementation in the paper Towards Efficient Polynomial Multiplication for
Lattice-Based Cryptography */
    // // Make a copy of input
    // data_t temp[n];
    // for (int i = 0; i < n; i++) {
    //     temp[BIT_REVERSE_LUT[i]] = input[i];
    // }

    // int lgn = hls::log2(n);

    // for (int s = 1; s <= lgn; s++) {
    //     int m = hls::pow(2, s);
    //     int exp = n / m;
    //     data_t w_m = 1;
    //     for (int i = 0; i < exp; i++) {
    //         w_m = (w_m * w) % q;
    //     }
    //     printf("test w_m = %d\n", w_m);

    //     for (int k = 0; k < n; k += m) {  // Corrected to iterate in steps of m
    //         data_t omega = 1;
    //         for (int j = 0; j < m/2; j++) {
    //             data_t t = (omega * temp[k + j + m/2]) % q;
    //             data_t u = temp[k + j];
    //             temp[k + j] = (u + t) % q;
    //             // Handle potential negative values
    //             temp[k + j + m/2] = ((u - t) % q + q) % q;  // Ensure positive result
    //             omega = (omega * w_m) % q;
    //         }
    //     }
    // }
    
    // // Apply bit reversal
    // for (int i = 0; i < n; i++) {
    //     output[i] = temp[i];
    // }

    /* Same Implementation */
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
            printf("test w_m = %d\n", w_m);
            
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
void intt_reference(data_t* input, data_t* output, int n, int q, int w_inv, int n_inv) {
    /* Corrected implementation */
    
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

// Helper function to print array
void print_array(const char* name, data_t* arr, int size) {
    std::cout << name << ": [";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
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
}

// Test function
int main() {
    // Initialize test data
    data_t input[POLYNOMIAL_DEGREE];
    data_t output_hls[POLYNOMIAL_DEGREE];
    data_t output_ref[POLYNOMIAL_DEGREE];
    data_t output_innt_hls[POLYNOMIAL_DEGREE];
    data_t output_innt_ref[POLYNOMIAL_DEGREE];
    
    // Fill input with sample values
    for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        input[i] = i % CIPHERTEXT_MODULUS; // Simple pattern
    }
    
    std::cout << "Testing NTT transform with parameters:" << std::endl;
    std::cout << "POLYNOMIAL_DEGREE: " << POLYNOMIAL_DEGREE << std::endl;
    std::cout << "CIPHERTEXT_MODULUS: " << CIPHERTEXT_MODULUS << std::endl;
    std::cout << "PRIMITIVE_N_TH_ROOT_OF_UNITY: " << PRIMITIVE_N_TH_ROOT_OF_UNITY << std::endl;
    
    // Print input data
    print_array("Input", input, POLYNOMIAL_DEGREE);
    
    // Call HLS implementation
    ntt_transform(input, output_hls);
    
    // Call reference implementation for verification
    ntt_reference(input, output_ref, POLYNOMIAL_DEGREE, CIPHERTEXT_MODULUS, PRIMITIVE_N_TH_ROOT_OF_UNITY);
    
    // Call INNT
    intt_transform(output_hls, output_innt_hls);

    // Call reference INNT implementation for verification
    intt_reference(output_ref, output_innt_ref, POLYNOMIAL_DEGREE, CIPHERTEXT_MODULUS, INVERSE_PRIMITIVE_N_TH_ROOT_OF_UNITY, INVERSE_POLYNOMIAL_DEGREE);
    
    printBitReverseLUT();

    // Print both outputs
    print_array("HLS Output", output_hls, POLYNOMIAL_DEGREE);
    print_array("Reference Output", output_ref, POLYNOMIAL_DEGREE);

    // Print INNT outputs
    print_array("HLS INNT Output", output_innt_hls, POLYNOMIAL_DEGREE);
    print_array("Reference INNT Output", output_innt_ref, POLYNOMIAL_DEGREE);
    
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