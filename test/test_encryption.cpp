#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <random>
#include <cstdlib>
#include <ctime>
// #include "hls_stream.h"
#include "hls_math.h"
#include "encryption.hpp"
#include "constants.hpp"

int n = POLYNOMIAL_DEGREE;
int p = PLAINTEXT_MODULUS;
int q = CIPHERTEXT_MODULUS;

// Helper function to print array
void print_array(const char* name, data_t* arr, int size) {
    std::cout << name << ": [";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));

void polynomial_multiplication_reference(data_t* input1, data_t* input2, data_t* output) {
    // data_t temp[POLYNOMIAL_DEGREE];
    for(int i = 0; i < n; i++) {
        int pos = 0;
        for(int j = 0; j <= i; j++) {
            pos += (input1[j] * input2[i - j]) % q;
        }
        int neg = 0;
        for(int j = i + 1; j < n; j++) {
            neg += (input1[j] * input2[n + i - j]) % q;
        }
        output[i] = (((pos - neg) % q) + q) % q;
    }
}

void generate_key(data_t* private_key, data_t* public_key1, data_t* public_key2) {
    // data_t a_prime[n] = {1, 2, 3, 4};
    // data_t error[n] = {1, 0, -1, 1};
    data_t a_prime[n];
    data_t error[n];

    std::uniform_int_distribution<int> dist1(-1, 1);

    for(int i = 0; i < n; i++) {
        private_key[i] = dist1(rng);
        error[i] = dist1(rng);
    }

    std::uniform_int_distribution<int> dist2(0, q - 1);

    for(int i = 0; i < n; i++) {
        a_prime[i] = dist2(rng);
        // std::cout << "a_prime = " << a_prime[i] << std::endl;
    }

    data_t temp[n];
    polynomial_multiplication_reference(a_prime, private_key, temp);
    for(int i = 0; i < n; i++) {
        printf("temp[%d] = %d\n", i, temp[i]);
    }

    for(int i = 0; i < n; i++) {
        public_key1[i] = (temp[i] + p * error[i] + q) % q;
        public_key2[i] = -a_prime[i];
    }
}

void encryption_reference(data_t* error1, data_t* error2, data_t* r, data_t* publick_key1, data_t* publick_key2, data_t* plaintext, data_t* ciphertext1, data_t* ciphertext2) {
    data_t temp1[n];
    data_t temp2[n];

    polynomial_multiplication_reference(publick_key1, r, temp1);
    polynomial_multiplication_reference(publick_key2, r, temp2);

    for(int i = 0; i < n; i++) {
        ciphertext1[i] = (plaintext[i] + p * error1[i] + temp1[i] + q) % q;
        ciphertext2[i] = (p * error2[i] + temp2[i] + q) % q;
    }
}

data_t modulo(data_t a, data_t b) {
    data_t c = (a % b + b) % b;
    if (c <= b /2) {
        return c;
    } else {
        return c - b;
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

// Test function
int main() {
    // Initialize test data
    data_t private_key[n];
    data_t public_key1[n];
    data_t public_key2[n];

    generate_key(private_key, public_key1, public_key2);
    for(int i = 0; i < n; i++) {
        printf("sk[%d] = %d\n", i, private_key[i]);
    }
    for(int i = 0; i < n; i++) {
        printf("pk[%d] = (%d, %d)\n", i, public_key1[i], public_key2[i]);
    }
    printf("\n");

    // int plaintext[n] = {0, 1, 0, 1};
    data_t plaintext[n];
    data_t ciphertext1_hls[n];
    data_t ciphertext2_hls[n];
    data_t ciphertext1_ref[n];
    data_t ciphertext2_ref[n];

    data_t decrypted_pt[n];
    
    std::cout << "Testing encryption/decryption with parameters:" << std::endl;
    std::cout << "POLYNOMIAL_DEGREE: " << POLYNOMIAL_DEGREE << std::endl;
    std::cout << "CIPHERTEXT_MODULUS: " << CIPHERTEXT_MODULUS << std::endl;
    std::cout << "PRIMITIVE_N_TH_ROOT_OF_UNITY: " << PRIMITIVE_N_TH_ROOT_OF_UNITY << std::endl;

    // for(int i = 0; i < n; i++) {
    //     plaintext[i] = 1;
    // }
    // print_array("plaintext", plaintext, POLYNOMIAL_DEGREE);

    // int error1[n] = {1, 1, 0, 0};
    // int error2[n] = {-1, 1, 0, 0};
    // int r[n] = {0, 0, 0, 0};
    // int r[n] = {4, 3, 2, 1};
    // encryption(error1, error2, r, public_key1, public_key2, plaintext, ciphertext1_hls, ciphertext2_hls);
    // encryption_reference(error1, error2, r, public_key1, public_key2, plaintext, ciphertext1_ref, ciphertext2_ref);

    // print_array("ciphertext1_hls", ciphertext1_hls, POLYNOMIAL_DEGREE);
    // print_array("ciphertext1_ref", ciphertext1_ref, POLYNOMIAL_DEGREE);
    // print_array("ciphertext2_hls", ciphertext2_hls, POLYNOMIAL_DEGREE);
    // print_array("ciphertext2_ref", ciphertext2_ref, POLYNOMIAL_DEGREE);

    // data_t decrypted_hls[n];
    // data_t decrypted_ref[n];
    // decryption(private_key, ciphertext1_hls, ciphertext2_hls, decrypted_hls);
    // decryption_reference(private_key, ciphertext1_ref, ciphertext2_ref, decrypted_ref);

    // print_array("decrypted_hls", decrypted_hls, POLYNOMIAL_DEGREE);
    // print_array("decrypted_ref", decrypted_ref, POLYNOMIAL_DEGREE);

    int diff = 0;
    // Seed random number generator
    srand(time(NULL));

    // Global flag for overall test success
    bool all_tests_passed = true;

    // Test 20 random plaintexts
    for (int test = 0; test < 20; ++test) {
        // Generate random plaintext values between 0 and p-1
        for (int j = 0; j < n; ++j) {
            plaintext[j] = rand() % p;
        }

        data_t error1[n];
        data_t error2[n];
        data_t r[n];

        std::uniform_int_distribution<int> dist1(-1, 1);
        // std::uniform_int_distribution<int> dist2(0, q - 1);

        for(int i = 0; i < n; i++) {
            error1[i] = (data_t) dist1(rng);
            // printf("error1[%d] = %d\n", i, error1[i]);
            error2[i] = (data_t) dist1(rng);
            // printf("error2[%d] = %d\n", i, error2[i]);
            r[i] = (data_t) dist1(rng);
            // printf("r[%d] = %d\n", i, r[i]);
        }

        // Print the current plaintext
        std::cout << "Test " << test + 1 << " - plaintext = {";
        for (int j = 0; j < n; ++j) {
            std::cout << plaintext[j];
            if (j != n - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;

        top_encryption_decryption_test(error1, error2, r, private_key, public_key1, public_key2, plaintext, decrypted_pt);

        encryption(error1, error2, r, public_key1, public_key2, plaintext, ciphertext1_hls, ciphertext2_hls);
        encryption_reference(error1, error2, r, public_key1, public_key2, plaintext, ciphertext1_ref, ciphertext2_ref);

        print_array("ciphertext1_hls", ciphertext1_hls, POLYNOMIAL_DEGREE);
        print_array("ciphertext1_ref", ciphertext1_ref, POLYNOMIAL_DEGREE);
        print_array("ciphertext2_hls", ciphertext2_hls, POLYNOMIAL_DEGREE);
        print_array("ciphertext2_ref", ciphertext2_ref, POLYNOMIAL_DEGREE);

        data_t decrypted_hls[n];
        data_t decrypted_ref[n];
        decryption(private_key, ciphertext1_hls, ciphertext2_hls, decrypted_hls);
        decryption_reference(private_key, ciphertext1_ref, ciphertext2_ref, decrypted_ref);
    
        print_array("decrypted_hls", decrypted_hls, POLYNOMIAL_DEGREE);
        print_array("decrypted_ref", decrypted_ref, POLYNOMIAL_DEGREE);
        print_array("decrypted_pt", decrypted_pt, POLYNOMIAL_DEGREE);

        bool test_passed = true;
        for(int i = 0; i < n; i++) {
            if ((plaintext[i] != decrypted_hls[i]) && (plaintext[i] - decrypted_hls[i] != p)) {
                printf("plaintext[%d] = %d != decrypted[%d] = %d\n", i, plaintext[i], i, decrypted_hls[i]);
                diff = 1;
                test_passed = false;
                all_tests_passed = false;
            }
        }
        
        if (test_passed) {
            std::cout << "Test " << test + 1 << " passed" << std::endl;
        } else {
            std::cout << "Test " << test + 1 << " failed" << std::endl;
        }
    }
    
    // // Verify results
    // bool match = true;
    // for (int i = 0; i < POLYNOMIAL_DEGREE; i++) {
    //     if (output_hls[i] != output_ref[i]) {
    //         if (hls::abs(output_hls[i]) + hls::abs(output_ref[i]) != CIPHERTEXT_MODULUS) {
    //             match = false;
    //         }
    //         std::cout << "Mismatch at index " << i << ": HLS=" << output_hls[i] 
    //                   << ", Ref=" << output_ref[i] << std::endl;
    //     }
    // }
    
    // if (match) {
    //     std::cout << "PASS: HLS implementation matches reference!" << std::endl;
    // } else {
    //     std::cout << "FAIL: Results don't match!" << std::endl;
    // }
    std::cout << "Test ended" << std::endl;
    
    // return match ? 0 : 1;
    return 0;
}