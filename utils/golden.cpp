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

#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <cstdlib>
#include <ctime>

int n = 8;
int p = 256;
int q = 16974593;

std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));

void polynomial_multiplication_reference(int* input1, int* input2, int* output) {
    // int temp[POLYNOMIAL_DEGREE];
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

void generate_key(int* private_key, int* public_key1, int* public_key2) {
    // int a_prime[n] = {1, 2, 3, 4};
    // int error[n] = {1, 0, -1, 1};
    int a_prime[n];
    int error[n];

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

    int temp[n];
    polynomial_multiplication_reference(a_prime, private_key, temp);
    for(int i = 0; i < n; i++) {
        printf("temp[%d] = %d\n", i, temp[i]);
    }

    for(int i = 0; i < n; i++) {
        public_key1[i] = (temp[i] + p * error[i] + q) % q;
        public_key2[i] = -a_prime[i];
    }
}

void encryption(int* plaintext, int* publick_key1, int* publick_key2, int* ciphertext1, int* ciphertext2) {
    // int error1[n] = {1, 1, 0, 0};
    // int error2[n] = {-1, 1, 0, 0};
    // int r[n] = {0, 0, 0, 0};
    // int r[n] = {4, 3, 2, 1};
    int error1[n];
    int error2[n];
    int r[n];

    std::uniform_int_distribution<int> dist1(-1, 1);
    // std::uniform_int_distribution<int> dist2(0, q - 1);

    for(int i = 0; i < n; i++) {
        error1[i] = dist1(rng);
        // printf("error1[%d] = %d\n", i, error1[i]);
        error2[i] = dist1(rng);
        // printf("error2[%d] = %d\n", i, error2[i]);
        r[i] = dist1(rng);
        // printf("r[%d] = %d\n", i, r[i]);
    }

    int temp1[n];
    int temp2[n];

    polynomial_multiplication_reference(publick_key1, r, temp1);
    polynomial_multiplication_reference(publick_key2, r, temp2);

    for(int i = 0; i < n; i++) {
        ciphertext1[i] = (plaintext[i] + p * error1[i] + temp1[i] + q) % q;
        ciphertext2[i] = (p * error2[i] + temp2[i] + q) % q;
    }
}

int modulo(int a, int b) {
    int c = (a % b + b) % b;
    if (c <= b /2) {
        return c;
    } else {
        return c - b;
    }
    // return (a % b + b) % b;
}

void decryption(int* private_key, int* ciphertext1, int* ciphertext2, int* plaintext){
    int temp[n];
    polynomial_multiplication_reference(ciphertext2, private_key, temp);
    // for(int i = 0; i < n; i++) {
    //     printf("temp[%d] = %d\n", i, temp[i]);
    // }

    for(int i = 0; i < n; i++) {
        int intermittent = modulo(ciphertext1[i] + temp[i], q);
        // printf("intermittent = %d\n", intermittent);
        plaintext[i] = modulo(intermittent, p);
    }
    // for(int i = 0; i < n; i++) {
    //     int intermittent = (ciphertext1[i] + temp[i]) % q;
    //     // printf("intermittent = %d\n", intermittent);
    //     plaintext[i] = intermittent % p;
    //     // if (plaintext[i] < 0) {  // Convert from centered form to [0,p-1] range
    //     //     plaintext[i] += p;
    //     // }
    // }
}

int main(int argc, char** argv) {
    // int private_key[n] = {1, -1, 1, 0};
    int private_key[n];
    int public_key1[n];
    int public_key2[n];

    generate_key(private_key, public_key1, public_key2);
    for(int i = 0; i < n; i++) {
        printf("sk[%d] = %d\n", i, private_key[i]);
    }
    for(int i = 0; i < n; i++) {
        printf("pk[%d] = (%d, %d)\n", i, public_key1[i], public_key2[i]);
    }
    printf("\n");

    // int plaintext[n] = {0, 1, 0, 1};
    int plaintext[n];
    int ciphertext1[n];
    int ciphertext2[n];

    // for(int i = 0; i < n; i++) {
    //     plaintext[i] = 1;
    // }

    // encryption(plaintext, public_key1, public_key2, ciphertext1, ciphertext2);
    // for(int i = 0; i < n; i++) {
    //     printf("ct[%d] = (%d, %d)\n", i, ciphertext1[i], ciphertext2[i]);
    // }

    // int decrypted[n];

    // decryption(private_key, ciphertext1, ciphertext2, decrypted);
    // for(int i = 0; i < n; i++) {
    //     printf("decrypted[%d] = %d\n", i, decrypted[i]);
    // }

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

        // Print the current plaintext
        std::cout << "Test " << test + 1 << " - plaintext = {";
        for (int j = 0; j < n; ++j) {
            std::cout << plaintext[j];
            if (j != n - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        encryption(plaintext, public_key1, public_key2, ciphertext1, ciphertext2);
        
        int decrypted[n];
        decryption(private_key, ciphertext1, ciphertext2, decrypted);

        bool test_passed = true;
        for(int i = 0; i < n; i++) {
            if (plaintext[i] != decrypted[i] && (plaintext[i] - decrypted[i]) != p) {
                printf("plaintext[%d] = %d != decrypted[%d] = %d\n", i, plaintext[i], i, decrypted[i]);
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

    // Print overall test result
    if (all_tests_passed) {
        std::cout << "\nALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "\nTEST SUITE FAILED" << std::endl;
    }
    return 0;
}
