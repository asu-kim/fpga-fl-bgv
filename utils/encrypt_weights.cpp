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
#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <regex>

#include <../data/weights_bias.h>

int n = 128;
int p = 256;
int q = 16974593;

std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));

// Helper function to print array
void print_array(const char* name, int* arr, int size) {
    std::cout << name << ": [";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void polynomial_multiplication_reference(int* input1, int* input2, int* output) {
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
    }

    int temp[n];
    polynomial_multiplication_reference(a_prime, private_key, temp);
    // for(int i = 0; i < n; i++) {
    //     printf("temp[%d] = %d\n", i, temp[i]);
    // }

    for(int i = 0; i < n; i++) {
        public_key1[i] = (temp[i] + p * error[i] + q) % q;
        public_key2[i] = -a_prime[i];
    }
}

void encryption(int* plaintext, int* publick_key1, int* publick_key2, int* ciphertext1, int* ciphertext2) {
    int error1[n];
    int error2[n];
    int r[n];

    std::uniform_int_distribution<int> dist1(-1, 1);
    // std::uniform_int_distribution<int> dist2(0, q - 1);

    for(int i = 0; i < n; i++) {
        error1[i] = dist1(rng);
        error2[i] = dist1(rng);
        r[i] = dist1(rng);
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
    if (c < b /2) {
        return c;
    } else {
        return c - b;
    }
    // return (a % b + b) % b;
}

void decryption(int* private_key, int* ciphertext1, int* ciphertext2, int* plaintext){
    int temp[n];
    polynomial_multiplication_reference(ciphertext2, private_key, temp);

    for(int i = 0; i < n; i++) {
        int intermittent = modulo(ciphertext1[i] + temp[i], q);
        // printf("intermittent = %d\n", intermittent);
        plaintext[i] = modulo(intermittent, p);
    }
}

// Find a substring and return its position
size_t find_substring(const std::string& str, const std::string& sub) {
    return str.find(sub);
}

// Function to save keys to a file
void save_keys(const int* private_key, const int* public_key1, const int* public_key2, const std::string& filename) {
    std::ofstream outfile(filename);
    
    // Write header guards and includes
    outfile << "#ifndef KEYS_H\n";
    outfile << "#define KEYS_H\n\n";
    outfile << "#include <cstdint>\n\n";
    
    // Write private key
    outfile << "static const int32_t PRIVATE_KEY[" << n << "] = {\n  ";
    for (int i = 0; i < n; i++) {
        outfile << private_key[i];
        if (i < n - 1) outfile << ", ";
        if ((i + 1) % 10 == 0 && i != n - 1) outfile << "\n  ";
    }
    outfile << "\n};\n\n";
    
    // Write public key 1
    outfile << "static const int32_t PUBLIC_KEY1[" << n << "] = {\n  ";
    for (int i = 0; i < n; i++) {
        outfile << public_key1[i];
        if (i < n - 1) outfile << ", ";
        if ((i + 1) % 10 == 0 && i != n - 1) outfile << "\n  ";
    }
    outfile << "\n};\n\n";
    
    // Write public key 2
    outfile << "static const int32_t PUBLIC_KEY2[" << n << "] = {\n  ";
    for (int i = 0; i < n; i++) {
        outfile << public_key2[i];
        if (i < n - 1) outfile << ", ";
        if ((i + 1) % 10 == 0 && i != n - 1) outfile << "\n  ";
    }
    outfile << "\n};\n\n";
    
    // Close header guard
    outfile << "#endif // KEYS_H\n";
    
    outfile.close();
}

int main(int argc, char** argv) {
    int private_key[n];
    int public_key1[n];
    int public_key2[n];

    generate_key(private_key, public_key1, public_key2);

    save_keys(private_key, public_key1, public_key2, "../data/keys.h");
    
    // Global flag for overall test success
    bool all_tests_passed = true;
    
    // Open output file for the encrypted header
    std::ofstream encrypted_header;
    encrypted_header.open("../data/encrypted_weights_bias.h");
    
    // Write header guards and includes
    encrypted_header << "#ifndef ENCRYPTED_WEIGHTS_BIAS_H\n";
    encrypted_header << "#define ENCRYPTED_WEIGHTS_BIAS_H\n\n";
    encrypted_header << "#include <cstdint>\n\n";

    // Process CONV1_BIAS_INT8_DATA
    int num_bias = 1;
    for(int i = 0; i < sizeof(CONV1_BIAS_INT8_SHAPE) / sizeof(int); i++) {
        num_bias *= CONV1_BIAS_INT8_SHAPE[i];
    }
    std::cout << "CONV1_BIAS num elements: " << num_bias << std::endl;

    int bias_padded[n];
    int bias_padded_encrypted[2][n];
    int bias_decrypted[n];
    for(int i = 0; i < n; i++) {
        if(i < num_bias) {
            bias_padded[i] = CONV1_BIAS_INT8_DATA[i];
        } else {
            bias_padded[i] = 0;
        }
    }
    
    encryption(bias_padded, public_key1, public_key2, bias_padded_encrypted[0], bias_padded_encrypted[1]);
    decryption(private_key, bias_padded_encrypted[0], bias_padded_encrypted[1], bias_decrypted);

    bool test_passed = true;
    for(int i = 0; i < n; i++) {
        if(bias_padded[i] != bias_decrypted[i]) {
            printf("CONV1_BIAS: plaintext[%d] = %d != decrypted[%d] = %d\n", i, bias_padded[i], i, bias_decrypted[i]);
            test_passed = false;
            all_tests_passed = false;
        }
    }
    
    if(test_passed) {
        std::cout << "CONV1_BIAS encryption test passed" << std::endl;
    } else {
        std::cout << "CONV1_BIAS encryption test failed" << std::endl;
        exit(1);
    }
    
    // Write encrypted CONV1_BIAS to the header
    encrypted_header << "static const int32_t CONV1_BIAS_INT8_DATA_ENC1[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[0][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n";
    
    encrypted_header << "static const int32_t CONV1_BIAS_INT8_DATA_ENC2[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[1][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n\n";
    
    // Process CONV1_WEIGHT_INT8_DATA
    int num_weight = 1;
    for(int i = 0; i < sizeof(CONV1_WEIGHT_INT8_SHAPE) / sizeof(int); i++) {
        num_weight *= CONV1_WEIGHT_INT8_SHAPE[i];
    }
    std::cout << "CONV1_WEIGHT num elements: " << num_weight << std::endl;
    int num_slices = (int)std::ceil((double)num_weight / n);
    std::cout << "CONV1_WEIGHT num slices: " << num_slices << std::endl;
    
    for(int slice = 0; slice < num_slices; slice++) {
        int weights_padded[n];
        int weights_padded_encrypted[2][n];
        int weights_decrypted[n];
        
        for(int j = 0; j < n; j++) {
            int index = slice * n + j;
            if(index < num_weight) {
                weights_padded[j] = CONV1_WEIGHT_INT8_DATA[index];
            } else {
                weights_padded[j] = 0;
            }
        }
        
        encryption(weights_padded, public_key1, public_key2, weights_padded_encrypted[0], weights_padded_encrypted[1]);
        decryption(private_key, weights_padded_encrypted[0], weights_padded_encrypted[1], weights_decrypted);
        
        test_passed = true;
        for(int j = 0; j < n; j++) {
            if(weights_padded[j] != weights_decrypted[j]) {
                printf("CONV1_WEIGHT slice %d: plaintext[%d] = %d != decrypted[%d] = %d\n", 
                       slice, j, weights_padded[j], j, weights_decrypted[j]);
                test_passed = false;
                all_tests_passed = false;
            }
        }
        
        if(test_passed) {
            std::cout << "CONV1_WEIGHT slice " << slice << " encryption test passed" << std::endl;
        } else {
            std::cout << "CONV1_WEIGHT slice " << slice << " encryption test failed" << std::endl;
            exit(1);
        }
        
        // Write encrypted CONV1_WEIGHT slice to the header
        encrypted_header << "static const int32_t CONV1_WEIGHT_INT8_DATA" << slice << "_ENC1[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[0][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n";
        
        encrypted_header << "static const int32_t CONV1_WEIGHT_INT8_DATA" << slice << "_ENC2[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[1][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n\n";
    }
    
    // Process CONV2_BIAS_INT8_DATA
    num_bias = 1;
    for(int i = 0; i < sizeof(CONV2_BIAS_INT8_SHAPE) / sizeof(int); i++) {
        num_bias *= CONV2_BIAS_INT8_SHAPE[i];
    }
    std::cout << "CONV2_BIAS num elements: " << num_bias << std::endl;

    for(int i = 0; i < n; i++) {
        if(i < num_bias) {
            bias_padded[i] = CONV2_BIAS_INT8_DATA[i];
        } else {
            bias_padded[i] = 0;
        }
    }
    
    encryption(bias_padded, public_key1, public_key2, bias_padded_encrypted[0], bias_padded_encrypted[1]);
    decryption(private_key, bias_padded_encrypted[0], bias_padded_encrypted[1], bias_decrypted);

    test_passed = true;
    for(int i = 0; i < n; i++) {
        if(bias_padded[i] != bias_decrypted[i]) {
            printf("CONV2_BIAS: plaintext[%d] = %d != decrypted[%d] = %d\n", i, bias_padded[i], i, bias_decrypted[i]);
            test_passed = false;
            all_tests_passed = false;
        }
    }
    
    if(test_passed) {
        std::cout << "CONV2_BIAS encryption test passed" << std::endl;
    } else {
        std::cout << "CONV2_BIAS encryption test failed" << std::endl;
        exit(1);
    }
    
    // Write encrypted CONV2_BIAS to the header
    encrypted_header << "static const int32_t CONV2_BIAS_INT8_DATA_ENC1[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[0][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n";
    
    encrypted_header << "static const int32_t CONV2_BIAS_INT8_DATA_ENC2[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[1][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n\n";
        
    // Process CONV2_WEIGHT_INT8_DATA
    num_weight = 1;
    for(int i = 0; i < sizeof(CONV2_WEIGHT_INT8_SHAPE) / sizeof(int); i++) {
        num_weight *= CONV2_WEIGHT_INT8_SHAPE[i];
    }
    std::cout << "CONV2_WEIGHT num elements: " << num_weight << std::endl;
    num_slices = (int)std::ceil((double)num_weight / n);
    std::cout << "CONV2_WEIGHT num slices: " << num_slices << std::endl;

    for(int slice = 0; slice < num_slices; slice++) {
        int weights_padded[n];
        int weights_padded_encrypted[2][n];
        int weights_decrypted[n];
        
        for(int j = 0; j < n; j++) {
            int index = slice * n + j;
            if(index < num_weight) {
                weights_padded[j] = CONV2_WEIGHT_INT8_DATA[index];
            } else {
                weights_padded[j] = 0;
            }
        }
        
        encryption(weights_padded, public_key1, public_key2, weights_padded_encrypted[0], weights_padded_encrypted[1]);
        decryption(private_key, weights_padded_encrypted[0], weights_padded_encrypted[1], weights_decrypted);
        
        test_passed = true;
        for(int j = 0; j < n; j++) {
            if(weights_padded[j] != weights_decrypted[j]) {
                printf("CONV2_WEIGHT slice %d: plaintext[%d] = %d != decrypted[%d] = %d\n", 
                    slice, j, weights_padded[j], j, weights_decrypted[j]);
                test_passed = false;
                all_tests_passed = false;
            }
        }
        
        if(test_passed) {
            std::cout << "CONV2_WEIGHT slice " << slice << " encryption test passed" << std::endl;
        } else {
            std::cout << "CONV2_WEIGHT slice " << slice << " encryption test failed" << std::endl;
            exit(1);
        }
        
        // Write encrypted CONV2_WEIGHT slice to the header
        encrypted_header << "static const int32_t CONV2_WEIGHT_INT8_DATA" << slice << "_ENC1[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[0][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n";
        
        encrypted_header << "static const int32_t CONV2_WEIGHT_INT8_DATA" << slice << "_ENC2[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[1][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n\n";
    }

    // Process FC1_BIAS_INT8_DATA
    num_bias = 1;
    for(int i = 0; i < sizeof(FC1_BIAS_INT8_SHAPE) / sizeof(int); i++) {
        num_bias *= FC1_BIAS_INT8_SHAPE[i];
    }
    std::cout << "FC1_BIAS num elements: " << num_bias << std::endl;

    for(int i = 0; i < n; i++) {
        if(i < num_bias) {
            bias_padded[i] = FC1_BIAS_INT8_DATA[i];
        } else {
            bias_padded[i] = 0;
        }
    }

    encryption(bias_padded, public_key1, public_key2, bias_padded_encrypted[0], bias_padded_encrypted[1]);
    decryption(private_key, bias_padded_encrypted[0], bias_padded_encrypted[1], bias_decrypted);

    test_passed = true;
    for(int i = 0; i < n; i++) {
        if(bias_padded[i] != bias_decrypted[i]) {
            printf("FC1_BIAS: plaintext[%d] = %d != decrypted[%d] = %d\n", i, bias_padded[i], i, bias_decrypted[i]);
            test_passed = false;
            all_tests_passed = false;
        }
    }

    if(test_passed) {
        std::cout << "FC1_BIAS encryption test passed" << std::endl;
    } else {
        std::cout << "FC1_BIAS encryption test failed" << std::endl;
        exit(1);
    }

    // Write encrypted FC1_BIAS to the header
    encrypted_header << "static const int32_t FC1_BIAS_INT8_DATA_ENC1[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[0][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n";

    encrypted_header << "static const int32_t FC1_BIAS_INT8_DATA_ENC2[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[1][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n\n";

    // Process FC1_WEIGHT_INT8_DATA
    num_weight = 1;
    for(int i = 0; i < sizeof(FC1_WEIGHT_INT8_SHAPE) / sizeof(int); i++) {
        num_weight *= FC1_WEIGHT_INT8_SHAPE[i];
    }
    std::cout << "FC1_WEIGHT num elements: " << num_weight << std::endl;
    num_slices = (int)std::ceil((double)num_weight / n);
    std::cout << "FC1_WEIGHT num slices: " << num_slices << std::endl;

    for(int slice = 0; slice < num_slices; slice++) {
        int weights_padded[n];
        int weights_padded_encrypted[2][n];
        int weights_decrypted[n];
        
        for(int j = 0; j < n; j++) {
            int index = slice * n + j;
            if(index < num_weight) {
                weights_padded[j] = FC1_WEIGHT_INT8_DATA[index];
            } else {
                weights_padded[j] = 0;
            }
        }
        
        encryption(weights_padded, public_key1, public_key2, weights_padded_encrypted[0], weights_padded_encrypted[1]);
        decryption(private_key, weights_padded_encrypted[0], weights_padded_encrypted[1], weights_decrypted);
        
        test_passed = true;
        for(int j = 0; j < n; j++) {
            if(weights_padded[j] != weights_decrypted[j]) {
                printf("FC1_WEIGHT slice %d: plaintext[%d] = %d != decrypted[%d] = %d\n", 
                    slice, j, weights_padded[j], j, weights_decrypted[j]);
                test_passed = false;
                all_tests_passed = false;
            }
        }
        
        if(test_passed) {
            std::cout << "FC1_WEIGHT slice " << slice << " encryption test passed" << std::endl;
        } else {
            std::cout << "FC1_WEIGHT slice " << slice << " encryption test failed" << std::endl;
            exit(1);
        }
        
        // Write encrypted FC1_WEIGHT slice to the header
        encrypted_header << "static const int32_t FC1_WEIGHT_INT8_DATA" << slice << "_ENC1[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[0][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n";
        
        encrypted_header << "static const int32_t FC1_WEIGHT_INT8_DATA" << slice << "_ENC2[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[1][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n\n";
    }

    // Process FC2_BIAS_INT8_DATA
    num_bias = 1;
    for(int i = 0; i < sizeof(FC2_BIAS_INT8_SHAPE) / sizeof(int); i++) {
        num_bias *= FC2_BIAS_INT8_SHAPE[i];
    }
    std::cout << "FC2_BIAS num elements: " << num_bias << std::endl;

    for(int i = 0; i < n; i++) {
        if(i < num_bias) {
            bias_padded[i] = FC2_BIAS_INT8_DATA[i];
        } else {
            bias_padded[i] = 0;
        }
    }

    encryption(bias_padded, public_key1, public_key2, bias_padded_encrypted[0], bias_padded_encrypted[1]);
    decryption(private_key, bias_padded_encrypted[0], bias_padded_encrypted[1], bias_decrypted);

    test_passed = true;
    for(int i = 0; i < n; i++) {
        if(bias_padded[i] != bias_decrypted[i]) {
            printf("FC2_BIAS: plaintext[%d] = %d != decrypted[%d] = %d\n", i, bias_padded[i], i, bias_decrypted[i]);
            test_passed = false;
            all_tests_passed = false;
        }
    }

    if(test_passed) {
        std::cout << "FC2_BIAS encryption test passed" << std::endl;
    } else {
        std::cout << "FC2_BIAS encryption test failed" << std::endl;
        exit(1);
    }

    // Write encrypted FC2_BIAS to the header
    encrypted_header << "static const int32_t FC2_BIAS_INT8_DATA_ENC1[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[0][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n";

    encrypted_header << "static const int32_t FC2_BIAS_INT8_DATA_ENC2[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[1][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n\n";

    // Process FC2_WEIGHT_INT8_DATA
    num_weight = 1;
    for(int i = 0; i < sizeof(FC2_WEIGHT_INT8_SHAPE) / sizeof(int); i++) {
        num_weight *= FC2_WEIGHT_INT8_SHAPE[i];
    }
    std::cout << "FC2_WEIGHT num elements: " << num_weight << std::endl;
    num_slices = (int)std::ceil((double)num_weight / n);
    std::cout << "FC2_WEIGHT num slices: " << num_slices << std::endl;

    for(int slice = 0; slice < num_slices; slice++) {
        int weights_padded[n];
        int weights_padded_encrypted[2][n];
        int weights_decrypted[n];
        
        for(int j = 0; j < n; j++) {
            int index = slice * n + j;
            if(index < num_weight) {
                weights_padded[j] = FC2_WEIGHT_INT8_DATA[index];
            } else {
                weights_padded[j] = 0;
            }
        }
        
        encryption(weights_padded, public_key1, public_key2, weights_padded_encrypted[0], weights_padded_encrypted[1]);
        decryption(private_key, weights_padded_encrypted[0], weights_padded_encrypted[1], weights_decrypted);
        
        test_passed = true;
        for(int j = 0; j < n; j++) {
            if(weights_padded[j] != weights_decrypted[j]) {
                printf("FC2_WEIGHT slice %d: plaintext[%d] = %d != decrypted[%d] = %d\n", 
                    slice, j, weights_padded[j], j, weights_decrypted[j]);
                test_passed = false;
                all_tests_passed = false;
            }
        }
        
        if(test_passed) {
            std::cout << "FC2_WEIGHT slice " << slice << " encryption test passed" << std::endl;
        } else {
            std::cout << "FC2_WEIGHT slice " << slice << " encryption test failed" << std::endl;
            exit(1);
        }
        
        // Write encrypted FC2_WEIGHT slice to the header
        encrypted_header << "static const int32_t FC2_WEIGHT_INT8_DATA" << slice << "_ENC1[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[0][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n";
        
        encrypted_header << "static const int32_t FC2_WEIGHT_INT8_DATA" << slice << "_ENC2[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[1][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n\n";
    }

    // Process FC3_BIAS_INT8_DATA
    num_bias = 1;
    for(int i = 0; i < sizeof(FC3_BIAS_INT8_SHAPE) / sizeof(int); i++) {
        num_bias *= FC3_BIAS_INT8_SHAPE[i];
    }
    std::cout << "FC3_BIAS num elements: " << num_bias << std::endl;

    for(int i = 0; i < n; i++) {
        if(i < num_bias) {
            bias_padded[i] = FC3_BIAS_INT8_DATA[i];
        } else {
            bias_padded[i] = 0;
        }
    }

    encryption(bias_padded, public_key1, public_key2, bias_padded_encrypted[0], bias_padded_encrypted[1]);
    decryption(private_key, bias_padded_encrypted[0], bias_padded_encrypted[1], bias_decrypted);

    test_passed = true;
    for(int i = 0; i < n; i++) {
        if(bias_padded[i] != bias_decrypted[i]) {
            printf("FC3_BIAS: plaintext[%d] = %d != decrypted[%d] = %d\n", i, bias_padded[i], i, bias_decrypted[i]);
            test_passed = false;
            all_tests_passed = false;
        }
    }

    if(test_passed) {
        std::cout << "FC3_BIAS encryption test passed" << std::endl;
    } else {
        std::cout << "FC3_BIAS encryption test failed" << std::endl;
        exit(1);
    }

    // Write encrypted FC3_BIAS to the header
    encrypted_header << "static const int32_t FC3_BIAS_INT8_DATA_ENC1[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[0][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n";

    encrypted_header << "static const int32_t FC3_BIAS_INT8_DATA_ENC2[" << n << "] = {\n  ";
    for(int i = 0; i < n; i++) {
        encrypted_header << bias_padded_encrypted[1][i];
        if(i < n - 1) encrypted_header << ", ";
        if((i + 1) % 10 == 0 && i != n - 1) encrypted_header << "\n  ";
    }
    encrypted_header << "\n};\n\n";

    // Process FC3_WEIGHT_INT8_DATA
    num_weight = 1;
    for(int i = 0; i < sizeof(FC3_WEIGHT_INT8_SHAPE) / sizeof(int); i++) {
        num_weight *= FC3_WEIGHT_INT8_SHAPE[i];
    }
    std::cout << "FC3_WEIGHT num elements: " << num_weight << std::endl;
    num_slices = (int)std::ceil((double)num_weight / n);
    std::cout << "FC3_WEIGHT num slices: " << num_slices << std::endl;

    for(int slice = 0; slice < num_slices; slice++) {
        int weights_padded[n];
        int weights_padded_encrypted[2][n];
        int weights_decrypted[n];
        
        for(int j = 0; j < n; j++) {
            int index = slice * n + j;
            if(index < num_weight) {
                weights_padded[j] = FC3_WEIGHT_INT8_DATA[index];
            } else {
                weights_padded[j] = 0;
            }
        }
        
        encryption(weights_padded, public_key1, public_key2, weights_padded_encrypted[0], weights_padded_encrypted[1]);
        decryption(private_key, weights_padded_encrypted[0], weights_padded_encrypted[1], weights_decrypted);
        
        test_passed = true;
        for(int j = 0; j < n; j++) {
            if(weights_padded[j] != weights_decrypted[j]) {
                printf("FC3_WEIGHT slice %d: plaintext[%d] = %d != decrypted[%d] = %d\n", 
                    slice, j, weights_padded[j], j, weights_decrypted[j]);
                test_passed = false;
                all_tests_passed = false;
            }
        }
        
        if(test_passed) {
            std::cout << "FC3_WEIGHT slice " << slice << " encryption test passed" << std::endl;
        } else {
            std::cout << "FC3_WEIGHT slice " << slice << " encryption test failed" << std::endl;
            exit(1);
        }
        
        // Write encrypted FC3_WEIGHT slice to the header
        encrypted_header << "static const int32_t FC3_WEIGHT_INT8_DATA" << slice << "_ENC1[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[0][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n";
        
        encrypted_header << "static const int32_t FC3_WEIGHT_INT8_DATA" << slice << "_ENC2[" << n << "] = {\n  ";
        for(int j = 0; j < n; j++) {
            encrypted_header << weights_padded_encrypted[1][j];
            if(j < n - 1) encrypted_header << ", ";
            if((j + 1) % 10 == 0 && j != n - 1) encrypted_header << "\n  ";
        }
        encrypted_header << "\n};\n\n";
    }
    
    // Close the header with endif
    encrypted_header << "#endif // ENCRYPTED_WEIGHTS_BIAS_H\n";
    encrypted_header.close();
    
    if(all_tests_passed) {
        std::cout << "All encryption tests passed" << std::endl;
    } else {
        std::cout << "Some encryption tests failed" << std::endl;
        return 1;
    }
    
    std::cout << "Encrypted weights and biases written to encrypted_weights_bias.h" << std::endl;
    
    return 0;
}
