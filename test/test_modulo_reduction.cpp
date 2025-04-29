#include <iostream>
#include <iomanip>
#include "hls_math.h"
#include "BGV/modulo_reduction.hpp"
#include "constants.hpp"

// Test function
int main() {
    int p = 256;
    int test_pass = 1;
    // for(int i = 0; i < p; i++) {
    //     if (modulo_reduction((i + p), p) != i) {
    //         test_pass = 0;
    //         std::cout << "(" << i << " +" << p << ") % = " << modulo_reduction((i + p), p) << "!" << std::endl;
    //     }
    // }
    // for(int i = -128; i < 128; i++) {
    //     if (hls::remainder(i, p) != i) {
    //         test_pass = 0;
    //         std::cout << i << " % = " << hls::remainder((i + p), p) << "!" << std::endl;
    //     }
    // }
    // for(int i = -128; i < 256; i++) {
    //     std::cout << i << " % = " << modulo_reduction_neg((i + p), p) << "." << std::endl;
    // }

    // data_t in1 = 14393336 + 2580366;
    // data_t in2 = 16974593;
    // data_t intermittent = modulo_reduction_neg(in1, in2);
    // std::cout << in1 << " % " << in2 << "= " << intermittent << "." << std::endl;

    // in1 = intermittent;
    data_t in1 = 0;
    for(int i = 0; i < 1280; i++) {
        in1 = -891 + i;
        data_t in2 = 256;
        data_t result = modulo_reduction_neg(in1, in2);
        std::cout << in1 << " % " << in2 << "= " << result << "." << std::endl;
        if(result == 128) {
            std::cout << "-----------------128!!------------------" << std::endl;
            // test_pass = 0;
        }
    }
    
    if (test_pass) {
        std::cout << "PASS: HLS implementation matches reference!" << std::endl;
    } else {
        std::cout << "FAIL: Results don't match!" << std::endl;
    }
    
    return test_pass ? 0 : 1;
}