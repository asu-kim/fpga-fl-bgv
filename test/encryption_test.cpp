// #include "hls_stream.h"
#include "encryption.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

// Verification function to compare hardware results with expected values
void verify_results(const std::vector<data_t>& sw_results, 
                   const std::vector<data_t>& hw_results, 
                   int size) {
    bool match = true;
    
    std::cout << "Verification:" << std::endl;
    std::cout << std::setw(15) << "Index" 
              << std::setw(15) << "SW Result" 
              << std::setw(15) << "HW Result" 
              << std::setw(15) << "Status" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int i = 0; i < size; i++) {
        // Direct bit-level comparison for exact matching
        bool equal = (sw_results[i].to_double() == hw_results[i].to_double());
        
        // If exact match fails, check if within reasonable range for fixed-point
        if (!equal) {
            double sw_val = sw_results[i].to_double();
            double hw_val = hw_results[i].to_double();
            equal = (sw_val > hw_val) ? 
                   (sw_val - hw_val < 0.01) : 
                   (hw_val - sw_val < 0.01);
        }
        
        match &= equal;
        
        std::cout << std::setw(15) << i
                  << std::setw(15) << sw_results[i].to_double()
                  << std::setw(15) << hw_results[i].to_double()
                  << std::setw(15) << (equal ? "PASS" : "FAIL") << std::endl;
    }
    
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Test " << (match ? "PASSED" : "FAILED") << std::endl;
}

// Software implementation of remainder function for verification
data_t sw_remainder(data_t a, data_t b) {
    data_t quotient = a / b;
    int whole_part = quotient.to_int();
    return a - (whole_part * b);
}

int main() {
    // Initialize data
    const int size = 4;
    std::vector<data_t> source_in1(size);
    std::vector<data_t> source_in2(size);
    std::vector<data_t> source_hw_results(size);
    std::vector<data_t> source_sw_results(size);
    
    // Set input values
    source_in1[0] = 3.5;
    source_in2[0] = 3.0;

    source_in1[1] = 7.2;
    source_in2[1] = 3.0;

    source_in1[2] = 15.5;
    source_in2[2] = 6.2;

    source_in1[3] = 38.1;
    source_in2[3] = 1.2;
    
    // Print test input values
    std::cout << "Testbench for encryption HLS module" << std::endl;
    std::cout << "Input values:" << std::endl;
    std::cout << std::setw(15) << "Index" 
              << std::setw(15) << "In1" 
              << std::setw(15) << "In2" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (int i = 0; i < size; i++) {
        std::cout << std::setw(15) << i
                  << std::setw(15) << source_in1[i].to_double()
                  << std::setw(15) << source_in2[i].to_double() << std::endl;
    }
    std::cout << std::endl;
    
    // Calculate software results for verification
    for (int i = 0; i < size; i++) {
        source_sw_results[i] = sw_remainder(source_in1[i], source_in2[i]);
    }
    
    // Call the hardware function
    encryption(source_in1.data(), source_in2.data(), source_hw_results.data(), size);
    
    // Verify results
    verify_results(source_sw_results, source_hw_results, size);
    
    return 0;
}