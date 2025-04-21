#include <iostream>
#include <cmath>
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"
#include "test_utils.h"

#define IN_ROWS 28
#define IN_COLS 28
#define KERNEL_SIZE 5
#define IN_C 1
#define OUT_C 6

#ifndef __SYNTHESIS__
int main() {
    hls::stream<data_t> in_stream, out_stream;
    data_t weights[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE];
    data_t bias[OUT_C];

    Parameter param_obj;
    Parameter* param = &param_obj;

    std::cout << "check 1" << std::endl;
    for(int channel=0; channel<OUT_C; ++channel) {
        for(int ic=0; ic<IN_C; ++ic) {
            bias[channel] = 0.0;
            param->conv1_bias[channel] = 0.0;
            for(int i=0; i<KERNEL_SIZE; ++i) {
                for(int j=0; j<KERNEL_SIZE; ++j) {
                    weights[channel][ic][i][j] = 1.0;
                    param->conv1_weight[channel][ic][i][j] = 1.0;
                }
            }
        }
    }
    std::cout << "check 2" << std::endl;

    for(int i=0; i<IN_ROWS * IN_COLS; ++i) {
        in_stream.write(data_t(1));
    }
    std::cout << "check 3" << std::endl;

    // run conv2d
    // conv1(in_stream, out_stream, param);
    conv1(in_stream, out_stream, param->conv1_weight, param->conv1_bias);
    std::cout << "check 4" << std::endl;

    int errors = 0;
    const int output_size = (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1) * OUT_C;
    for(int i=0; i<output_size; ++i) {
        data_t val = out_stream.read();
        data_t golden = 25.0;
        if(fabs(static_cast<double>(val - golden)) > 0.01) {
            errors ++;
            std::cout << "Error at " << i << " : got " << val << " expected " << golden << std::endl;    
        }
    }
    std::cout << "check 5" << std::endl;

    std::cout << "Convolution test: " << (errors ? "Failed" : "Pass") 
        << " (" << errors << " errors)\n";

    return errors;
}

#endif
