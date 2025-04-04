#include <iostream>
#include <cmath>
#include "conv2d.h"
#include "test_utils.h"

#define IN_ROWS 28
#define IN_COLS 28
#define KERNEL_SIZE 5
#define OUT_C 6

int main() {
    hls::stream<data_t> in_stream, out_stream;
    data_t weights[OUT_C][KERNEL_SIZE][KERNEL_SIZE];
    data_t bias[OUT_C];

    for(int channel=0; channel<OUT_C; ++channel) {
        bias[channel] = 0.0;
        for(int i=0; i<KERNEL_SIZE; ++i) {
            for(int j=0; j<KERNEL_SIZE; ++j) {
                weights[channel][i][j] = 1.0;
            }
        }
    }

    for(int i=0; i<IN_ROWS * IN_COLS; ++i) {
        in_stream.write(data_t(1));
    }

    // run conv2d
    conv2d<OUT_C, KERNEL_SIZE>(in_stream, out_stream, weights, bias, IN_ROWS, IN_COLS);

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

    std::cout << "Convolution test: " << (errors ? "Failed" : "Pass") 
        << " (" << errors << " errors)\n";

    return errors;
}

