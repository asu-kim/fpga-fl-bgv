#include <iostream>
#include <cmath>
#include "lenet5/fc_layer.h"
#include "test_utils.h"

#define IN_DIM 256
#define OUT_DIM 128

#ifndef __SYNTHESIS__
int main() {
    hls::stream<data_t> in_stream, out_stream;
    data_t y_input[IN_DIM];
    data_t weight[OUT_DIM][IN_DIM];
    data_t bias[OUT_DIM];
    data_t y_output[OUT_DIM];
    bool use_relu = true;

    for(int j=0; j<OUT_DIM; ++j) {
        for(int i=0; i<IN_DIM; ++i) {
            data_t val = 0.01f * (i+j);
            weight[j][i] = val;
        }
    }

    for(int j=0; j<OUT_DIM; ++j) {
        data_t val = j;
        bias[j] = val;
    }

    for(int i=0; i<IN_DIM; ++i) {
        data_t val = i*0.5;
        in_stream.write(val);
        y_input[i] = val;
    }

    fc_layer<OUT_DIM, IN_DIM>(in_stream, out_stream, weight, bias, use_relu);

    for(int j=0; j<OUT_DIM; ++j) {
        y_output[j] = 0;
        for(int i=0; i<IN_DIM; ++i) {
            y_output[j] += y_input[i] * weight[j][i];
        }
        y_output[j] += bias[j];

        if(use_relu && y_output[j]<0) {
            y_output[j] = 0;
        }
    }

    int errs = 0;
    for(int j=0; j<OUT_DIM; ++j) {
        data_t out_val = out_stream.read();
        double golden = static_cast<double>(y_output[j]);
        if (fabs(static_cast<double>(out_val) - golden > 0.01)) {
            errs++;
            std::cout << "Error at output " << j << ": got " << out_val << ", expected " << y_output[j] << std::endl;
        }
    }

    std::cout << "Pooling Test: " << (errs ? "Fail" : "Pass") 
        << "(" << errs << " errors)\n";

    return errs;
}

#endif
