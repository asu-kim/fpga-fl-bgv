#ifndef FC_BWD_H
#define FC_BWD_H

#include "data_type.hpp"

template<int IN_DIM, int OUT_DIM>
void fc_backward(
        const data_ap_fixed_t in_activation[IN_DIM],
        const data_ap_fixed_t grads[OUT_DIM],
        const data_ap_fixed_t in_weight[IN_DIM*OUT_DIM],
        data_ap_fixed_t dX[IN_DIM],
        data_ap_fixed_t dW[IN_DIM*OUT_DIM],
        data_ap_fixed_t dB[OUT_DIM],
        bool use_relu = true
        ) {
// #pragma HLS INLINE off

// #pragma HLS ARRAY_PARTITION variable=in_weight complete dim=2
// #pragma HLS ARRAY_PARTITION variable=grads   complete dim=1
// #pragma HLS ARRAY_PARTITION variable=dW      complete dim=2
// #pragma HLS ARRAY_PARTITION variable=dB      complete dim=1
    
    // bias grads
    for(int j=0; j<OUT_DIM; ++j) {
#pragma HLS PIPELINE II=1
        dB[j] = grads[j];
    }

    // weight grads
    for(int i=0; i<IN_DIM; ++i) {
        for(int j=0; j<OUT_DIM; ++j) {
#pragma HLS PIPELINE II=1
            int idx = i * OUT_DIM + j;
            dW[idx] = in_activation[i] * grads[j];
        }
    }

    // input grads
    for(int i=0; i<IN_DIM; ++i) {
#pragma HLS PIPELINE II=2
        data_ap_fixed_t acc = 0;
        for(int j=0; j<OUT_DIM; ++j) {
#pragma HLS UNROLL factor=2
            int idx = i * OUT_DIM + j;
            acc += in_weight[idx] * grads[j];
        }
        if(use_relu) {
            acc *= (in_activation[i] > 0 ? data_ap_fixed_t(1.0) : data_ap_fixed_t(0.0));
        }
        dX[i] = acc;
    }
}

#endif
