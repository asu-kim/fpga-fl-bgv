#ifndef FC_BWD_H
#define FC_BWD_H

#include <hls_stream.h>
#include <stdint.h>

template<int IN_DIM, int OUT_DIM>
void fc_backward(
        hls::stream<float> &X,
        hls::stream<float> &grads,
        hls::stream<float> &out_stream,
        const float weight[IN_DIM][OUT_DIM],
        float dW[IN_DIM][OUT_DIM],
        float dB[OUT_DIM],
        bool use_relu = true
        ) {
#pragma HLS INLINE off
    
    // load input to buffer
    float x_buffer[IN_DIM];
    for(int i=0; i<IN_DIM; ++i) {
#pragma HLS PIPELINE II=1
        x_buffer[i] = X.read();
    }

    // load grads to buffer
    float grads_buffer[OUT_DIM];
    for(int j=0; j<OUT_DIM; ++j) {
#pragma HLS PIPELINE II=1
        grads_buffer[j] = grads.read();
    }

    // bias grads
    for(int j=0; j<OUT_DIM; ++j) {
#pragma HLS PIPELINE II=1
        dB[j] = grads_buffer[j];
    }

    // weight grads
    for(int i=0; i<IN_DIM; ++i) {
        for(int j=0; j<OUT_DIM; ++j) {
            dW[i][j] = x_buffer[i] * grads_buffer[j];
        }
    }

    // input grads
    for(int i=0; i<IN_DIM; ++i) {
#pragma HLS PIPELINE II=1
        float acc = 0;
        for(int j=0; j<OUT_DIM; ++j) {
            acc += weight[i][j] * grads_buffer[j];
        }
        if(use_relu) acc = acc > 0 ? acc : 0;
        out_stream.write(acc);
    }
}

#endif
