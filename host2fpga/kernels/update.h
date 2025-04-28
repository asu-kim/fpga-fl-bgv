#ifndef UPDATE_H
#define UPDATE_H

#define lr 1e-3

#include <hls_stream.h>

template<int N>
void bias_update(
        const float bias[N],
        const float dB[N],
        float update_bias[N]
        ) {
#pragma HLS PIPELINE
    for(int i=0; i<N; ++i) {
        update_bias[i] = bias[i];
    }

#pragma HLS PIPELINE
    for(int i=0; i<N; ++i) {
#pragma HLS UNROLL
        update_bias[i] -= lr * dB[i];
    }
}

template<int M, int N>
void general_update(
        const float weight[M*N],
        const float dW[M][N],
        float update_weight[M*N]
        ) {
#pragma HLS PIPELINE
    for(int i=0; i<M*N; ++i) {
        update_weight[i] = weight[i];
    }

#pragma HLS PIPELINE
    for(int i=0; i<M; ++i) {
        for(int j=0; j<N; ++j) {
            int idx = i*N + j;
#pragma HLS UNROLL
            update_weight[idx] -= lr * dW[i][j];
        }
    }
}

template<int OUT_C, int IN_C, int K>
void conv2d_update(
        const float weight[OUT_C*IN_C*K*K],
        const float dW[OUT_C][IN_C][K][K],
        float update_weight[OUT_C*IN_C*K*K]
        ) {
    for(int i=0; i<OUT_C * IN_C * K * K; ++i) {
        update_weight[i] = weight[i];
    }

    for(int k=0; k<OUT_C; ++k) {
#pragma HLS PIPELINE
        for(int i=0; i<IN_C; ++i) {
            for(int r=0; r<K; ++r) {
                for(int c=0; c<K; ++c) {
                    int idx = k*(IN_C*K*K) + i*(K*K) + r*K + c;
                    #pragma HLS UNROLL
                    update_weight[idx] -= lr * dW[k][i][r][c];
                }
            }
        }
    }
}

#endif
