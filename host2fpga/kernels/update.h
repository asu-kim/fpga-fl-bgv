#ifndef UPDATE_H
#define UPDATE_H

#define lr 1e-3

#include <hls_stream.h>

template<int M, int N>
void general_update(
        float weight[M][N],
        float dW[M][N],
        ) {
#pragma HLS PIPELINE
    for(int i=0; i<M; ++i) {
        for(int j=0; j<N; ++j) {
#pragma HLS UNROLL
            weight[i][j] -= lr * dW[i][j];
        }
    }
}

template<int OUT_C, int IN_C, int K>
void conv2d_update(
        float weight[OUT_C][IN_C][K][K],
        float dW[OUT_C][IN_C][K][K],
        ) {
    for(int k=0; k<OUT_C; ++k) {
#pragma HLS PIPELINE
        for(int i=0; i<IN_C; ++i) {
            for(int r=0; r<K; ++r) {
                for(int c=0; c<K; ++c) {
                    #pragma HLS UNROLL
                    weight[k][i][r][c] -= lr * dW[k][i][r][c];
                }
            }
        }
    }
}

#endif
