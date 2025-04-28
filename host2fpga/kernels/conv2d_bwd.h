#ifndef CONV_2D_BWD_H
#define CONV_2D_BWD_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <stdint.h>

/**
 * Notes.
 * - Inference: X @ K + b
 *   sum over i, kr, kc of X[i, r+kr, c+kc] * W[k, i, kr, kc] + b[k]
 * - delta = dL / dY
 * - Bias grads: channel-wise sum 
 * - Weight graids: sliding window over original input
 * - Input grads: flip conv d_out with weight
 */

// bias grads
template<int OUT_C, int OUT_H, int OUT_W>
void bias_grad(
        const float grads[OUT_C][OUT_H][OUT_W],
        float dB[OUT_C]
        ) {
#pragma HLS INLINE
    for(int k=0; k<OUT_C; ++k) dB[k] = 0;

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<OUT_H; ++r) {
            for(int c=0; c<OUT_W; ++c) {
#pragma HLS PIPELINE II=1
                dB[k] += grads[k][r][c];
            }
        }
    }
}

// wieght grads
template<int OUT_C, int IN_C, int K, int H, int W>
void weight_grad(
        const float X[IN_C][H][W],
        const float grads[OUT_C][H-K+1][W-K+1],
        float dW[OUT_C][IN_C][K][K]
        ) {
#pragma HLS INLINE
    for(int k=0; k<OUT_C; ++k) {
        for(int i=0; i<IN_C; ++i) {
            for(int r=0; r<K; ++r) {
                for(int c=0; c<K; ++c) {
                    dW[k][i][r][c] = 0;
                }
            }
        }
    }

    constexpr int OUT_H = H-K+1;
    constexpr int OUT_W = W-K+1;

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<OUT_H; ++r) {
            for(int c=0; c<OUT_W; ++c) {
#pragma HLS PIPELINE II=1
                float grad = grads[k][r][c];

                for(int i=0; i<IN_C; ++i) {
                    for(int kr=0; kr<K; ++kr) {
                        for(int kc=0; kc<K; ++kc) {
                            dW[k][i][kr][kc] += grad * X[i][r+kr][c+kc];
                        }
                    }
                }
            }
        }
    }
}

// input grads
template<int OUT_C, int IN_C, int K, int H, int W>
void input_grad(
        const float weight[OUT_C][IN_C][K][K],
        const float grads[OUT_C][H-K+1][W-K+1],
        float dX[IN_C][H][W]
        ) {
#pragma HLS INLINE Off
    for(int i=0; i<IN_C; ++i) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
                dX[i][r][c] = 0;
            }
        }
    }

    constexpr int OUT_H = H - K + 1;
    constexpr int OUT_W = W - K + 1;

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<OUT_H; ++r) {
            for(int c=0; c<OUT_W; ++c) {
#pragma HLS PIPELINE II=1
                float grad = grads[k][r][c];

                for(int i=0; i<IN_C; ++i) {
                    for(int kr=0; kr<K; ++kr) {
                        for(int kc=0; kc<K; ++kc) {
                            dX[i][r+kr][c+kc] += grad * weight[k][i][K-kr-1][K-kc-1];
                        }
                    }
                }
            }
        }
    }
}

// backprop for conv2d
template<int OUT_C, int IN_C, int K, int H, int W>
void conv2d_backward(
        const float in_activation[IN_C * H * W],
        const float grads[OUT_C * (H-K+1) * (W-K+1)],
        float out_grads[IN_C * H * W],
        const float weight[OUT_C][IN_C][K][K],
        float dW[OUT_C][IN_C][K][K],
        float dB[OUT_C]
        ) {
#pragma HLS INLINE off

    constexpr int OUT_H = H - K + 1;
    constexpr int OUT_W = W - K + 1;

    // buffer input
    float x_pad[IN_C][H][W];
#pragma HLS ARRAY_PARTITION variable=x_pad complete dim=1
    for(int i=0; i<IN_C; ++i) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
                int idx = i*(H*W) + r*W + c;
                x_pad[i][r][c] = in_activation[idx];
            }
        }
    }
    
    // buffer the grads
    float grad_buffer[OUT_C][OUT_H][OUT_W];
#pragma HLS ARRAY_PARTITION variable=grad_buffer complete dim=1
    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<OUT_H; ++r) {
            for(int c=0; c<OUT_W; ++c) {
                int idx = k*(OUT_H*OUT_W) + r*OUT_W + c;
                grad_buffer[k][r][c] = grads[idx];
            }
        }
    }

    float dX[IN_C][H][W];
    bias_grad<OUT_C, OUT_H, OUT_W>(grad_buffer, dB);
    weight_grad<OUT_C, IN_C, K, H, W>(x_pad, grad_buffer, dW);
    input_grad<OUT_C, IN_C, K, H, W>(weight, grad_buffer, dX);

    for(int i=0; i<IN_C; ++i) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
#pragma HLS PIPELINE II=1
                int idx = i*(H*W) + (r*W) + c;
                out_grads[idx] = dX[i][r][c];
            }
        }
    }
}

#endif
