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
template<int OUT_C, int H, int W>
void bias_grad(
        hls::stream<float> &grads,
        float dB[OUT_C]
        ) {
#pragma HLS INLINE
    for(int k=0; k<OUT_C; ++k) dB[k] = 0;

    for(int k=0; k<OUT_C; ++k) {
        for(int i=0; i<H * W; ++i) {
#pragma HLS PIPELINE II=1
            dB[k] += grads.read();
        }
    }
}

// wieght grads
template<int OUT_C, int IN_C, int K, int H, int W>
void weight_grad(
        float X[IN_C][K+H-1][K+W-1],
        hls::stream<float> &grads,
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

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
#pragma HLS PIPELINE II=1
                float grad = grads.read();

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
        hls::stream<float> &grads,
        float dX[IN_C][H+K-1][W+K-1]
        ) {
#pragma HLS INLINE Off
    for(int i=0; i<IN_C; ++i) {
        for(int r=0; r<H+K-1; ++r) {
            for(int c=0; c<W+K-1; ++c) {
                dX[i][r][c] = 0;
            }
        }
    }

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
#pragma HLS PIPELINE II=1
                float grad = grads.read();

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
        hls::stream<float> &in_stream, /* original activation */
        hls::stream<float> &grads, /* gradient pased from next layer (passing grads backward) */
        hls::stream<float> &out_stream, /* resulting gradients to be passed to prev layer */
        const float weight[OUT_C][IN_C][K][K],
        float dW[OUT_C][IN_C][K][K],
        float dB[OUT_C]
        ) {
#pragma HLS INLINE off
    static float x_pad[IN_C][K+H-1][K+W-1];
    static float dX[IN_C][K+H-1][K+W-1]; /* prepare input gradient */
#pragma HLS ARRAY_PARTITION variable=x_pad complete dim=1
#pragma HLS ARRAY_PARTITION variable=dX complete dim=1

    for(int i=0; i<IN_C; ++i) {
        for(int r=0; r<H+K-1; ++r) {
            for(int c=0; c<W+K-1; ++c) {
#pragma HLS PIPELINE II=1
                x_pad[i][r][c] = in_stream.read();
            }
        }
    }

    // buffer the grads
    static float grad_buffer[OUT_C][H][W];
#pragma HLS ARRAY_PARTITION variable=grad_buffer complete dim=1
    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
                grad_buffer[k][r][c] = grads.read();
            }
        }
    }

    // streams for bias, weight and input
    hls::stream<float> grads_bias("db"), grads_weight("dw"), grads_input("dx");
#pragma HLS STREAM variable=grads_bias depth=1024
#pragma HLS STREAM variable=grads_weight depth=1024
#pragma HLS STREAM variable=grads_input depth=1024

    // stream in from grads
    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
                float val = grad_buffer[k][r][c];
                grads_bias.write(val);
                grads_weight.write(val);
                grads_input.write(val);
            }
        }
    }

#pragma HLS DATAFLOW
    bias_grad<OUT_C, H, W>(grads_bias, dB);
    weight_grad<OUT_C, IN_C, K, H, W>(x_pad, grads_weight, dW);
    input_grad<OUT_C, IN_C, K, H, W>(weight, grads_input, dX);

    for(int i=0; i<IN_C; ++i) {
        for(int r=0; r<K+H-1; ++r) {
            for(int c=0; c<K+W-1; ++c) {
#pragma HLS PIPELINE II=1
                out_stream.write(dX[i][r][c]);
            }
        }
    }
}

#endif
