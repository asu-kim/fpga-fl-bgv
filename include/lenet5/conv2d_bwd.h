#ifndef CONV_2D_BWD_H
#define CONV_2D_BWD_H

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
        const float grads[OUT_C*OUT_H*OUT_W],
        float dB[OUT_C]
        ) {
    for(int k=0; k<OUT_C; ++k) dB[k] = 0;

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<OUT_H; ++r) {
            for(int c=0; c<OUT_W; ++c) {
                int idx = k*(OUT_H*OUT_W) + r*(OUT_W) + c; 
                dB[k] += grads[idx];
                // dB[k] += grads[k][r][c];
            }
        }
    }
}

// wieght grads
template<int OUT_C, int IN_C, int K, int H, int W>
void weight_grad(
        const float X[IN_C*H*W],
        const float grads[OUT_C*(H-K+1)*(W-K+1)],
        float dW[OUT_C*IN_C*K*K]
        ) {
    for(int k=0; k<OUT_C; ++k) {
        for(int i=0; i<IN_C; ++i) {
            for(int r=0; r<K; ++r) {
                for(int c=0; c<K; ++c) {
                    int idx = k*(IN_C*K*K) + i*(K*K) + r*(K) + c;
                    dW[idx] = 0;
                    // dW[k][i][r][c] = 0;
                }
            }
        }
    }

    constexpr int OUT_H = H-K+1;
    constexpr int OUT_W = W-K+1;

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<OUT_H; ++r) {
            for(int c=0; c<OUT_W; ++c) {
                int grad_idx = k*(OUT_H*OUT_W) + r*OUT_W + c;
                float grad = grads[grad_idx];
                // float grad = grads[k][r][c];

                for(int i=0; i<IN_C; ++i) {
                    for(int kr=0; kr<K; ++kr) {
                        for(int kc=0; kc<K; ++kc) {
                            int w_idx = k*(IN_C*K*K) + i*(K*K) + kr*(K) + kc;
                            int x_idx = (i)*(H*W) + (r+kr)*(W) + (c+kc);
                            dW[w_idx] += grad * X[x_idx];
                            // dW[k][i][kr][kc] += grad * X[i][r+kr][c+kc];
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
        const float weight[OUT_C*IN_C*K*K],
        const float grads[OUT_C * (H-K+1) * (W-K+1)],
        float dX[IN_C*H*W]
        ) {
    for(int i=0; i<IN_C; ++i) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
                int idx = i*(H*W) + r*(W) + c;
                dX[idx] = 0;
                // dX[i][r][c] = 0;
            }
        }
    }

    constexpr int OUT_H = H - K + 1;
    constexpr int OUT_W = W - K + 1;

    for(int k=0; k<OUT_C; ++k) {
        for(int r=0; r<OUT_H; ++r) {
            for(int c=0; c<OUT_W; ++c) {
                int grad_idx = k*(OUT_H*OUT_W) + r*(OUT_W) + c;
                float grad = grads[grad_idx];
                // float grad = grads[k][r][c];

                for(int i=0; i<IN_C; ++i) {

                    for(int kr=0; kr<K; ++kr) {
                        for(int kc=0; kc<K; ++kc) {
                            int x_idx = i*(H*W) + (r+kr)*(W) + (c+kc);
                            int w_idx = k*(IN_C*K*K) + i*(K*K) + (K-kr-1)*(K) + (K-kc-1);
                            dX[x_idx] += grad * weight[w_idx];
                            // dX[i][r+kr][c+kc] += grad * weight[k][i][K-kr-1][K-kc-1];
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
        const float in_weight[OUT_C*IN_C*K*K],
        float out_grads[IN_C * H * W],
        float dW[OUT_C*IN_C*K*K],
        float dB[OUT_C]
        ) {

    const int OUT_H = H - K + 1;
    const int OUT_W = W - K + 1;
    float local_weight[OUT_C * IN_C * K * K];
    float local_grads[OUT_C * (H-K+1) * (W-K+1)];
    
    copy_loop_weight: for (int i = 0; i < OUT_C * IN_C * K * K; i++) {
        #pragma HLS PIPELINE II=1
        local_weight[i] = in_weight[i];
    }
    copy_loop_grads: for (int i = 0; i < OUT_C * (H-K+1) * (W-K+1); i++) {
        #pragma HLS PIPELINE II=1
        local_grads[i] = grads[i];
    }

    float dX[IN_C * H * W];

    bias_grad<OUT_C, OUT_H, OUT_W>(grads, dB);
    weight_grad<OUT_C, IN_C, K, H, W>(in_activation, local_grads, dW);
    input_grad<OUT_C, IN_C, K, H, W>(local_weight, local_grads, dX);

    for(int i = 0; i < IN_C * H * W; i++) {
        out_grads[i] = dX[i];
    }
}

#endif
