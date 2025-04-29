#ifndef AVG_POOL_BWD_H
#define AVG_POOL_BWD_H

#include <hls_stream.h>
#include <stdint.h>

template<int P, int S, int H, int W, int C>
void avg_pool_backward(
        const float grads[C*((H-P)/S+1)*((W-P)/S+1)],
        float dX[C*H*W]
        ) {
#pragma HLS INLINE off

    constexpr int PH = (H-P)/S+1;
    constexpr int PW = (W-P)/S+1;
    const float inverse = 1.0f / (float)(P*P);

    // reconstruct grads
    float grads_buffer[C][PH][PW];
#pragma HLS ARRAY_PARTITION variable=grads_buffer complete dim=1
    for(int ch=0; ch<C; ++ch) {
        for(int r=0; r<PH; ++r) {
            for(int c=0; c<PW; ++c) {
                int idx = ch*(PH*PW) + r*PW + c;
                grads_buffer[ch][r][c] = grads[idx];
            }
        }
    }

    float x_buffer[C][H][W];
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
    for(int ch=0; ch<C; ++ch) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
                x_buffer[ch][r][c] = 0;
            }
        }
    }

    for(int k=0; k<C; ++k) {
        for(int r=0; r<PH; ++r) {
            for(int c=0; c<PW; ++c) {
                float grad = grads_buffer[k][r][c] * inverse;

                for(int pr=0; pr<P; ++pr) {
                    for(int pc=0; pc<P; ++pc) {
#pragma HLS PIPELINE II=1
                        x_buffer[k][r*S+pr][c*S+pc] += grad;
                    } 
                }
            }
        }
    }

    for(int k=0; k<C; ++k) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
#pragma HLS PIPELINE II=1
                int idx = k * (H*W) + r*W + c;
                dX[idx] = x_buffer[k][r][c];
            }
        }
    }

}

#endif
