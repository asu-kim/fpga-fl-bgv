#ifndef AVG_POOL_BWD_H
#define AVG_POOL_BWD_H

#include <hls_stream.h>
#include <stdint.h>

template<int POOL_SIZE, int STRIDE, int H, int W, int C>
void avg_pool_backward(
        hls::stream<float> &grads,
        hls::stream<float> &out_stream
        ) {
    const int PH = H / POOL_SIZE;
    const int PW = W / POOL_SIZE;
    const float inverse = 1.0 / (POOL_SIZE * POOL_SIZE);

    static float grads_buffer[C][PH][PW];
#pragma HLS ARRAY_PARTITION variable=grads_buffer complete dim=1
    for(int r=0; r<PH; ++r) {
        for(int c=0; c<PW; ++c) {
            for(int ch=0; ch<C; ++ch) {
#pragma HLS PIPELINE II=1
                grads_buffer[ch][r][c] = grads.read();
            }
        }
    }

    static float x_buffer[C][H][W];
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
    for(int ch=0; ch<C; ++ch) {
        for(int r=0; r<H; ++r) {
            for(int c=0; c<W; ++c) {
                x_buffer[ch][r][c] = 0;
            }
        }
    }

    for(int r=0; r<PH; ++r) {
        for(int c=0; c<PW; ++c) {
            for(int ch=0; ch<C; ++ch) {
                float val = grads_buffer[ch][r][c] * inverse;
                for(int i=0; i<POOL_SIZE; ++i) {
                    for(int j=0; j<POOL_SIZE; ++j) {
#pragma HLS PIPELINE II=1
                        x_buffer[ch][r*POOL_SIZE + i][c*POOL_SIZE + j] += val;
                    }
                }
            }
        }
    }


    for(int r=0; r<H; ++r) {
        for(int c=0; c<W; ++c) {
            for(int ch=0; ch<C; ++ch) {
#pragma HLS PIPELINE II=1
               out_stream.write(x_buffer[ch][r][c]); 
            }
        }
    }


}

#endif
