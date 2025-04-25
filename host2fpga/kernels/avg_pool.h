#ifndef AVG_POOL_H
#define AVG_POOL_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <math.h>

typedef ap_int<32> data_t; // 8 bit fixed point as precision

//---------------------------
// 2x2 average pool, stride=2
//---------------------------
template<int POOL_SIZE, int STRIDE, int IN_HEIGHT, int IN_WIDTH, int IN_C>
void avg_pool(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream
        ) {
    const int POOLED_ROWS = IN_HEIGHT / POOL_SIZE;
    const int POOLED_COLS = IN_WIDTH / POOL_SIZE;
    const int divisor = POOL_SIZE * POOL_SIZE;

    static data_t plane[IN_C][IN_HEIGHT][IN_WIDTH]; // scartch pad
#pragma HLS ARRAY_PARTITION variable=plane complete dim=1

    for(int r=0; r<IN_HEIGHT; ++r) {
        for(int ch=0; ch<IN_C; ++ch) {
            for(int c=0; c<IN_WIDTH; ++c) {
#pragma HLS PIPELINE II=1
                plane[ch][r][c] = in_stream.read();
            }
        }
    }

    for(int ch=0; ch<IN_C; ++ch) {
        for(int pr=0; pr<POOLED_ROWS; pr += STRIDE) {
            for(int pc=0; pc<POOLED_COLS; pc += STRIDE) {
                ap_int<64> sum = 0;

                for(int i=0; i<POOL_SIZE; ++i) {
                    for(int j=0; j<POOL_SIZE; ++j) {
                        sum += plane[ch][pr*POOL_SIZE + i][pc*POOL_SIZE + j];
                    }
                }
                float sum_val = (float)sum / divisor;
                float sum_rounded = (sum_val >= 0.0f) ? floorf(sum_val + 0.5f) : floorf(sum_val - 0.5f);
                data_t sat = (sum_rounded > INT32_MAX) ? (data_t)INT32_MAX : (sum_rounded < INT32_MIN) ? (data_t)INT32_MIN : (data_t)sum_rounded;
                out_stream.write(sat);
            }
        }
    }
}

#endif
