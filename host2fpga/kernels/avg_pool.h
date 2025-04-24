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

    for(int channel=0; channel<IN_C; ++channel) {
        data_t line_buffer[IN_HEIGHT][IN_WIDTH];
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=0

        // load channel input
        for(int r=0; r<IN_HEIGHT; ++r) {
            for(int c=0; c<IN_WIDTH; ++c) {
                #pragma HLS PIPELINE II=1
                line_buffer[r][c] = in_stream.read();
            }
        }

        // compute avg pool
        for(int pr=0; pr<POOLED_ROWS; pr+=STRIDE) {
            for(int pc=0; pc<POOLED_COLS; pc+=STRIDE) {
                #pragma HLS PIPELINE II=1
                data_t sum = 0; // we need a wider bitwidth for accumulation

                for(int i=0; i<POOL_SIZE; ++i) {
                    for(int j=0; j<POOL_SIZE; ++j) {
                        sum += line_buffer[pr*POOL_SIZE+i][pc*POOL_SIZE+j];
                    }
                }
                float avg = (((float)sum + ((float)divisor / 2)) / (float)divisor);
                float avg_rounded = (avg >= 0.0f) ? (floorf(avg + 0.5f)) : (floorf(avg - 0.5f));
                data_t sat;
                if (avg_rounded > (float)INT32_MAX) sat = INT32_MAX;
                else if (avg_rounded < (float)INT32_MIN) sat = INT32_MIN;
                else sat = (data_t)avg_rounded;

                out_stream.write((data_t)sat);
            }
        }
    }
}

#endif
