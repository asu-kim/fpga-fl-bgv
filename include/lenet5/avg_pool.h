#ifndef AVG_POOL_H
#define AVG_POOL_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <limits>
#include <cmath>

typedef ap_int<32> data_t;

template<int POOL_SIZE>
void avg_pool(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        int rows,
        int cols, 
        int channels
        ) {
    const int POOLED_ROWS = rows / POOL_SIZE;
    const int POOLED_COLS = cols / POOL_SIZE;
    const int divisor = POOL_SIZE * POOL_SIZE;

    for(int channel=0; channel<channels; ++channel) {
        data_t line_buffer[rows][cols];
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=0

        // load channel input
        for(int r=0; r<rows; ++r) {
            for(int c=0; c<cols; ++c) {
                #pragma HLS PIPELINE II=1
                line_buffer[r][c] = in_stream.read();
            }
        }

        // compute avg pool
        for(int pr=0; pr<POOLED_ROWS; ++pr) {
            for(int pc=0; pc<POOLED_COLS; ++pc) {
                #pragma HLS PIPELINE II=1
                ap_int<64> sum = 0; // we need a wider bitwidth for accumulation

                for(int i=0; i<POOL_SIZE; ++i) {
                    for(int j=0; j<POOL_SIZE; ++j) {
                        sum += line_buffer[pr*POOL_SIZE+i][pc*POOL_SIZE+j];
                    }
                }
                int32_t avg = (int32_t)((sum + (divisor / 2)) / divisor);
                avg = std::max(std::numeric_limits<int32_t>::min(), std::min(std::numeric_limits<int32_t>::max(), avg));
                out_stream.write((data_t)avg);
            }
        }
    }
}

#endif
