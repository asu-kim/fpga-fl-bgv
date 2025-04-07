#ifndef AVG_POOL_H
#define AVG_POOL_H

#include <hls_stream.h>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t;

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
                ap_fixed<16, 8> sum = 0; // we need a wider bitwidth for accumulation

                for(int i=0; i<POOL_SIZE; ++i) {
                    for(int j=0; j<POOL_SIZE; ++j) {
                        sum += line_buffer[pr*POOL_SIZE+i][pc*POOL_SIZE+j];
                    }
                }

                ap_fixed<16, 8> result(sum / (POOL_SIZE * POOL_SIZE));
                data_t cast_result = static_cast<data_t>(result);
                out_stream.write(cast_result);
            }
        }
    }
}

#endif
