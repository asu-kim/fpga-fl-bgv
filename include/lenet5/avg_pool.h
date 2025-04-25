#ifndef AVG_POOL_H
#define AVG_POOL_H

#include <hls_stream.h>
#include <hls_math.h>
#include <limits>
#include <cmath>

// typedef ap_int<32> data_t;

template<int POOL_SIZE, int CHANNELS, int ROWS, int COLS>
void avg_pool(
        const data_t* in_data,  // Input data array of size CHANNELS*ROWS*COLS
        data_t* out_data        // Output data array of size CHANNELS*(ROWS/POOL_SIZE)*(COLS/POOL_SIZE)
        ) {
    const int pooled_rows = ROWS / POOL_SIZE;
    const int pooled_cols = COLS / POOL_SIZE;
    const int divisor = POOL_SIZE * POOL_SIZE;

    for(int channel=0; channel<CHANNELS; ++channel) {
        data_t line_buffer[ROWS*COLS];
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=0

        // load channel input from pointer
        for(int r=0; r<ROWS; ++r) {
            for(int c=0; c<COLS; ++c) {
                #pragma HLS PIPELINE II=1
                // Access pattern for CHANNELS*ROWS*COLS layout
                // line_buffer[r*COLS + c] = in_data[channel + CHANNELS*(r*COLS + c)];
                line_buffer[r*COLS + c] = in_data[channel*ROWS*COLS + r*COLS + c];
            }
        }

        // compute avg pool
        for(int pr=0; pr<pooled_rows; ++pr) {
            for(int pc=0; pc<pooled_cols; ++pc) {
                #pragma HLS PIPELINE II=1
                data_t sum = 0; // we need a wider bitwidth for accumulation

                for(int i=0; i<POOL_SIZE; ++i) {
                    for(int j=0; j<POOL_SIZE; ++j) {
                        sum += line_buffer[(pr*POOL_SIZE+i)*(COLS)+pc*POOL_SIZE+j];
                    }
                }
                data_t avg = (data_t)((sum + (divisor / 2)) / divisor);
                avg = hls::max(MIN_VAL, hls::min(MAX_VAL, avg));
                // Write to output array with CHANNELS*(ROWS/POOL_SIZE)*(COLS/POOL_SIZE) layout
                // out_data[channel + CHANNELS*(pr*pooled_cols + pc)] = avg;
                out_data[channel*pooled_rows*pooled_cols + pr*pooled_cols + pc] = avg;
            }
        }
    }
}

#endif
