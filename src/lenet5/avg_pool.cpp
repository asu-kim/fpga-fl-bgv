#include "avg_pool.h"

template<int POOL_SIZE>
void avg_pool(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        int rows,
        int cols, 
        int channels) {
    const int POOLED_ROWS = rows / POOL_SIZE;
    const int POOLED_COLS = cols / POOL_SIZE;

    for(int channel=0; channel<channels; ++channel) {
        for(int r=0; r<POOLED_ROWS; ++r) {
            for(int c=0; c<POOLED_COLS; ++c) {
                #pragma HLS PIPELINE II=1
                ap_ufixed<16, 8> sum = 0; // we need a wider bitwidth for accumulation

                for(int i=0; i<POOL_SIZE; ++i) {
                    for(int j=0; j<POOL_SIZE; ++j) {
                        sum += in_stream.read();
                    }
                }

                ap_ufixed<16, 8> result(sum / (POOL_SIZE * POOL_SIZE));
                out_stream.write(result);
            }
        }
    }
}
