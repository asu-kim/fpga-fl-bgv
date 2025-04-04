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
        );

#endif
