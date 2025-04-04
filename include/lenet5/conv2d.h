#ifndef CONV_5x5_h
#define CONV_5x5_h

#include <hls_stream.h>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t; // 8 bit fixed point as precision

//----------------------
// 5x5 kernel, 6 filters
//----------------------
template<int IN_C, int KERNEL_SIZE>
void conv_5x5_2d(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weights[IN_C][KERNEL_SIZE][KERNEL_SIZE],
        const data_t bias[IN_C],
        int rows, int cols
        );

#endif
