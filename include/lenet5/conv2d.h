#ifndef CONV_2D_H
#define CONV_2D_H

#include <hls_stream.h>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t; // 8 bit fixed point as precision

//----------------------
// 5x5 kernel, 6 filters
//----------------------
template<int IN_C, int KERNEL_SIZE>
void conv2d(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weight[IN_C][KERNEL_SIZE][KERNEL_SIZE],
        const data_t bias[IN_C],
        int rows, int cols
        ) {
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    data_t line_buffer[KERNEL_SIZE][cols];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    for(int r=0; r < rows+KERNEL_SIZE-1; ++r) {
        for(int c=0; c < cols; ++c) {
            #pragma HLS PIPELINE II=1

            // read if inbound
            data_t pixel = (r < rows) ? in_stream.read() : data_t(0);

            // shift col downward
            for(int nr=KERNEL_SIZE-1; nr>0; --nr) {
                line_buffer[nr][c] = line_buffer[nr-1][c];
            }
            line_buffer[0][c] = pixel;

            // compute conv2d
            if(r >= KERNEL_SIZE-1 && c >= KERNEL_SIZE-1) {
                for(int channel=0; channel < IN_C; ++channel) {
                    #pragma HLS UNROLL factor=4
                    data_t sum = bias[channel];

                    for(int i=0; i<KERNEL_SIZE; ++i) {
                        for(int j=0; j<KERNEL_SIZE; ++j) {
                            sum += line_buffer[i][c-KERNEL_SIZE+1+j] * weight[channel][i][j];
                        }
                    }

                    out_stream.write((sum>0) ? sum : data_t(0));
                }
            }
        }
    }
}

#endif
