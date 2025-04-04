#include "conv_5x5.h"

template<int IN_C, int KERNEL_SIZE>
void conv_5x5_2d(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weights[IN_C][KERNEL_SIZE][KERNEL_SIZE],
        const data_t bias[IN_C],
        int rows, int cols
        ) {
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    data_t line_buffer[KERNEL_SIZE-1][cols];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    for(int r=0; r < rows+KERNEL_SIZE-1; ++r) {
        for(int c=0; c < cols; ++c) {
            #pragma HLS PIPELINE II=1
            data_t pixel = (r < rows) ? in_stream.read() : data_t(0);

            // move pixels to buffer
            if(r < rows) {
                for(int i=KERNEL_SIZE-1; i>0; --i) {
                    line_buffer[i][c] = line_buffer[i-1][c];
                }
                line_buffer[0][c] = pixel;
            }

            if(r >= KERNEL_SIZE-1 && c >= KERNEL_SIZE-1) {
                for(int channel=0; channel < IN_C; ++channel) {
                    #pragma HLS UNROLL factor=4
                    data_t sum = bias[ch];

                    for(int i=0; i<KERNEL_SIZE; ++i) {
                        for(int j=0; j<KERNEL_SIZE; ++j) {
                            sum += line_buffer[i][c-KERNEL_SIZE+j] * weights[channel][i][j];
                        }
                    }

                    out_stream.write((sum>0) ? sum : data_t(0));
                }
            }
        }
    }
}
