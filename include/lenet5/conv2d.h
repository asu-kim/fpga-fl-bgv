#ifndef CONV_2D_H
#define CONV_2D_H

#include <hls_stream.h>
#include <hls_math.h>
#include "data_type.hpp"

// typedef ap_int<32> data_t; // 8 bit fixed point as precision

//----------------------
// 5x5 kernel, 6 filters
//----------------------
template<int OUT_C, int IN_C, int KERNEL_SIZE, int ROW, int COL>
void conv2d(
        // data_t in_data[IN_C][ROW][COL],
        data_t in_data[IN_C * ROW * COL],
        data_t out_data[OUT_C * (ROW - KERNEL_SIZE + 1) * (COL - KERNEL_SIZE + 1)],
        // hls::stream<data_t>& in_stream,
        // hls::stream<data_t>& out_stream,
        const data_t weight[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE],
        const data_t bias[OUT_C],
        float act_out_scale=1, int act_out_zp=0
        ) {
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    data_t line_buffer[IN_C][KERNEL_SIZE][COL];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    int cur_row=0;
    for(int r=0; r < ROW; ++r) {
        for(int c=0; c < COL; ++c) {
            #pragma HLS PIPELINE II=1
            for(int ch=0; ch<IN_C; ++ch) {
                #pragma HLS UNROLL
                // data_t pixel = in_stream.read();
                // line_buffer[ch][cur_row][c] = pixel;
                line_buffer[ch][cur_row][c] = in_data[r * COL * IN_C + c * IN_C + ch];
            }
        }

        if(r >= KERNEL_SIZE-1) {
            int row_start = (cur_row+1) % KERNEL_SIZE;

            for(int c = 0; c <= COL-KERNEL_SIZE; ++c) { 
                #pragma HLS PIPELINE II=1
                for(int oc=0; oc < OUT_C; ++oc) {
                    #pragma HLS UNROLL factor=4
                    ap_int<128> sum = bias[oc];

                    for(int ic=0; ic<IN_C; ++ic) {
                        for(int i=0; i<KERNEL_SIZE; ++i) {
                            int row_idx = (row_start+i) % KERNEL_SIZE;
                            for(int j=0; j<KERNEL_SIZE; ++j) {
                                data_t in_val = line_buffer[ic][row_idx][c+j];
                                data_t w_val = weight[oc][ic][i][j];
                                sum += (ap_int<64>)in_val * (ap_int<64>)w_val;
                            }
                        }
                    }
                    
                    // quant output activation
                    float sum_float_val = float(sum);
                    float sum_scaled = sum_float_val * act_out_scale + (float)act_out_zp;
                    float sum_rounded = hls::floor(sum_scaled+0.5f);
                    data_t sum_clipped = (data_t)sum_rounded;
                    sum_clipped = hls::max(MIN_VAL, hls::min(MAX_VAL, sum_clipped));
                    // out_stream.write((data_t)sum_clipped);
                    int out_row = r - (KERNEL_SIZE - 1);
                    int out_col = c;
                    int out_index = oc * (ROW - KERNEL_SIZE + 1) * (COL - KERNEL_SIZE + 1)
                                    + out_row * (COL - KERNEL_SIZE + 1)
                                    + out_col;
                    out_data[out_index] = sum_clipped;
                }
            }
        }
        cur_row = (cur_row+1) % KERNEL_SIZE;
    }
}
#endif