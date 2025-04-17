#ifndef CONV_2D_H
#define CONV_2D_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <math.h>
#include <stdint.h>

typedef ap_int<32> data_t; // 8 bit fixed point as precision

//----------------------
// 5x5 kernel, 6 filters
//----------------------
template<int OUT_C, int IN_C, int KERNEL_SIZE, int MAX_COLS>
void conv2d(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weight[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE],
        const data_t bias[OUT_C],
        int rows, int cols,
        float act_out_scale=1, int act_out_zp=0
        ) {
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    data_t line_buffer[IN_C][KERNEL_SIZE][MAX_COLS];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    int cur_row=0;
    for(int r=0; r < rows+KERNEL_SIZE-1; ++r) {
        for(int c=0; c < cols; ++c) {

            #pragma HLS PIPELINE II=1
            for(int ch=0; ch<IN_C; ++ch) {
                #pragma HLS UNROLL
                data_t pixel;
                if(r < rows) {
                    pixel = in_stream.read();
                } else {
                    pixel = data_t(0);
                }
                line_buffer[ch][cur_row][c] = pixel; // read rows across all input channel
            }
        }

        // compute conv2d
        if(r >= KERNEL_SIZE-1) {
            int row_start = (cur_row+1) % KERNEL_SIZE;

            for(int c = KERNEL_SIZE-1; c<cols; ++c) { 
                #pragma HLS PIPELINE II=1
                for(int oc=0; oc < OUT_C; ++oc) {

                    #pragma HLS UNROLL factor=4
                    ap_int<128> sum = bias[oc];

                    for(int ic=0; ic<IN_C; ++ic) {
                        for(int i=0; i<KERNEL_SIZE; ++i) {
                            int row_idx = (row_start+i) % KERNEL_SIZE;
                            for(int j=0; j<KERNEL_SIZE; ++j) {
                                data_t in_val = line_buffer[ic][row_idx][c-KERNEL_SIZE+1+j];
                                data_t w_val = weight[oc][ic][i][j];
                                sum += (ap_int<64>)in_val * (ap_int<64>)w_val;
                            }
                        }
                    }
                    
                    // quant output activation
                    float sum_float_val = float(sum);
                    float sum_scaled = sum_float_val * act_out_scale + (float)act_out_zp;
                    float sum_rounded = (sum_scaled >= 0.0f) ? (floorf(sum_scaled + 0.5f)) : (floorf(sum_scaled - 0.5f));
                    data_t sat;
                    if (sum_rounded > (float)INT32_MAX) sat = INT32_MAX;
                    else if (sum_rounded < (float)INT32_MIN) sat = INT32_MIN;
                    else sat = (data_t)sum_rounded;

                    out_stream.write(sat);
                }
            }
        }
        cur_row = (cur_row+1) % KERNEL_SIZE;
    }
}

#endif
