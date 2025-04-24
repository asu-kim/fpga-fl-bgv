// #include "hls_stream.h"
#include <hls_math.h>
#include "constants.hpp"
#include "data_type.hpp"
// #include "lenet5/conv2d.h"
#include "lenet5/conv1.h"

int OUT_C = 6;
int IN_C = 1;
int KERNEL_SIZE = 5;
int ROW = 28;
int COL = 28;

extern "C" {
    void conv1(
        // hls::stream<data_t>& in_stream,
        // hls::stream<data_t>& out_stream,
        data_t* in_data,
        data_t* out_data,
        data_t* conv1_weight,
        data_t* conv1_bias
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=3456
        #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem2 depth=256
        #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=128
    
        float act_out_scale=1;
        float act_out_zp=0;

        // Create local copy
        data_t local_weight[6*1*5*5];
        data_t local_bias[6];

        // Copy data
        for(int i=0; i<150; i++) {
            local_weight[i] = conv1_weight[i];
        }
        for(int i=0; i<6; i++) {
            local_bias[i] = conv1_bias[i];
        }
        // data_t line_buffer[IN_C][KERNEL_SIZE][COL];
        data_t line_buffer[IN_C*KERNEL_SIZE*COL];
        // #pragma HLS ARRAY_PARTITION variable=line_buffer cyclic factor=2 dim=1

        int cur_row=0;
        for(int r=0; r < ROW; ++r) {
            for(int c=0; c < COL; ++c) {
                #pragma HLS PIPELINE II=2
                for(int ch=0; ch<IN_C; ++ch) {
                    // line_buffer[ch*(COL*KERNEL_SIZE) + cur_row*COL + c] = in_data[r * COL * IN_C + c * IN_C + ch];
                    line_buffer[ch*(COL*KERNEL_SIZE) + cur_row*COL + c] = in_data[ch * ROW * COL + r * COL + c];
                }
            }

            if(r >= KERNEL_SIZE-1) {
                int row_start = (cur_row+1) % KERNEL_SIZE;

                for(int c = 0; c <= COL-KERNEL_SIZE; ++c) { 
                    #pragma HLS PIPELINE II=3
                    for(int oc=0; oc < OUT_C; ++oc) {
                        // #pragma HLS UNROLL factor=2
                        data_t sum = local_bias[oc];

                        for(int ic=0; ic<IN_C; ++ic) {
                            for(int i=0; i<KERNEL_SIZE; ++i) {
                                int row_idx = (row_start+i) % KERNEL_SIZE;
                                for(int j=0; j<KERNEL_SIZE; ++j) {
                                    data_t in_val = line_buffer[ic*(COL*KERNEL_SIZE) + row_idx*(COL) + (c+j)];
                                    data_t w_val = local_weight[oc*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + ic*(KERNEL_SIZE*KERNEL_SIZE) + i*(KERNEL_SIZE) + j];
                                    // Reduced multiplication bit width
                                    sum += in_val * w_val;
                                }
                            }
                        }
                        
                        // quant output activation
                        float sum_float_val = float(sum);
                        float sum_scaled = sum_float_val * act_out_scale + act_out_zp;
                        float sum_rounded = hls::floor(sum_scaled+0.5f);
                        data_t sum_clipped = (data_t)sum_rounded;
                        sum_clipped = hls::max(MIN_VAL, hls::min(MAX_VAL, sum_clipped));
                        
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
        // conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
        // conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
    }
}