// #ifndef CONV_2D_H
// #define CONV_2D_H

// #include <hls_stream.h>
// #include <hls_math.h>
// #include "data_type.hpp"

// // typedef ap_int<32> data_t; // 8 bit fixed point as precision

// //----------------------
// // 5x5 kernel, 6 filters
// //----------------------
// template<int OUT_C, int IN_C, int KERNEL_SIZE, int ROW, int COL>
// void conv2d(
//         // data_t in_data[IN_C][ROW][COL],
//         data_t in_data[IN_C * ROW * COL],
//         data_t out_data[OUT_C * (ROW - KERNEL_SIZE + 1) * (COL - KERNEL_SIZE + 1)],
//         // hls::stream<data_t>& in_stream,
//         // hls::stream<data_t>& out_stream,
//         const data_t weight[OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE],
//         const data_t bias[OUT_C],
//         float act_out_scale=1, int act_out_zp=0
//         ) {
//     #pragma HLS INLINE OFF
//     // #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=2 dim=1
//     // #pragma HLS ARRAY_PARTITION variable=bias cyclic factor=2 dim=1
//     #pragma HLS ARRAY_PARTITION variable=weight complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=bias   complete dim=1

//     // data_t line_buffer[IN_C][KERNEL_SIZE][COL];
//     data_t line_buffer[IN_C*KERNEL_SIZE*COL];
//     // #pragma HLS ARRAY_PARTITION variable=line_buffer cyclic factor=2 dim=1
//     #pragma HLS ARRAY_PARTITION variable=line_buffer cyclic factor=IN_C dim=1

//     int cur_row=0;
//     for(int r=0; r < ROW; ++r) {
//         #pragma HLS PIPELINE II=2
//         for(int c=0; c < COL; ++c) {
//             for(int ch=0; ch<IN_C; ++ch) {
//                 // line_buffer[ch*(COL*KERNEL_SIZE) + cur_row*COL + c] = in_data[r * COL * IN_C + c * IN_C + ch];
//                 line_buffer[ch*(COL*KERNEL_SIZE) + cur_row*COL + c] = in_data[ch * COL * ROW + r * COL + c];
//             }
//         }

//         if(r >= KERNEL_SIZE-1) {
//             int row_start = (cur_row+1) % KERNEL_SIZE;

//             #pragma HLS PIPELINE II=4
//             for(int c = 0; c <= COL-KERNEL_SIZE; ++c) { 
//                 #pragma HLS UNROLL
//                 for(int oc=0; oc < OUT_C; ++oc) {
//                     data_t sum = bias[oc];

//                     for(int ic=0; ic<IN_C; ++ic) {
//                         for(int i=0; i<KERNEL_SIZE; ++i) {
//                             int row_idx = (row_start+i) % KERNEL_SIZE;
//                             for(int j=0; j<KERNEL_SIZE; ++j) {
//                                 data_t in_val = line_buffer[ic*(COL*KERNEL_SIZE) + row_idx*(COL) + (c+j)];
//                                 data_t w_val = weight[oc*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + ic*(KERNEL_SIZE*KERNEL_SIZE) + i*(KERNEL_SIZE) + j];
//                                 // Reduced multiplication bit width
//                                 sum += in_val * w_val;
//                             }
//                         }
//                     }
                    
//                     // quant output activation
//                     float sum_float_val = float(sum);
//                     float sum_scaled = sum_float_val * act_out_scale + (float)act_out_zp;
//                     float sum_rounded = hls::floor(sum_scaled+0.5f);
//                     data_t sum_clipped = (data_t)sum_rounded;
//                     sum_clipped = hls::max(MIN_VAL, hls::min(MAX_VAL, sum_clipped));
                    
//                     int out_row = r - (KERNEL_SIZE - 1);
//                     int out_col = c;
//                     int out_index = oc * (ROW - KERNEL_SIZE + 1) * (COL - KERNEL_SIZE + 1)
//                                     + out_row * (COL - KERNEL_SIZE + 1)
//                                     + out_col;
//                     out_data[out_index] = sum_clipped;
//                 }
//             }
//         }
//         cur_row = (cur_row+1) % KERNEL_SIZE;
//     }
// }
// #endif

#ifndef CONV_2D_H
#define CONV_2D_H

#include <hls_stream.h>
#include <hls_math.h>
#include "data_type.hpp"

template<int OUT_C, int IN_C, int KERNEL_SIZE, int ROW, int COL>
void conv2d(
        data_t in_data[IN_C * ROW * COL],
        data_t out_data[OUT_C * (ROW - KERNEL_SIZE + 1) * (COL - KERNEL_SIZE + 1)],
        const data_t weight[OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE],
        const data_t bias[OUT_C],
        float act_out_scale=1, int act_out_zp=0
        ) {
    #pragma HLS INLINE OFF
    
    // Constants
    const int OUT_H = ROW - KERNEL_SIZE + 1;
    const int OUT_W = COL - KERNEL_SIZE + 1;
    
    // Process each output channel sequentially to avoid resource conflicts
    for (int oc = 0; oc < OUT_C; oc++) {
        // Create line buffer with better partitioning
        data_t line_buffer[IN_C][KERNEL_SIZE][COL];
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
        
        // Initialize buffer
        for (int ic = 0; ic < IN_C; ic++) {
            for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                for (int c = 0; c < COL; c++) {
                    line_buffer[ic][kr][c] = 0;
                }
            }
        }
        
        int buf_row = 0;
        
        // Process each row of input
        for (int r = 0; r < ROW; r++) {
            // Store current line buffer row index before pipeline to avoid carried dependency
            int current_buf_row = buf_row;
            
            // Load current row into line buffer - without dependency on buf_row update
            for (int c = 0; c < COL; c++) {
                #pragma HLS PIPELINE II=1
                for (int ic = 0; ic < IN_C; ic++) {
                    line_buffer[ic][current_buf_row][c] = in_data[ic*ROW*COL + r*COL + c];
                }
            }
            
            // Process output if we have enough rows
            if (r >= KERNEL_SIZE-1) {
                for (int c = 0; c <= COL-KERNEL_SIZE; c++) {
                    // Pre-compute kernel row indices to avoid repeated modulo operations
                    int kr_indices[KERNEL_SIZE];
                    for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                        kr_indices[kr] = (current_buf_row + KERNEL_SIZE - kr) % KERNEL_SIZE;
                    }
                    
                    // Initialize accumulation
                    data_t sum = bias[oc];
                    
                    // Compute convolution at this position
                    for (int ic = 0; ic < IN_C; ic++) {
                        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                            for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                                #pragma HLS PIPELINE II=1
                                data_t in_val = line_buffer[ic][kr_indices[kr]][c+kc];
                                data_t w_val = weight[oc*IN_C*KERNEL_SIZE*KERNEL_SIZE + 
                                                     ic*KERNEL_SIZE*KERNEL_SIZE + 
                                                     kr*KERNEL_SIZE + kc];
                                sum += in_val * w_val;
                            }
                        }
                    }
                    
                    // Quantize output
                    float sum_float = float(sum);
                    float sum_scaled = sum_float * act_out_scale + (float)act_out_zp;
                    float sum_rounded = hls::floor(sum_scaled + 0.5f);
                    data_t sum_quant = (data_t)sum_rounded;
                    sum_quant = hls::max(MIN_VAL, hls::min(MAX_VAL, sum_quant));
                    
                    // Store output
                    int out_row = r - (KERNEL_SIZE - 1);
                    out_data[oc*OUT_H*OUT_W + out_row*OUT_W + c] = sum_quant;
                }
            }
            
            // Update buffer row index AFTER all operations are complete
            buf_row = (buf_row + 1) % KERNEL_SIZE;
        }
    }
}
#endif