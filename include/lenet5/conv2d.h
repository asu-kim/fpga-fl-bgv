#ifndef CONV_2D_H
#define CONV_2D_H

#include <hls_stream.h>
#include <hls_math.h>

template<int OUT_C, int IN_C, int KERNEL_SIZE, int ROW, int COL>
void conv2d(
        data_ap_fixed_t in_data[IN_C * ROW * COL],
        data_ap_fixed_t out_data[OUT_C * (ROW - KERNEL_SIZE + 1) * (COL - KERNEL_SIZE + 1)],
        const data_ap_fixed_t weight[OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE],
        const data_ap_fixed_t bias[OUT_C]
) {
    #pragma HLS INLINE
    // Constants
    const int OUT_H = ROW - KERNEL_SIZE + 1;
    const int OUT_W = COL - KERNEL_SIZE + 1;
    
    // Initialize output array with bias values
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
                #pragma HLS PIPELINE II=1
                out_data[oc*OUT_H*OUT_W + oh*OUT_W + ow] = bias[oc];
            }
        }
    }

    // Add partitioning for weights for parallel access
    #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=KERNEL_SIZE dim=3
    #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=KERNEL_SIZE dim=4

    // #pragma HLS ARRAY_PARTITION variable=in_data cyclic factor=4 dim=1
    // #pragma HLS ARRAY_PARTITION variable=out_data cyclic factor=4 dim=1
    // #pragma HLS ARRAY_PARTITION variable=bias complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=4 dim=1
    // #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=4 dim=2


    // Main computation with same loop pattern as original
    for (int oc = 0; oc < OUT_C; oc++) {
        // #pragma HLS UNROLL factor=complete
        #pragma HLS UNROLL factor=2

        for (int ic = 0; ic < IN_C; ic++) {
            // Create line buffer for current input channel
            data_ap_fixed_t line_buffer[KERNEL_SIZE][COL];
            #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
            #pragma HLS ARRAY_PARTITION variable=line_buffer cyclic factor=KERNEL_SIZE dim=2
            
            // Initialize buffer - same as original
            for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                for (int c = 0; c < COL; c++) {
                    line_buffer[kr][c] = 0;
                }
            }
            
            int buf_row = 0;
            
            for (int r = 0; r < ROW; r++) {
                // Current buffer row index - same as original
                int current_buf_row = buf_row;
                
                // Load current row into line buffer - keep original behavior
                for (int c = 0; c < COL; c++) {
                    #pragma HLS PIPELINE II=1
                    line_buffer[current_buf_row][c] = in_data[ic*ROW*COL + r*COL + c];
                }
                
                // Process output if we have enough rows - exactly as original
                if (r >= KERNEL_SIZE-1) {
                    for (int c = 0; c <= COL-KERNEL_SIZE; c++) {
                        #pragma HLS PIPELINE II=1

                        // Pre-compute kernel row indices - same as original
                        int kr_indices[KERNEL_SIZE];
                        #pragma HLS ARRAY_PARTITION variable=kr_indices complete dim=1

                        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                            kr_indices[kr] = (current_buf_row + 1 + kr) % KERNEL_SIZE;
                        }
                        
                        // Compute partial convolution for current input channel
                        int out_row = r - (KERNEL_SIZE - 1);
                        data_ap_fixed_t partial_sum = 0;
                        
                        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                            #pragma HLS UNROLL
                            for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                                #pragma HLS UNROLL
                                data_ap_fixed_t in_val = line_buffer[kr_indices[kr]][c+kc];
                                data_ap_fixed_t w_val = weight[oc*IN_C*KERNEL_SIZE*KERNEL_SIZE + 
                                                     ic*KERNEL_SIZE*KERNEL_SIZE + 
                                                     kr*KERNEL_SIZE + kc];
                                partial_sum += in_val * w_val;
                            }
                        }
                        
                        // Accumulate partial sum into output
                        out_data[oc*OUT_H*OUT_W + out_row*OUT_W + c] += partial_sum;
                    }
                }
                
                // Update buffer row index
                buf_row = (buf_row + 1) % KERNEL_SIZE;
            }
        }
    }
}
#endif