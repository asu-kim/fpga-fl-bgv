#ifndef CONV_2D_H
#define CONV_2D_H

#include <hls_stream.h>
#include <hls_math.h>

template<int OUT_C, int IN_C, int KERNEL_SIZE, int ROW, int COL>
void conv2d(
        float in_data[IN_C * ROW * COL],
        float out_data[OUT_C * (ROW - KERNEL_SIZE + 1) * (COL - KERNEL_SIZE + 1)],
        const float weight[OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE],
        const float bias[OUT_C]
        ) {
    // #pragma HLS INLINE OFF
    
    // Constants
    const int OUT_H = ROW - KERNEL_SIZE + 1;
    const int OUT_W = COL - KERNEL_SIZE + 1;
    
    // Initialize output array with bias values
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
                out_data[oc*OUT_H*OUT_W + oh*OUT_W + ow] = bias[oc];
            }
        }
    }
    
    // Main computation with reordered loop pattern: out_channel -> in_channel -> row -> col
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int ic = 0; ic < IN_C; ic++) {
            // Create line buffer for current input channel only
            float line_buffer[KERNEL_SIZE][COL];
            #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
            
            // Initialize buffer
            for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                for (int c = 0; c < COL; c++) {
                    line_buffer[kr][c] = 0;
                }
            }
            
            int buf_row = 0;
            
            for (int r = 0; r < ROW; r++) {
                // Current buffer row index
                int current_buf_row = buf_row;
                
                // Load current row into line buffer
                for (int c = 0; c < COL; c++) {
                    #pragma HLS PIPELINE II=1
                    line_buffer[current_buf_row][c] = in_data[ic*ROW*COL + r*COL + c];
                }
                
                // Process output if we have enough rows
                if (r >= KERNEL_SIZE-1) {
                    for (int c = 0; c <= COL-KERNEL_SIZE; c++) {
                        // Pre-compute kernel row indices
                        int kr_indices[KERNEL_SIZE];
                        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                            kr_indices[kr] = (current_buf_row + 1 + kr) % KERNEL_SIZE;
                        }
                        
                        // Compute partial convolution for current input channel
                        int out_row = r - (KERNEL_SIZE - 1);
                        float partial_sum = 0;
                        
                        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                            for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                                #pragma HLS PIPELINE II=1
                                float in_val = line_buffer[kr_indices[kr]][c+kc];
                                float w_val = weight[oc*IN_C*KERNEL_SIZE*KERNEL_SIZE + 
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