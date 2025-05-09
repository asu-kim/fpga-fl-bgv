#ifndef CONV_2D_H
#define CONV_2D_H

#include <hls_stream.h>
#include <hls_math.h>

template<int OUT_C, int IN_C, int KERNEL_SIZE, int ROW, int COL>
void conv2d(
    data_ap_fixed_t in_data[IN_C][ROW][COL],
    data_ap_fixed_t out_data[OUT_C][ROW - KERNEL_SIZE + 1][COL - KERNEL_SIZE + 1],
    const data_ap_fixed_t weight[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE],
    const data_ap_fixed_t bias[OUT_C]
) {
    #pragma HLS ARRAY_PARTITION variable=in_data cyclic factor=2 dim=0
    #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=2 dim=0
    #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=3 dim=2
    #pragma HLS ARRAY_PARTITION variable=bias complete
    #pragma HLS ARRAY_PARTITION variable=out_data cyclic factor=4 dim=0

    const int OUT_H = ROW - KERNEL_SIZE + 1;
    const int OUT_W = COL - KERNEL_SIZE + 1;

    // Initialize output with bias
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
                #pragma HLS PIPELINE II=1
                out_data[oc][oh][ow] = bias[oc];
            }
        }
    }

    // Convolution loop
    for (int ic = 0; ic < IN_C; ic++) {
        data_ap_fixed_t line_buffer[KERNEL_SIZE][COL];
        #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

        // Clear line buffer
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < COL; j++) {
                line_buffer[i][j] = 0;
            }
        }

        for (int r = 0; r < ROW; r++) {
            // Shift line buffer up
            for (int i = KERNEL_SIZE - 1; i > 0; i--) {
                for (int j = 0; j < COL; j++) {
                    #pragma HLS UNROLL
                    line_buffer[i][j] = line_buffer[i - 1][j];
                }
            }

            // Insert new row
            for (int j = 0; j < COL; j++) {
                #pragma HLS PIPELINE II=1
                line_buffer[0][j] = in_data[ic][r][j];
            }

            // Apply convolution if enough rows are filled
            if (r >= KERNEL_SIZE - 1) {
                for (int c = 0; c <= COL - KERNEL_SIZE; c++) {
                    #pragma HLS PIPELINE
                    for (int oc = 0; oc < OUT_C; oc++) {
                        #pragma HLS UNROLL factor=2
                        data_ap_fixed_t acc = 0;
                        for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                            for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                                // #pragma HLS UNROLL
                                data_ap_fixed_t val = line_buffer[kr][c + kc];
                                data_ap_fixed_t w = weight[oc][ic][kr][kc];
                                acc += val * w;
                            }
                        }
                        out_data[oc][r - KERNEL_SIZE + 1][c] += acc;
                    }
                }
            }
        }
    }
}

#endif
