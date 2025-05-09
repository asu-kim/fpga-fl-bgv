#ifndef AVG_POOL_H
#define AVG_POOL_H

#include <hls_stream.h>
#include <hls_math.h>
#include <limits>
#include <cmath>

template<int POOL_SIZE, int STRIDE, int IN_C, int IN_ROWS, int IN_COLS>
void avg_pool(
        const data_ap_fixed_t in_data[IN_C][IN_ROWS][IN_COLS],
        data_ap_fixed_t out_data[IN_C][(IN_ROWS - POOL_SIZE) / STRIDE + 1][(IN_COLS - POOL_SIZE) / STRIDE + 1]
        ) {
    const int OUT_ROWS = (IN_ROWS - POOL_SIZE) / STRIDE + 1;
    const int OUT_COLS = (IN_COLS - POOL_SIZE) / STRIDE + 1;
    const data_ap_fixed_t pool_area = POOL_SIZE * POOL_SIZE;

    static data_ap_fixed_t plane[IN_ROWS][IN_COLS];
    #pragma HLS ARRAY_PARTITION variable=plane cyclic factor=4 dim=1

    for (int channel = 0; channel < IN_C; channel++) {
        // Load input channel into local buffer
        for (int r = 0; r < IN_ROWS; ++r) {
            for (int c = 0; c < IN_COLS; ++c) {
                #pragma HLS PIPELINE II=1
                plane[r][c] = in_data[channel][r][c];
            }
        }

        // Apply average pooling
        for (int out_r = 0; out_r < OUT_ROWS; ++out_r) {
            for (int out_c = 0; out_c < OUT_COLS; ++out_c) {
                #pragma HLS PIPELINE II=1
                data_ap_fixed_t sum = 0.0f;

                for (int i = 0; i < POOL_SIZE; ++i) {
                    for (int j = 0; j < POOL_SIZE; ++j) {
                        int r = out_r * STRIDE + i;
                        int c = out_c * STRIDE + j;
                        sum += plane[r][c];
                    }
                }

                data_ap_fixed_t avg = sum / pool_area;
                out_data[channel][out_r][out_c] = avg;
            }
        }
    }
}

#endif