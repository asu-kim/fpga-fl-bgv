#ifndef AVG_POOL_H
#define AVG_POOL_H

#include <hls_stream.h>
#include <hls_math.h>
#include <limits>
#include <cmath>

template<int POOL_SIZE, int STRIDE, int IN_C, int IN_ROWS, int IN_COLS>
void avg_pool(
        const float* in_data,  // Input data array of size IN_C*IN_ROWS*IN_COLS
        float* out_data        // Output data array
        ) {
    // Calculate output dimensions based on stride
    const int OUT_ROWS = (IN_ROWS - POOL_SIZE) / STRIDE + 1;
    const int OUT_COLS = (IN_COLS - POOL_SIZE) / STRIDE + 1;
    const float pool_area = POOL_SIZE * POOL_SIZE;

    static float plane[IN_ROWS][IN_COLS]; // scratch pad
    #pragma HLS ARRAY_PARTITION variable=plane cyclic factor=4 dim=1

    for(int channel = 0; channel < IN_C; channel++) {
        // Process each channel
        const int channel_offset = channel * IN_ROWS * IN_COLS;
        const int out_channel_offset = channel * OUT_ROWS * OUT_COLS;

        // Load channel data into scratch pad
        for(int r = 0; r < IN_ROWS; ++r) {
            for(int c = 0; c < IN_COLS; ++c) {
                #pragma HLS PIPELINE II=1
                plane[r][c] = in_data[channel_offset + r * IN_COLS + c];
            }
        }
        
        // Perform pooling with stride
        for(int out_r = 0; out_r < OUT_ROWS; out_r++) {
            for(int out_c = 0; out_c < OUT_COLS; out_c++) {
                #pragma HLS PIPELINE II=1
                float sum = 0.0f;

                // Sum the values in the pooling window
                for(int i = 0; i < POOL_SIZE; i++) {
                    for(int j = 0; j < POOL_SIZE; j++) {
                        int r = out_r * STRIDE + i;
                        int c = out_c * STRIDE + j;
                        sum += plane[r][c];
                    }
                }
                // printf("sum = %f\n", sum);
                // Compute average
                float avg = sum / pool_area;
                
                // Store result
                out_data[out_channel_offset + out_r * OUT_COLS + out_c] = avg;
            }
        }
    }
}

#endif