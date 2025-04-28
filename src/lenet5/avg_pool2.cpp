// #include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/avg_pool.h"
#include "lenet5/avg_pool1.h"
#include <stdio.h>

extern "C" {
    void avg_pool2(
        // hls::stream<data_t>& in_stream,
        // hls::stream<data_t>& out_stream,
        float* in_data,
        float* out_data
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=16*8*8
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=16*(8/2)*(8/2)

        avg_pool<2, 2, 16, 8, 8>(in_data, out_data);
    }

    // int POOL_SIZE = 2; int STRIDE = 2; int IN_ROWS = 24; int IN_COLS = 24; int IN_C = 1;
    // void avg_pool1(
    //     // hls::stream<data_t>& in_stream,
    //     // hls::stream<data_t>& out_stream,
    //     float* in_data,
    //     float* out_data
    // ) { 
    //     #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=1*24*24
    //     #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=1*(24/2)*(24/2)
    //     const int OUT_ROWS = (IN_ROWS - POOL_SIZE) / STRIDE + 1;
    //     const int OUT_COLS = (IN_COLS - POOL_SIZE) / STRIDE + 1;
    //     const float pool_area = POOL_SIZE * POOL_SIZE;

    //     static float plane[24][24]; // scratch pad
    //     #pragma HLS ARRAY_PARTITION variable=plane complete dim=1

    //     for(int channel = 0; channel < IN_C; channel++) {
    //         // Process each channel
    //         const int channel_offset = channel * IN_ROWS * IN_COLS;
    //         const int out_channel_offset = channel * OUT_ROWS * OUT_COLS;

    //         // Load channel data into scratch pad
    //         for(int r = 0; r < IN_ROWS; ++r) {
    //             for(int c = 0; c < IN_COLS; ++c) {
    //                 #pragma HLS PIPELINE II=1
    //                 plane[r][c] = in_data[channel_offset + r * IN_COLS + c];
    //             }
    //         }
            
    //         // Perform pooling with stride
    //         for(int out_r = 0; out_r < OUT_ROWS; out_r++) {
    //             for(int out_c = 0; out_c < OUT_COLS; out_c++) {
    //                 #pragma HLS PIPELINE II=3
    //                 float sum = 0.0f;

    //                 // Sum the values in the pooling window
    //                 for(int i = 0; i < POOL_SIZE; i++) {
    //                     for(int j = 0; j < POOL_SIZE; j++) {
    //                         int r = out_r * STRIDE + i;
    //                         int c = out_c * STRIDE + j;
    //                         sum += plane[r][c];
    //                     }
    //                 }
    //                 printf("r = %d, c = %d\n", out_r, out_c);
    //                 printf("sum = %f\n", sum);
    //                 // Compute average
    //                 float avg = sum / pool_area;
                    
    //                 // Store result
    //                 out_data[out_channel_offset + out_r * OUT_COLS + out_c] = avg;
    //                 printf("avg = %f\n\n", sum);
    //             }
    //         }
    //     }
    //     // avg_pool<2, 2, 1, 24, 24>(in_data, out_data);
    // }
}