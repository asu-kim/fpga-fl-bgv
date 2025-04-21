#include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"

extern "C" {
    // void conv1(
    //     hls::stream<data_t>& in_stream,
    //     hls::stream<data_t>& out_stream,
    //     Parameter* param
    // ) {
    //     // #pragma HLS INTERFACE axis port=in_stream 
    //     // #pragma HLS INTERFACE axis port=out_stream
    //     #pragma HLS INTERFACE m_axi port=param bundle=gmem2 offset=slave depth = 156

    //     // Create local copy
    //     data_t local_weight[6][1][5][5];
    //     data_t local_bias[6];
        
    //     // Copy data
    //     for(int i=0; i<6; i++) {
    //         local_bias[i] = param->conv1_bias[i];
    //         for(int j=0; j<1; j++) {
    //             for(int k=0; k<5; k++) {
    //                 for(int l=0; l<5; l++) {
    //                     local_weight[i][j][k][l] = param->conv1_weight[i][j][k][l];
    //                 }
    //             }
    //         }
    //     }
        
    //     // Now use local arrays with partition pragmas
    //     #pragma HLS ARRAY_PARTITION variable=local_weight complete dim=1
    //     #pragma HLS ARRAY_PARTITION variable=local_bias complete dim=1
    //     conv2d<6, 1, 5, 28, 28>(in_stream, out_stream, local_weight, local_bias); 
    // }
    // In conv1.cpp
    void conv1(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        data_t conv1_weight[6][1][5][5],
        data_t conv1_bias[6]
    ) {
        #pragma HLS INTERFACE s_axilite port=conv1_weight bundle=control depth=150
        #pragma HLS INTERFACE s_axilite port=conv1_bias bundle=control depth=6
        
        // // Then use these arrays directly
        // #pragma HLS ARRAY_PARTITION variable=conv1_weight complete dim=1
        // #pragma HLS ARRAY_PARTITION variable=conv1_bias complete dim=1
        conv2d<6, 1, 5, 28, 28>(in_stream, out_stream, conv1_weight, conv1_bias);
    }
}