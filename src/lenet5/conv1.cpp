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
        // hls::stream<data_t>& in_stream,
        // hls::stream<data_t>& out_stream,
        data_t* in_data,
        data_t* out_data,
        data_t* conv1_weight,
        data_t* conv1_bias
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=3456
        #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem3 depth=256
        #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=128

        // Create local copy
        data_t local_weight[6][1][5][5];
        data_t local_bias[6];

        // Copy data
        for(int i=0; i<6; i++) {
            local_bias[i] = conv1_bias[i];
            for(int j=0; j<1; j++) {
                for(int k=0; k<5; k++) {
                    for(int l=0; l<5; l++) {
                        local_weight[i][j][k][l] = conv1_weight[i * 25 + j * 25 + k * 5 + l];
                    }
                }
            }
        }
        // conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
        conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
    }
}