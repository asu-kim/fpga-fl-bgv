#include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"

extern "C" {
    void conv1(
        // hls::stream<data_t>& in_stream,
        // hls::stream<data_t>& out_stream,
        data_t* in_data,
        data_t* out_data,
        data_t* conv1_weight,
        data_t* conv1_bias
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=100
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=64
        #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem3 depth=128
        #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=128

        // Create local copy
        data_t local_weight[25];
        data_t local_bias[1];

        // Copy data
        for(int i=0; i<1; i++) {
            local_bias[i] = conv1_bias[i];
            for(int j=0; j<1; j++) {
                for(int k=0; k<5; k++) {
                    for(int l=0; l<5; l++) {
                        local_weight[k*5+l] = conv1_weight[i * 25 + j * 25 + k * 5 + l];
                    }
                }
            }
        }
        // conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
        conv2d<1, 1, 5, 10, 10>(in_data, out_data, local_weight, local_bias);
    }
}