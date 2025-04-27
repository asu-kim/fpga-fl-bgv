// #include "hls_stream.h"
#include <hls_math.h>
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"

extern "C" {
    void conv2(
        // hls::stream<float>& in_stream,
        // hls::stream<float>& out_stream,
        float* in_data,
        float* out_data,
        float* conv1_weight,
        float* conv1_bias
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=864
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=2304
        #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem2 depth=2400
        #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=16
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        // Create local copy
        float local_weight[16*6*5*5];
        float local_bias[16];

        // Copy data
        for(int i=0; i<16*6*5*5; i++) {
            local_weight[i] = conv1_weight[i];
        }
        for(int i=0; i<16; i++) {
            local_bias[i] = conv1_bias[i];
        }
        conv2d<16, 6, 5, 12, 12>(in_data, out_data, local_weight, local_bias);
    }
}