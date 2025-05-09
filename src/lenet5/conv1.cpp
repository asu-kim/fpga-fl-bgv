// #include "hls_stream.h"
#include <hls_math.h>
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"

int OUT_C = 6;
int IN_C = 1;
int KERNEL_SIZE = 5;
int ROW = 28;
int COL = 28;

extern "C" {
    void conv1(
        // hls::stream<data_ap_fixed_t>& in_stream,
        // hls::stream<data_ap_fixed_t>& out_stream,
        data_ap_fixed_t* in_data,
        data_ap_fixed_t* out_data,
        data_ap_fixed_t* conv1_weight,
        data_ap_fixed_t* conv1_bias
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=3456
        #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem2 depth=256
        #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=128
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        // Create local copy
        data_ap_fixed_t local_weight[6*1*5*5];
        data_ap_fixed_t local_bias[6];

        // Copy data
        for(int i=0; i<150; i++) {
            local_weight[i] = conv1_weight[i];
        }
        for(int i=0; i<6; i++) {
            local_bias[i] = conv1_bias[i];
        }
        conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
    }
}