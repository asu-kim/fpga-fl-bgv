// #include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/fc_layer.h"
#include "lenet5/fc2.h"

extern "C" {
    void fc2(
        const float* in_data,
        float* out_data,
        const float* weight,
        const float* bias,
        bool use_relu
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=120
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=84
        #pragma HLS INTERFACE m_axi port=weight bundle=gmem2 depth=120*84
        #pragma HLS INTERFACE m_axi port=bias bundle=gmem3 depth=84

        #pragma HLS INTERFACE s_axilite port=in_data   bundle=control
        #pragma HLS INTERFACE s_axilite port=out_data  bundle=control
        #pragma HLS INTERFACE s_axilite port=weight    bundle=control
        #pragma HLS INTERFACE s_axilite port=bias      bundle=control
        #pragma HLS INTERFACE s_axilite port=use_relu  bundle=control
        #pragma HLS INTERFACE s_axilite port=return    bundle=control

        fc<120, 84>(in_data, out_data, weight, bias, use_relu);
    }
}