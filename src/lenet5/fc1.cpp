#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/fc_layer.h"
#include "lenet5/fc1.h"

extern "C" {
    void fc1(
        const data_ap_fixed_t* in_data,
        data_ap_fixed_t* out_data,
        const data_ap_fixed_t* weight,
        const data_ap_fixed_t* bias,
        bool use_relu
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=256
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=120
        #pragma HLS INTERFACE m_axi port=weight bundle=gmem2 depth=256*120
        #pragma HLS INTERFACE m_axi port=bias bundle=gmem3 depth=120

        #pragma HLS INTERFACE s_axilite port=in_data   bundle=control
        #pragma HLS INTERFACE s_axilite port=out_data  bundle=control
        #pragma HLS INTERFACE s_axilite port=weight    bundle=control
        #pragma HLS INTERFACE s_axilite port=bias      bundle=control
        #pragma HLS INTERFACE s_axilite port=use_relu  bundle=control
        #pragma HLS INTERFACE s_axilite port=return    bundle=control

        fc<256, 120>(in_data, out_data, weight, bias, use_relu);
    }
}