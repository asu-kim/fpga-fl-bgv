#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/fc_bwd.h"
#include "lenet5/fc3_bwd.h"

extern "C" {
    void fc3_bwd(
        const data_ap_fixed_t in_activation[84],
        const data_ap_fixed_t grads[10],
        const data_ap_fixed_t in_weight[84*10],
        data_ap_fixed_t dX[84],
        data_ap_fixed_t dW[84*10],
        data_ap_fixed_t dB[10]
    ) { 
        #pragma HLS INTERFACE m_axi port=in_activation bundle=gmem0 depth=84
        #pragma HLS INTERFACE m_axi port=grads bundle=gmem1 depth=10
        #pragma HLS INTERFACE m_axi port=in_weight bundle=gmem2 depth=84*10
        #pragma HLS INTERFACE m_axi port=dX bundle=gmem2 depth=84
        #pragma HLS INTERFACE m_axi port=dW bundle=gmem3 depth=84*10
        #pragma HLS INTERFACE m_axi port=dB bundle=gmem3 depth=10


        #pragma HLS INTERFACE s_axilite port=in_activation  bundle=control
        #pragma HLS INTERFACE s_axilite port=grads          bundle=control
        #pragma HLS INTERFACE s_axilite port=in_weight      bundle=control
        #pragma HLS INTERFACE s_axilite port=dX             bundle=control
        #pragma HLS INTERFACE s_axilite port=dW             bundle=control
        #pragma HLS INTERFACE s_axilite port=dB             bundle=control
        #pragma HLS INTERFACE s_axilite port=return         bundle=control

        fc_backward<84, 10>(in_activation, grads, in_weight, dX, dW, dB, false);
    }
}