#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/fc_bwd.h"
#include "lenet5/fc3_bwd.h"

extern "C" {
    void fc3_bwd(
        const float in_activation[84],
        const float grads[10],
        const float in_weight[84*10],
        float dX[84],
        float dW[84*10],
        float dB[10]
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