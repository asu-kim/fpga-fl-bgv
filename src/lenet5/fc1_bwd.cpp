#include "lenet5/fc_bwd.h"
#include "lenet5/fc1_bwd.h"

extern "C" {
    void fc1_bwd(
        const float in_activation[256],
        const float grads[120],
        const float in_weight[256*120],
        float dX[256],
        float dW[256*120],
        float dB[120]
    ) { 
        #pragma HLS INTERFACE m_axi port=in_activation bundle=gmem0 depth=256
        #pragma HLS INTERFACE m_axi port=grads bundle=gmem1 depth=120
        #pragma HLS INTERFACE m_axi port=in_weight bundle=gmem2 depth=256*120
        #pragma HLS INTERFACE m_axi port=dX bundle=gmem2 depth=256
        #pragma HLS INTERFACE m_axi port=dW bundle=gmem3 depth=256*120
        #pragma HLS INTERFACE m_axi port=dB bundle=gmem3 depth=120


        #pragma HLS INTERFACE s_axilite port=in_activation  bundle=control
        #pragma HLS INTERFACE s_axilite port=grads          bundle=control
        #pragma HLS INTERFACE s_axilite port=in_weight      bundle=control
        #pragma HLS INTERFACE s_axilite port=dX             bundle=control
        #pragma HLS INTERFACE s_axilite port=dW             bundle=control
        #pragma HLS INTERFACE s_axilite port=dB             bundle=control
        #pragma HLS INTERFACE s_axilite port=return         bundle=control

        fc_backward<256, 120>(in_activation, grads, in_weight, dX, dW, dB, false);
    }
}