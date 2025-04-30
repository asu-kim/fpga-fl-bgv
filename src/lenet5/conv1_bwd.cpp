#include "lenet5/conv2d_bwd.h"
#include "lenet5/conv1_bwd.h"

int OUT_C = 6;
int IN_C = 1;
int KERNEL_SIZE = 5;
int ROW = 28;
int COL = 28;

extern "C" {
    void conv1_bwd(
        const float* in_activation,
        const float* grads,
        const float* in_weight,
        float* out_grads,
        float* dW,
        float* dB
    ) { 
        #pragma HLS INTERFACE m_axi port=in_activation bundle=gmem0 depth=784
        #pragma HLS INTERFACE m_axi port=grads bundle=gmem1 depth=3456
        #pragma HLS INTERFACE m_axi port=in_weight bundle=gmem2 depth=150
        #pragma HLS INTERFACE m_axi port=out_grads bundle=gmem3 depth=784
        #pragma HLS INTERFACE m_axi port=dW bundle=gmem4 depth=150
        #pragma HLS INTERFACE m_axi port=dB bundle=gmem5 depth=6

        #pragma HLS INTERFACE s_axilite port=in_activation bundle=control
        #pragma HLS INTERFACE s_axilite port=grads bundle=control
        #pragma HLS INTERFACE s_axilite port=in_weight bundle=control
        #pragma HLS INTERFACE s_axilite port=out_grads bundle=control
        #pragma HLS INTERFACE s_axilite port=dW bundle=control
        #pragma HLS INTERFACE s_axilite port=dB bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        // // Create local copy
        // float local_weight[6*1*5*5];
        // float local_bias[6];

        // // Copy data
        // for(int i=0; i<150; i++) {
        //     local_weight[i] = conv1_weight[i];
        // }
        // for(int i=0; i<6; i++) {
        //     local_bias[i] = conv1_bias[i];
        // }
        conv2d_backward<6, 1, 5, 28, 28>(in_activation, grads, in_weight, out_grads, dW, dB);
    }
}