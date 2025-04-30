// #include "hls_stream.h"
#include <hls_math.h>
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d_bwd.h"
#include "lenet5/conv1_bwd.h"

int OUT_C = 16;
int IN_C = 6;
int KERNEL_SIZE = 5;
int ROW = 12;
int COL = 12;

extern "C" {
    void conv2_bwd(
        const float* in_activation,
        const float* grads,
        const float* in_weight,
        float* out_grads,
        float* dW,
        float* dB
    ) { 
        #pragma HLS INTERFACE m_axi port=in_activation bundle=gmem0 depth=864
        #pragma HLS INTERFACE m_axi port=grads bundle=gmem1 depth=1024
        #pragma HLS INTERFACE m_axi port=in_weight bundle=gmem2 depth=2400
        #pragma HLS INTERFACE m_axi port=out_grads bundle=gmem3 depth=864
        #pragma HLS INTERFACE m_axi port=dW bundle=gmem4 depth=2400
        #pragma HLS INTERFACE m_axi port=dB bundle=gmem5 depth=16

        #pragma HLS INTERFACE s_axilite port=in_activation bundle=control
        #pragma HLS INTERFACE s_axilite port=grads bundle=control
        #pragma HLS INTERFACE s_axilite port=in_weight bundle=control
        #pragma HLS INTERFACE s_axilite port=out_grads bundle=control
        #pragma HLS INTERFACE s_axilite port=dW bundle=control
        #pragma HLS INTERFACE s_axilite port=dB bundle=control
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        conv2d_backward<16, 6, 5, 12, 12>(in_activation, grads, in_weight, out_grads, dW, dB);
    }
}