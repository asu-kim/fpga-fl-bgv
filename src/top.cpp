#include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "encryption.hpp"

extern "C" {
void top(
    hls::stream<data_t>& in_stream,
    hls::stream<data_t>& out_stream,
    Shared_mem shm_mem
    ) {
    #pragma HLS INTERFACE m_axi port=e1 bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=e2 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=r bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pk1 bundle=gmem0 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pk2 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=pt bundle=gmem1 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct1 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    #pragma HLS INTERFACE m_axi port=ct2 bundle=gmem2 depth=POLYNOMIAL_DEGREE
    // #pragma HLS INTERFACE s_axilite port=size bundle=control
    // #pragma HLS INTERFACE s_axilite port=return bundle=control

    conv2d<6, 1, 5>(in_stream, conv1_out, shm_mem->weight, shm_mem->bias, 28, 28, conv1_act_out_scale, conv1_act_out_zp);
}
}