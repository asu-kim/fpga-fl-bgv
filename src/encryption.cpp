// #include "hls_stream.h"
#include "encryption.hpp"

extern "C" {
void encryption(data_t* in1, data_t* in2, data_t* out, int size) {
    #pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 depth=4096
    #pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 depth=4096
    #pragma HLS INTERFACE m_axi port=out bundle=gmem2 depth=4096
    #pragma HLS INTERFACE s_axilite port=size bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    for(int i = 0; i < size; i++) {
        // #pragma HLS PIPELINE II=1
        out[i] = hls::remainder(in1[i], in2[i]);
    }
}
}