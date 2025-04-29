#ifndef FLATTEN_H
#define FLATTEN_H

#include <hls_stream.h>

//--------
// flatten 
//--------
template<int H, int W, int C >
void flatten(
    const float *in,
    float *out
        ) {
#pragma HLS PIPELINE II=1
    const int flat_dim = H*W*C;
    for(int i=0; i<flat_dim; ++i) {
        out[i] = in[i];
    }
}

#endif
