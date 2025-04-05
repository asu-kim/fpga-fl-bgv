#ifndef FLATTEN_H
#define FLATTEN_H

#include <hls_stream.h>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t;

template<int... DIMs>
void flatten(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
#pragma HLS PIPELINE II=1
    const int flat_dim = (1 * ... * DIMs);
    for(int i=0; i<flat_dim; ++i) {
        out_stream.write(in_stream.read());
    }
}

#endif
