#ifndef FC_LAYER_H
#define FC_LAYER_H

#include <hls_stream.h>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t;

template<int OUT_DIM, int IN_DIM>
void fc_layer(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weight[OUT_DIM][IN_DIM],
        const data_t bias[OUT_DIM],
        bool use_relu
        ) {
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    data_t ret[OUT_DIM];
    for(int j=0; j<OUT_DIM; ++j) {
        ret[j] = bias[j];
    }

    for(int i=0; i<IN_DIM; ++i) {
        data_t val = in_stream.read();
        for(int j=0; j<OUT_DIM; ++j) {
#pragma HLS UNROLL factor=4
            ret[j] += weight[j][i] * val;
        }
    }

    for(int j=0; j<OUT_DIM; ++j) {
        data_t result = use_relu ? relu(ret[j]) : ret[j];
        out_stream.write(result);
    }
}

data_t relu(data_t x) {
    return x > 0 ? x : data_t(0);
}

#endif
