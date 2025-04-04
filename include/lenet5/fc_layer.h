#ifndef FC_LAYER_H
#define FC_LAYER_H

#include <hls_stream.h>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t;

template<int OUT_DIM, IN_DIM>
void fc_layer(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weights[OUT_DIM][IN_DIM],
        data_t bias[OUT_DIM],
        bool use_relu
        );

data_t relu(data_t x);
#endif
