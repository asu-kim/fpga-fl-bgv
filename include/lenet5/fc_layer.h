#ifndef FC_LAYER_H
#define FC_LAYER_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <cmath>
#include <limits>

typedef ap_int<32> data_t;

template<int OUT_DIM, int IN_DIM>
void fc_layer(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weight[OUT_DIM][IN_DIM],
        const data_t bias[OUT_DIM],
        float act_out_scale=1,
        int act_out_zp=0,
        bool use_relu=true
        ) {
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    ap_int<64> ret[OUT_DIM];
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

    // post quant activation
    for(int j=0; j<OUT_DIM; ++j) {
        #pragma HLS PIPELINE II=1
        float val_scaled = (float)ret[j] * act_out_scale + (float)act_out_zp;
        int32_t val_rounded = (int32_t)std::floor(val_scaled + 0.5f);
        val_rounded = std::max(std::numeric_limits<int32_t>::min(), std::min(std::numeric_limits<int32_t>::max(), val_rounded));

        if(use_relu && val_rounded < 0) {
            val_rounded = 0;
        }
        data_t result = (data_t)val_rounded;
        out_stream.write(result);
    }
}

data_t relu(data_t x) {
    return x > 0 ? x : data_t(0);
}

#endif
