#ifndef FC_H
#define FC_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <math.h>
#include <stdint.h>

typedef ap_int<32> data_t; // 8 bit fixed point as precision

//----------------------
// fully connected layer
//----------------------
template<int IN_DIM, int OUT_DIM>
void fc(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t weight[IN_DIM][OUT_DIM],
        const data_t bias[OUT_DIM],
        bool use_relu = true,
        float act_out_scale=1, int act_out_zp=0
        ) {
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    data_t line_buffer[OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    for(int j=0; j<OUT_DIM; ++j) {
        #pragma HLS PIPELINE II=1
        line_buffer[j] = bias[j];
    }

    for(int i=0; i<IN_DIM; ++i) {
        data_t val = in_stream.read();
        for(int j=0; j<OUT_DIM; ++j) {
            #pragma HLS PIPELINE II=1
            line_buffer[j] += val * weight[i][j];
        }
    }

    // post quant activation
    for(int j=0; j<OUT_DIM; ++j) {
        #pragma HLS PIPELINE II=1
        float val_scaled = (float)line_buffer[j] * act_out_scale + (float)act_out_zp;
        float val_rounded = (val_scaled >= 0.0f) ? (floorf(val_scaled) + 0.5f) : (floorf(val_scaled) - 0.5f);
        data_t sat;
        if (val_rounded > (float)INT32_MAX) sat = INT32_MAX;
        else if(val_rounded < (float)INT32_MIN) sat = INT32_MIN;
        else sat = (data_t)val_rounded;

        if(use_relu && sat < 0) {
            sat = 0;
        }
        data_t result = (data_t)sat;
        out_stream.write(result);
    }
}

data_t relu(data_t x) {
    return x > 0 ? x : data_t(0);
}


#endif
