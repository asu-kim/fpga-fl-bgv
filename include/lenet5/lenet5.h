#ifndef LENET5_H
#define LENET5_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include "conv2d.h"
#include "avg_pool.h"
#include "fc_layer.h"
#include "flatten.h"

typedef ap_int<32> data_t;
typedef ap_int<32> bias_t;

void lenet5(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t conv1_weight[6][1][5][5],
        const bias_t conv1_bias[6],
        const float conv1_act_out_scale,
        const int conv1_act_out_zp,
        const data_t conv2_weight[16][6][5][5],
        const bias_t conv2_bias[16],
        const float conv2_act_out_scale,
        const int conv2_act_out_zp,
        const data_t fc1_weight[120][256],
        const bias_t fc1_bias[120],
        const float fc1_act_out_scale,
        const int fc1_act_out_zp,
        const data_t fc2_weight[84][120],
        const bias_t fc2_bias[84],
        const float fc2_act_out_scale,
        const int fc2_act_out_zp,
        const data_t fc3_weight[10][84],
        const bias_t fc3_bias[10],
        const float fc3_act_out_scale,
        const int fc3_act_out_zp,
    ) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS DATAFLOW
    
    hls::stream<data_t> conv1_out, pool1_out, conv2_out, pool2_out, flatten_out, fc1_out, fc2_out;

    // tempalte param: # of filters, size of kernel
    // layer1: conv2d [Nx1x10x10 -> Nx1x6x6]
    conv2d<6, 1, 5>(in_stream, conv1_out, conv1_weight, conv1_bias, 28, 28, conv1_act_out_scale, conv1_act_out_zp);

    // template param: pool size
    // layer2: avg pool [Nx1x6x6 -> Nx1x3x3]
    avg_pool<2>(conv1_out, pool1_out, 24, 24, 6);

    // flatten [Nx3x3x1 -> Nx9]
    flatten<4,4,16>(pool2_out, flatten_out);

    // layer5: mlp [Nx9 -> Nx5]
    fc_layer<120, 256>(flatten_out, fc1_out, fc1_weight, fc1_bias, fc1_act_out_scale, fc1_act_out_zp, true);
}

#endif
