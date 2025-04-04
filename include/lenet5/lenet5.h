#ifndef LENET5_H
#define LENET5_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include "conv2d.h"
#include "avg_pool.h"
#include "fc_layer.h"

typedef ap_fixed<8, 3> data_t;

void lenet5(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t conv1_weights[6][5][5],
        const data_t conv1_bias[6],
        const data_t conv2_weights[16][6][5][5],
        const data_t conv2_bias[16],
        const data_t fc1_weights[120][400],
        const data_t fc1_bias[120],
        const data_t fc2_weights[84][120],
        const data_t fc2_bias[84],
        const data_t fc3_weights[10][84],
        const data_t fc3_bias[10]
    );

#endif
