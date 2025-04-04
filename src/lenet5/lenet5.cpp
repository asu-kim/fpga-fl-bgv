#include "lenet5.h"

void lenet5(
        hls::stream<data_t>& in_stream,
        hls::stream<data_t>& out_stream,
        const data_t conv1_weights[6][5][5],
        const data_t conv1_bias[6],
        const data_t conv2_weights[16][6][5][5],
        const data_t conv2_bias[16],
        const data_t fc1_weights[120][256],
        const data_t fc1_bias[120],
        const data_t fc2_weights[84][120],
        const data_t fc2_bias[84],
        const data_t fc3_weights[10][84],
        const data_t fc3_bias[10]
        ) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS DATAFLOW
    
    hls::stream<data_t> conv1_out, pool1_out, conv2_out, pool2_out, flatten_out, fc1_out, fc2_out;

    // tempalte param: # of filters, size of kernel
    // layer1: conv2d [Nx1x28x28 -> Nx24x24x6]
    conv2d<6, 5>(in_stream, conv1_out, conv1_weights, conv1_bias, 28, 28);

    // template param: pool size
    // layer2: avg pool [Nx24x24x6 -> Nx12x12x6]
    avg_pool<2>(conv1_out, pool1_out, 24, 24, 6);


    // layer3: conv2d [Nx12x12x6 -> Nx8x8x16]
    conv2d<16, 5>(pool1_out, conv2_out, conv2_weights, conv2_bias, 12, 12);

    // layer4: pool2d [Nx8x8x16 -> Nx4x4x16]
    avg_pool<2>(conv2_out, pool2_out, 8, 8, 16);

    // flatten [Nx4x4x16 -> Nx256]
    flatten<4,4,16>(pool2_out, flatten_out);

    // layer5: mlp [Nx256 -> Nx120]
    fc_layer<120, 256>(flatten_out, fc1_out, true);

    // layer6: mlp [Nx120 -> Nx84]
    fc_layer<84, 120>(fc1_out, fc2_out, true);

    // layer7: mlp [Nx84 -> Nx10]
    fc_layer<10, 84>(fc2_out, out_stream, false);
}

