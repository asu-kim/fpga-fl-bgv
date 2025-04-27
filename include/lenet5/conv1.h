#ifndef CONV1_H
#define CONV1_H

// #include <hls_stream.h>
#include "data_type.hpp"

extern "C" {
// void conv1(
//     hls::stream<float>& in_stream,
//     hls::stream<float>& out_stream,
//     Parameter* param
// );
void conv1(
    // hls::stream<float>& in_stream,
    // hls::stream<float>& out_stream,
    float in_data[784],
    float out_data[3456],
    float conv1_weight[256],
    float conv1_bias[128]
);
}
#endif
