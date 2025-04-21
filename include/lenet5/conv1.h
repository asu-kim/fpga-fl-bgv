#ifndef CONV1_H
#define CONV1_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include "data_type.hpp"

extern "C" {
// void conv1(
//     hls::stream<data_t>& in_stream,
//     hls::stream<data_t>& out_stream,
//     Parameter* param
// );
void conv1(
    hls::stream<data_t>& in_stream,
    hls::stream<data_t>& out_stream,
    data_t conv1_weight[6][1][5][5],
    data_t conv1_bias[6]
);
}
#endif
