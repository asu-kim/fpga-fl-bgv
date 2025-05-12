#ifndef CONV2_H
#define CONV2_H

// #include <hls_stream.h>
// #include "data_type.hpp"

extern "C" {
void conv2(
    data_ap_fixed_t in_data[864],
    data_ap_fixed_t out_data[2304],
    data_ap_fixed_t conv1_weight[2400],
    data_ap_fixed_t conv1_bias[16]
);
}
#endif
