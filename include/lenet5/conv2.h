#ifndef CONV2_H
#define CONV2_H

// #include <hls_stream.h>
// #include "data_type.hpp"

extern "C" {
void conv2(
    float in_data[864],
    float out_data[2304],
    float conv1_weight[2400],
    float conv1_bias[16]
);
}
#endif
