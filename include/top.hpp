#ifndef TOP_HPP
#define TOP_HPP

#include "hls_math.h"
#include "data_type.hpp"

extern "C"
{
    void top(
        data_t *private_key,
        data_t *encrypted_conv1_weight0_0,
        data_t *encrypted_conv1_weight0_1,
        data_t *encrypted_conv1_weight1_0,
        data_t *encrypted_conv1_weight1_1,
        data_t *encrypted_conv1_bias0,
        data_t *encrypted_conv1_bias1,
        data_t *input,
        data_t *output);
}

#endif // TOP_HPP
