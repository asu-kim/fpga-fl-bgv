#ifndef NTT_TRANSFORM_H
#define NTT_TRANSFORM_H

#include "data_type.hpp"

extern "C"
{
    void bit_reverse(data_t *coeffs);
    void ntt_transform(data_t *coeffs, data_t *result);
    void intt_transform(data_t *coeffs, data_t *result);
}

#endif
