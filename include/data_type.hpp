#ifndef DATA_TYPE_HPP
#define DATA_TYPE_HPP

#include "ap_int.h"
#include "constants.hpp"

// typedef ap_int<COEFFICIENT_WIDTH> data_t;
typedef ap_int<54> data_t;
// typedef int data_t;
// typedef int64_t data_t;

struct Parameter
{
    /* data */
    data_t conv1_weight[6][1][5][5];
    data_t conv1_bias[6];
};


#endif // DATA_TYPE_HPP
