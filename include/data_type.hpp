#ifndef DATA_TYPE_HPP
#define DATA_TYPE_HPP

#include "ap_int.h"
#include "constants.hpp"

// typedef ap_int<COEFFICIENT_WIDTH> data_t;
typedef ap_int<54> data_t;
// typedef int data_t;
// typedef int64_t data_t;

struct Weight
{
    /* data */
    data_t weight[6][1][5];
};


#endif // DATA_TYPE_HPP
