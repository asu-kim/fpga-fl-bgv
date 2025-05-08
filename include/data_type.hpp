#ifndef DATA_TYPE_HPP
#define DATA_TYPE_HPP

#include "ap_int.h"
#include "constants.hpp"

// typedef ap_int<COEFFICIENT_WIDTH> data_t;
typedef ap_int<54> data_t;

typedef ap_int<54> data_ap_fixed_t;

const data_t MAX_VAL = 9007199254740991;
const data_t MIN_VAL = -9007199254740992;
// typedef int data_t;
// typedef int64_t data_t;

#endif // DATA_TYPE_HPP
