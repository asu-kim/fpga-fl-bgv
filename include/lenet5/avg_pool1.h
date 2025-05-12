#ifndef AVG_POOL1_H
#define AVG_POOL1_H

#include "data_type.hpp"

extern "C" {
void avg_pool1(
    data_ap_fixed_t* in_data,
    data_ap_fixed_t* out_data
);
}
#endif
