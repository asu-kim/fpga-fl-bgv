#ifndef FORWARD_PATH_H
#define FORWARD_PATH_H

#include "data_type.hpp"

extern "C" {
void forward_path(
    data_ap_fixed_t* in_data,
    data_ap_fixed_t* weights,       // Single array for all weights
    data_ap_fixed_t* biases,        // Single array for all biases
    data_ap_fixed_t* outs
);
}
#endif
