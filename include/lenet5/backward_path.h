#ifndef BACKWARD_PATH_H
#define BACKWARD_PATH_H

#include "data_type.hpp"

extern "C" {
void backward_path(
    const data_ap_fixed_t* in_data,             // gmem0
    const data_ap_fixed_t* weights,             // gmem1 - consolidated weights
    const data_ap_fixed_t* biases,              // gmem2 - consolidated biases
    const data_ap_fixed_t* outputs,             // gmem3 - consolidated outputs
    const data_ap_fixed_t* label,               // gmem4
    data_ap_fixed_t* updated_weights,           // gmem5 - consolidated updated weights
    data_ap_fixed_t* updated_biases,            // gmem6 - consolidated updated biases
    data_ap_fixed_t& loss
);
}
#endif
