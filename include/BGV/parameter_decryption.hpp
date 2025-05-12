#ifndef PARAMETER_DECRYPTION_HPP
#define PARAMETER_DECRYPTION_HPP

#include "constants.hpp"
#include "data_type.hpp"

extern "C" {
void parameter_decryption(
    data_t sk[POLYNOMIAL_DEGREE],
    data_t ct0[POLYNOMIAL_DEGREE],
    data_t ct1[POLYNOMIAL_DEGREE],
    data_ap_fixed_t scale,
    data_ap_fixed_t zp,

    data_ap_fixed_t pt[POLYNOMIAL_DEGREE]
);
}

#endif // PARAMETER_DECRYPTION_HPP
