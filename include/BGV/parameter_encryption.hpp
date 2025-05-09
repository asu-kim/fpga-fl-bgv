#ifndef PARAMETER_ENCRYPTION_HPP
#define PARAMETER_ENCRYPTION_HPP

#include "constants.hpp"
#include "data_type.hpp"

extern "C" {
void parameter_encryption(
    data_ap_fixed_t pt[POLYNOMIAL_DEGREE],
    data_ap_fixed_t scale,
    data_ap_fixed_t zp,
    data_t errors[POLYNOMIAL_DEGREE*3],
    data_t pk0[POLYNOMIAL_DEGREE],
    data_t pk1[POLYNOMIAL_DEGREE],

    data_t ct0[POLYNOMIAL_DEGREE],
    data_t ct1[POLYNOMIAL_DEGREE]
);
}

#endif // PARAMETER_ENCRYPTION_HPP
