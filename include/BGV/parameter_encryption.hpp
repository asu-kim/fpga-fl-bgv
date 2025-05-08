#ifndef PARAMETER_ENCRYPTION_HPP
#define PARAMETER_ENCRYPTION_HPP

#include "constants.hpp"
#include "data_type.hpp"

extern "C" {
void parameter_encryption(
    float pt[POLYNOMIAL_DEGREE],
    float scale,
    float zp,
    data_t errors[POLYNOMIAL_DEGREE*3],
    data_t pk0[POLYNOMIAL_DEGREE],
    data_t pk1[POLYNOMIAL_DEGREE],

    data_t ct0[POLYNOMIAL_DEGREE],
    data_t ct1[POLYNOMIAL_DEGREE]
);
}

#endif // PARAMETER_ENCRYPTION_HPP
