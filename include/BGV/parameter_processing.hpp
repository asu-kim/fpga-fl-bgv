#ifndef PARAMETER_PROCESSING_HPP
#define PARAMETER_PROCESSING_HPP

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

void parameter_decryption(
    data_t* sk,
    data_t* ct0,
    data_t* ct1,
    float scale,
    float zp,

    float* pt
);
}

#endif // ENCRYPTION_HPP
