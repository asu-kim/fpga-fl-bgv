#ifndef ENCRYPTION_HPP
#define ENCRYPTION_HPP

#include "hls_math.h"
#include "data_type.hpp"

extern "C" {
    void encryption(data_t* e1, data_t* e2, data_t* r, data_t* pk1, data_t* pk2, data_t* pt, data_t* ct1, data_t* ct2);
    void decryption(data_t* sk, data_t* ct1, data_t* ct2, data_t* pt);
    void top_encryption_decryption_test(
        data_t* e1, data_t* e2, data_t* r,
        data_t* sk, data_t* pk1, data_t* pk2,
        data_t* original_pt, data_t* decrypted_pt);
}

#endif // ENCRYPTION_HPP
