#ifndef ENCRYPTION_HPP
#define ENCRYPTION_HPP

#include "hls_math.h"
#include "data_type.h"

extern "C" {
    void encryption(data_t* in1, data_t* in2, data_t* out, int size);
}

#endif // ENCRYPTION_HPP
