#include "constants.hpp"
#include "hls_math.h"
#include "hls_stream.h"
#include "modulo_reduction.hpp"

data_t modulo_reduction(data_t in1, data_t in2) {
    // #pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 depth=POLYNOMIAL_DEGREE
    // #pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 depth=POLYNOMIAL_DEGREE
    // #pragma HLS INTERFACE m_axi port=out bundle=gmem2 depth=POLYNOMIAL_DEGREE

    data_t temp = hls::remainder(in1, in2);
    temp = temp < 0 ? temp + in2 : temp;
    return temp;
}