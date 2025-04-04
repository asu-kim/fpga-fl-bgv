#ifndef FLATTEN_H
#define FLATTEN_H

#include <hls_stream.h>
#include <ap_fixed.h>

typedef ap_fixed<8, 3> data_t;

template<int N, int... Ns>
void flatten(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream);

#endif
