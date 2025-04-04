#include "flatten.h"

template<int N, int... Ns>
void flatten(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
#pragma HLS PIPELINE II=1
    constexpr int flat_dim = (N*...*Ns);
    for(int i=0; i<flat_dim; ++i) {
        out_stream.write(in_stream.read());
    }
}
