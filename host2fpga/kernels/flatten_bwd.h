#ifndef FLATTEN_BWD_H
#define FLATTEN_BWD_H

#include <hls_stream.h>
#include <stdint.h>

// just pass back whatever we get from next layer to the previous one
template<int... DIMs>
void flatten_backward(hls::stream<float>&grads, hls::stream<float>& out_stream) {
    const int flat_dims = (1 * ... * DIMs);
    for(int i=0; i<flat_dims; ++i) {
        out_stream.write(grads.read());
    }
}

#endif
