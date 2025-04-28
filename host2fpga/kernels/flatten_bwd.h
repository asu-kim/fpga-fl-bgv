#ifndef FLATTEN_BWD_H
#define FLATTEN_BWD_H

#include <hls_stream.h>
#include <stdint.h>

// just pass back whatever we get from next layer to the previous one
template<int OC, int IC, int H, int W>
void flatten_backward(
        const float grads[OC*IC*H*W],
        float dX[OC*IC*H*W]
        ) {
    int dim = OC * IC * H * W;
    for(int i=0; i<dim; ++i) {
        dX[i] = grads[i];
    }
}

#endif
