#include "lenet5/avg_pool_bwd.h"
#include "lenet5/avg_pool2_bwd.h"

extern "C" {
    void avg_pool2_bwd(
        const float* grads,
        float* dX
    ) { 
        #pragma HLS INTERFACE m_axi port=grads bundle=gmem0 depth=16*(8/2)*(8/2)
        #pragma HLS INTERFACE m_axi port=dX bundle=gmem1 depth=16*8*8

        avg_pool_backward<2, 2, 16, 8, 8>(grads, dX);
    }
}