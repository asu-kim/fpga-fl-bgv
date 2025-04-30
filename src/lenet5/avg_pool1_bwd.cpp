#include "lenet5/avg_pool_bwd.h"
#include "lenet5/avg_pool1_bwd.h"

extern "C" {
    void avg_pool1_bwd(
        const float* grads,
        float* dX
    ) { 
        #pragma HLS INTERFACE m_axi port=grads bundle=gmem0 depth=6*(24/2)*(24/2)
        #pragma HLS INTERFACE m_axi port=dX bundle=gmem1 depth=6*24*24

        avg_pool_backward<2, 2, 6, 24, 24>(grads, dX);
    }
}