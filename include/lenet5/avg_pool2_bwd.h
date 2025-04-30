#ifndef AVG_POOL2_BWD_H
#define AVG_POOL2_BWD_H

extern "C" {
void avg_pool2_bwd(
    const float* grads,
    float* dX
);
}
#endif
