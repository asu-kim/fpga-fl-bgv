#ifndef AVG_POOL1_BWD_H
#define AVG_POOL1_BWD_H

extern "C" {
void avg_pool1_bwd(
    const float* grads,
    float* dX
);
}
#endif
