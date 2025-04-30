#ifndef CONV1_BWD_H
#define CONV1_BWD_H

extern "C" {
void conv1_bwd(
    const float* in_activation,
    const float* grads,
    const float* in_weight,
    float* out_grads,
    float* dW,
    float* dB
);
}
#endif
