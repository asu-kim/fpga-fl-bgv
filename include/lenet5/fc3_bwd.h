#ifndef FC3_BWD_H
#define FC3_BWD_H

extern "C" {
void fc3_bwd(
    const float in_activation[10],
    const float grads[84],
    const float in_weight[84*10],
    float dX[10],
    float dW[84*10],
    float dB[84]
);
}
#endif
