#ifndef FC1_BWD_H
#define FC1_BWD_H

extern "C" {
void fc1_bwd(
    const float in_activation[120],
    const float grads[256],
    const float in_weight[256*120],
    float dX[120],
    float dW[256*120],
    float dB[256]
);
}
#endif
