#ifndef FC1_BWD_H
#define FC1_BWD_H

extern "C" {
void fc1_bwd(
    const data_ap_fixed_t in_activation[120],
    const data_ap_fixed_t grads[256],
    const data_ap_fixed_t in_weight[256*120],
    data_ap_fixed_t dX[120],
    data_ap_fixed_t dW[256*120],
    data_ap_fixed_t dB[256]
);
}
#endif
