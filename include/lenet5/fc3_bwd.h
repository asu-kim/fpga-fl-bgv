#ifndef FC3_BWD_H
#define FC3_BWD_H

extern "C" {
void fc3_bwd(
    const data_ap_fixed_t in_activation[10],
    const data_ap_fixed_t grads[84],
    const data_ap_fixed_t in_weight[84*10],
    data_ap_fixed_t dX[10],
    data_ap_fixed_t dW[84*10],
    data_ap_fixed_t dB[84]
);
}
#endif
