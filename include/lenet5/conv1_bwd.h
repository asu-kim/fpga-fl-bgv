#ifndef CONV1_BWD_H
#define CONV1_BWD_H

extern "C" {
void conv1_bwd(
    const data_ap_fixed_t* in_activation,
    const data_ap_fixed_t* grads,
    const data_ap_fixed_t* in_weight,
    data_ap_fixed_t* out_grads,
    data_ap_fixed_t* dW,
    data_ap_fixed_t* dB
);
}
#endif
