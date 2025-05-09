#ifndef FC1_H
#define FC1_H

extern "C" {
void fc2(
    const data_ap_fixed_t* in_data,
    data_ap_fixed_t* out_data,
    const data_ap_fixed_t* weight,
    const data_ap_fixed_t* bias,
    bool use_relu
);
}
#endif
