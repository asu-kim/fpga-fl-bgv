#ifndef AVG_POOL2_BWD_H
#define AVG_POOL2_BWD_H

extern "C" {
void avg_pool2_bwd(
    const data_ap_fixed_t* grads,
    data_ap_fixed_t* dX
);
}
#endif
