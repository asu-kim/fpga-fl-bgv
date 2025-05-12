#ifndef AVG_POOL1_BWD_H
#define AVG_POOL1_BWD_H

extern "C" {
void avg_pool1_bwd(
    const data_ap_fixed_t* grads,
    data_ap_fixed_t* dX
);
}
#endif
