#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "data_type.hpp"

/*
 * loss = (1/n) sum of square y_true[i] - y_pred[i]
 * grad = 2 * (y_true - y_pred) * (1/n)
 */
template<int N>
void mse_loss(
        const data_ap_fixed_t y_pred[N],
        const data_ap_fixed_t y_true[N],
        data_ap_fixed_t &loss,
        data_ap_fixed_t grads[N]
        ) {
    data_ap_fixed_t acc_loss = 0.0f;

    for(int i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
        data_ap_fixed_t p = y_pred[i];
        data_ap_fixed_t t = y_true[i];
        data_ap_fixed_t diff = t - p;
        acc_loss += diff * diff;
        grads[i] = (data_ap_fixed_t(2.0) * diff) / N;
    }

    loss = (acc_loss / N);
}

#endif
