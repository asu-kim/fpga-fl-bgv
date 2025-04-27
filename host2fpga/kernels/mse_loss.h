#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include <hls_stream.h>
#include <stdint.h>

/*
 * loss = (1/n) sum of square y_true[i] - y_pred[i]
 * grad = 2 * (y_true - y_pred) * (1/n)
 */
template<int N>
void mse_loss(
        hls::stream<float> &y_pred,
        hls::stream<float> &y_true,
        hls::stream<float> &loss,
        hls::stream<float> &grads
        ) {
#pragma HLS PIPELINE
    float acc_loss = 0.0f;

    for(int i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
        float p = y_pred.read();
        float t = y_true.read();
        float diff = t - p;
        acc_loss += diff * diff;
        grads.write((2.0f * diff) / N);
    }

    loss.write(acc_loss / N);
}

#endif
