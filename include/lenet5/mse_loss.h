#ifndef MSE_LOSS_H
#define MSE_LOSS_H

/*
 * loss = (1/n) sum of square y_true[i] - y_pred[i]
 * grad = 2 * (y_true - y_pred) * (1/n)
 */
template<int N>
void mse_loss(
        const float y_pred[N],
        const float y_true[N],
        float &loss,
        float grads[N]
        ) {
    float acc_loss = 0.0f;

    for(int i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
        float p = y_pred[i];
        float t = y_true[i];
        float diff = t - p;
        acc_loss += diff * diff;
        grads[i] = (2.0f * diff) / N;
    }

    loss = (acc_loss / N);
}

#endif
