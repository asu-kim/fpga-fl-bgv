#ifndef TRAIN_H
#define TRAIN_H

extern "C" {
    void train_lenet5_top(
            const float *image,
            const float *arg_conv1_weight,
            const float *arg_conv1_bias,
            const float *arg_conv2_weight,
            const float *arg_conv2_bias,
            const float *arg_fc1_weight,
            const float *arg_fc1_bias,
            const float *arg_fc2_weight,
            const float *arg_fc2_bias,
            const float *arg_fc3_weight,
            const float *arg_fc3_bias,
            float *logits,
            const float *label
            );
}

#endif
