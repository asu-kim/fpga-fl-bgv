#ifndef FORWARD_PATH_H
#define FORWARD_PATH_H

extern "C" {
void forward_path(
    float* in_data,
    float* conv1_out,
    float* pool1_out,
    float* conv2_out,
    float* pool2_out,
    float* fc1_out,
    float* fc2_out,
    float* fc3_out,
    float* conv1_weight,
    float* conv1_bias,
    float* conv2_weight,
    float* conv2_bias,
    float* fc1_weight,
    float* fc1_bias,
    float* fc2_weight,
    float* fc2_bias,
    float* fc3_weight,
    float* fc3_bias,
    float* probs
);
}
#endif
