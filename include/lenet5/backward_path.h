#ifndef BACKWARD_PATH_H
#define BACKWARD_PATH_H

extern "C" {
void backward_path(
    const float* in_data,
    const float* conv1_weight,
    const float* conv1_bias,
    const float* conv1_out,
    const float* pool1_out,
    const float* conv2_weight,
    const float* conv2_bias,
    const float* conv2_out,
    const float* pool2_out,
    const float* fc1_weight,
    const float* fc1_bias,
    const float* fc1_out,
    const float* fc2_weight,
    const float* fc2_bias,
    const float* fc2_out,
    const float* fc3_weight,
    const float* fc3_bias,
    const float* fc3_out,
    const float* label,
    float* conv1_updated_weight,
    float* conv1_updated_bias,
    float* conv2_updated_weight,
    float* conv2_updated_bias,
    float* fc1_updated_weight,
    float* fc1_updated_bias,
    float* fc2_updated_weight,
    float* fc2_updated_bias,
    float* fc3_updated_weight,
    float* fc3_updated_bias,
    float loss
);
}
#endif
