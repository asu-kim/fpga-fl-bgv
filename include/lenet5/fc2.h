#ifndef FC1_H
#define FC1_H

extern "C" {
void fc2(
    const float* in_data,
    float* out_data,
    const float* weight,
    const float* bias,
    bool use_relu
);
}
#endif
