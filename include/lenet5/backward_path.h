#ifndef BACKWARD_PATH_H
#define BACKWARD_PATH_H

extern "C" {
void backward_path(
    const float* in_data,             // gmem0
    const float* weights,             // gmem1 - consolidated weights
    const float* biases,              // gmem2 - consolidated biases
    const float* outputs,             // gmem3 - consolidated outputs
    const float* label,               // gmem4
    float* updated_weights,           // gmem5 - consolidated updated weights
    float* updated_biases,            // gmem6 - consolidated updated biases
    float& loss
);
}
#endif
