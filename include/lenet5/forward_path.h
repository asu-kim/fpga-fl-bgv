#ifndef FORWARD_PATH_H
#define FORWARD_PATH_H

extern "C" {
void forward_path(
    float* in_data,
    float* weights,       // Single array for all weights
    float* biases,        // Single array for all biases
    float* outs
);
}
#endif
