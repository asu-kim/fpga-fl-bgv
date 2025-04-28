#include <string.h>

#include "lenet5/forward_path.h"
#include "lenet5/conv2d.h"
#include "lenet5/avg_pool.h"
#include "lenet5/fc_layer.h"

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
        float* fc3_bias
    ) {
    // Input
    #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784 // 1*28*28

    // Intermediate feature maps (outputs of layers)
    #pragma HLS INTERFACE m_axi port=conv1_out bundle=gmem1 depth=3456 // 6*24*24
    #pragma HLS INTERFACE m_axi port=pool1_out bundle=gmem2 depth=864  // 6*12*12
    #pragma HLS INTERFACE m_axi port=conv2_out bundle=gmem3 depth=1024 // 16*8*8
    #pragma HLS INTERFACE m_axi port=pool2_out bundle=gmem4 depth=256  // 16*4*4

    // Fully connected layer outputs
    #pragma HLS INTERFACE m_axi port=fc1_out bundle=gmem5 depth=120
    #pragma HLS INTERFACE m_axi port=fc2_out bundle=gmem6 depth=84
    #pragma HLS INTERFACE m_axi port=fc3_out bundle=gmem7 depth=10

    // Weights and Biases
    #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem8 depth=150 // 6*5*5
    #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem9 depth=6

    #pragma HLS INTERFACE m_axi port=conv2_weight bundle=gmem10 depth=2400 // 16*5*5
    #pragma HLS INTERFACE m_axi port=conv2_bias bundle=gmem11 depth=16

    #pragma HLS INTERFACE m_axi port=fc1_weight bundle=gmem12 depth=30720 // 256*120
    #pragma HLS INTERFACE m_axi port=fc1_bias bundle=gmem13 depth=120

    #pragma HLS INTERFACE m_axi port=fc2_weight bundle=gmem14 depth=10080 // 120*84
    #pragma HLS INTERFACE m_axi port=fc2_bias bundle=gmem15 depth=84

    #pragma HLS INTERFACE m_axi port=fc3_weight bundle=gmem16 depth=840 // 84*10
    #pragma HLS INTERFACE m_axi port=fc3_bias bundle=gmem17 depth=10

    // Controls
    #pragma HLS INTERFACE s_axilite port=in_data bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_out bundle=control
    #pragma HLS INTERFACE s_axilite port=pool1_out bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_out bundle=control
    #pragma HLS INTERFACE s_axilite port=pool2_out bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_out bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_out bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_out bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // #pragma HLS DATAFLOW
    // Local memory for each stage input and output
    float local_in_data[1*28*28];
    float local_conv1_out[6*24*24];
    float local_pool1_out[6*12*12];
    float local_conv2_out[16*8*8];
    float local_pool2_out[16*4*4];
    float local_fc1_out[120];
    float local_fc2_out[84];
    float local_fc3_out[10];
    
    // Local memory for weights and biases
    float local_conv1_weight[6*1*5*5];
    float local_conv1_bias[6];
    float local_conv2_weight[16*6*5*5];
    float local_conv2_bias[16];
    float local_fc1_weight[256*120];
    float local_fc1_bias[120];
    float local_fc2_weight[120*84];
    float local_fc2_bias[84];
    float local_fc3_weight[84*10];
    float local_fc3_bias[10];
    
    // Copy input data to local memory
    memcpy(local_in_data, in_data, sizeof(float)*784);
    
    // Copy weights and biases to local memory
    memcpy(local_conv1_weight, conv1_weight, sizeof(float)*6*1*5*5);
    memcpy(local_conv1_bias, conv1_bias, sizeof(float)*6);
    memcpy(local_conv2_weight, conv2_weight, sizeof(float)*16*6*5*5);
    memcpy(local_conv2_bias, conv2_bias, sizeof(float)*16);
    memcpy(local_fc1_weight, fc1_weight, sizeof(float)*256*120);
    memcpy(local_fc1_bias, fc1_bias, sizeof(float)*120);
    memcpy(local_fc2_weight, fc2_weight, sizeof(float)*120*84);
    memcpy(local_fc2_bias, fc2_bias, sizeof(float)*84);
    memcpy(local_fc3_weight, fc3_weight, sizeof(float)*84*10);
    memcpy(local_fc3_bias, fc3_bias, sizeof(float)*10);
    
    // Execute the network with local memory
    conv2d<6, 1, 5, 28, 28>(local_in_data, local_conv1_out, local_conv1_weight, local_conv1_bias);
    memcpy(conv1_out, local_conv1_out, sizeof(float)*6*24*24);

    avg_pool<2, 2, 6, 24, 24>(local_conv1_out, local_pool1_out);
    memcpy(pool1_out, local_pool1_out, sizeof(float)*6*12*12);

    conv2d<16, 6, 5, 12, 12>(local_pool1_out, local_conv2_out, local_conv2_weight, local_conv2_bias);
    memcpy(conv2_out, local_conv2_out, sizeof(float)*16*8*8);

    avg_pool<2, 2, 16, 8, 8>(local_conv2_out, local_pool2_out);
    memcpy(pool2_out, local_pool2_out, sizeof(float)*16*4*4);

    fc<256, 120>(local_pool2_out, local_fc1_out, local_fc1_weight, local_fc1_bias);
    memcpy(fc1_out, local_fc1_out, sizeof(float)*120);

    fc<120, 84>(local_fc1_out, local_fc2_out, local_fc2_weight, local_fc2_bias);
    memcpy(fc2_out, local_fc2_out, sizeof(float)*84);

    fc<84, 10>(local_fc2_out, local_fc3_out, local_fc3_weight, local_fc3_bias);
    memcpy(fc3_out, local_fc3_out, sizeof(float)*10);
}
}