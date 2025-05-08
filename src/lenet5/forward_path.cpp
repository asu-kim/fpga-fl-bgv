#include "lenet5/forward_path.h"
#include "lenet5/conv2d.h"
#include "lenet5/avg_pool.h"
#include "lenet5/fc_layer.h"
#include "constants.hpp"

extern "C" {
void forward_path(
    float* in_data,
    float* weights,       // Single array for all weights
    float* biases,        // Single array for all biases
    float* outs
    ) {
    // Input data
    #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784

    // Combined weights and biases
    #pragma HLS INTERFACE m_axi port=weights bundle=gmem1 depth=TOTAL_WEIGHTS_SIZE
    #pragma HLS INTERFACE m_axi port=biases bundle=gmem2 depth=TOTAL_BIASES_SIZE

    #pragma HLS INTERFACE m_axi port=outs bundle=gmem3 depth=TOTAL_OUTS_SIZE

    // Controls
    #pragma HLS INTERFACE s_axilite port=in_data bundle=control
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=biases bundle=control
    #pragma HLS INTERFACE s_axilite port=outs bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

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
    for(int i = 0; i < 784; i++) {
        #pragma HLS PIPELINE II=1
        local_in_data[i] = in_data[i];
    }
    
    // Copy Conv1's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < 150; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_weight[i] = weights[CONV1_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_bias[i] = biases[CONV1_BIAS_OFFSET + i];
    }

    conv2d<6, 1, 5, 28, 28>(local_in_data, local_conv1_out, local_conv1_weight, local_conv1_bias);

    avg_pool<2, 2, 6, 24, 24>(local_conv1_out, local_pool1_out);
    
    // Copy Conv2's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < 2400; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_weight[i] = weights[CONV2_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 16; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_bias[i] = biases[CONV2_BIAS_OFFSET + i];
    }
    conv2d<16, 6, 5, 12, 12>(local_pool1_out, local_conv2_out, local_conv2_weight, local_conv2_bias);

    avg_pool<2, 2, 16, 8, 8>(local_conv2_out, local_pool2_out);
    
    // Copy FC1's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < 30720; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_weight[i] = weights[FC1_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_bias[i] = biases[FC1_BIAS_OFFSET + i];
    }
    fc<256, 120>(local_pool2_out, local_fc1_out, local_fc1_weight, local_fc1_bias, true);
    
    // Copy FC2's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < 10080; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_weight[i] = weights[FC2_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_bias[i] = biases[FC2_BIAS_OFFSET + i];
    }
    fc<120, 84>(local_fc1_out, local_fc2_out, local_fc2_weight, local_fc2_bias, true);
    
    // Copy FC3's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < 840; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_weight[i] = weights[FC3_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_bias[i] = biases[FC3_BIAS_OFFSET + i];
    }
    fc<84, 10>(local_fc2_out, local_fc3_out, local_fc3_weight, local_fc3_bias, false);

    for(int i = 0; i < NUM_CONV1_OUTS; i++) {
        #pragma HLS PIPELINE II=1
        outs[CONV1_OUT_OFFSET + i] = local_conv1_out[i];
    }

    for(int i = 0; i < NUM_POOL1_OUTS; i++) {
        #pragma HLS PIPELINE II=1
        outs[POOL1_OUT_OFFSET + i] = local_pool1_out[i];
    }

    for(int i = 0; i < NUM_CONV2_OUTS; i++) {
        #pragma HLS PIPELINE II=1
        outs[CONV2_OUT_OFFSET + i] = local_conv2_out[i];
    }

    for(int i = 0; i < NUM_POOL2_OUTS; i++) {
        #pragma HLS PIPELINE II=1
        outs[POOL2_OUT_OFFSET + i] = local_pool2_out[i];
    }

    for(int i = 0; i < NUM_FC1_OUTS; i++) {
        #pragma HLS PIPELINE II=1
        outs[FC1_OUT_OFFSET + i] = local_fc1_out[i];
    }

    for(int i = 0; i < NUM_FC2_OUTS; i++) {
        #pragma HLS PIPELINE II=1
        outs[FC2_OUT_OFFSET + i] = local_fc2_out[i];
    }

    for(int i = 0; i < NUM_FC3_OUTS; i++) {
        #pragma HLS PIPELINE II=1
        outs[FC3_OUT_OFFSET + i] = local_fc3_out[i];
    }
}
}