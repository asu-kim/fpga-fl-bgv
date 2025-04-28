#include "lenet5/forward_path.h"
#include "lenet5/conv2d.h"
#include "lenet5/avg_pool.h"
#include "lenet5/fc_layer.h"

extern "C" {
void forward_path(
    float* in_data,
    float* conv1_weight,
    float* conv1_bias,
    float* conv1_out,
    float* pool1_out,
    float* conv2_weight,
    float* conv2_bias,
    float* conv2_out,
    float* pool2_out,
    float* fc1_weight,
    float* fc1_bias,
    float* fc1_out,
    float* fc2_weight,
    float* fc2_bias,
    float* fc2_out,  
    float* fc3_weight,
    float* fc3_bias,
    float* fc3_out
    ) {
    // Input data
    #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784

    // Layer 1: Conv1
    #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem1 depth=150
    #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem2 depth=6
    #pragma HLS INTERFACE m_axi port=conv1_out bundle=gmem3 depth=3456

    // Layer 2: Pool1
    #pragma HLS INTERFACE m_axi port=pool1_out bundle=gmem4 depth=864

    // Layer 3: Conv2
    #pragma HLS INTERFACE m_axi port=conv2_weight bundle=gmem5 depth=2400
    #pragma HLS INTERFACE m_axi port=conv2_bias bundle=gmem6 depth=16
    #pragma HLS INTERFACE m_axi port=conv2_out bundle=gmem7 depth=1024

    // Layer 4: Pool2
    #pragma HLS INTERFACE m_axi port=pool2_out bundle=gmem8 depth=256

    // Layer 5: FC1
    #pragma HLS INTERFACE m_axi port=fc1_weight bundle=gmem9 depth=30720
    #pragma HLS INTERFACE m_axi port=fc1_bias bundle=gmem10 depth=120
    #pragma HLS INTERFACE m_axi port=fc1_out bundle=gmem11 depth=120

    // Layer 6: FC2
    #pragma HLS INTERFACE m_axi port=fc2_weight bundle=gmem12 depth=10080
    #pragma HLS INTERFACE m_axi port=fc2_bias bundle=gmem13 depth=84
    #pragma HLS INTERFACE m_axi port=fc2_out bundle=gmem14 depth=84

    // Layer 7: FC3
    #pragma HLS INTERFACE m_axi port=fc3_weight bundle=gmem15 depth=840
    #pragma HLS INTERFACE m_axi port=fc3_bias bundle=gmem16 depth=10
    #pragma HLS INTERFACE m_axi port=fc3_out bundle=gmem17 depth=10

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
    for(int i = 0; i < 784; i++) {
        #pragma HLS PIPELINE II=1
        local_in_data[i] = in_data[i];
    }
    
    // Copy Conv1's weights and biases to local memory
    for(int i = 0; i < 150; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_weight[i] = conv1_weight[i];
    }
    for(int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_bias[i] = conv1_bias[i];
    }

    conv2d<6, 1, 5, 28, 28>(local_in_data, local_conv1_out, local_conv1_weight, local_conv1_bias);
    for(int i = 0; i < 3456; i++) {
        #pragma HLS PIPELINE II=1
        conv1_out[i] = local_conv1_out[i];
        // conv1_out[i] = 0;
    }

    avg_pool<2, 2, 6, 24, 24>(local_conv1_out, local_pool1_out);
    for(int i = 0; i < 864; i++) {
        #pragma HLS PIPELINE II=1
        pool1_out[i] = local_pool1_out[i];
        // pool1_out[i] = 0;
    }
    
    // Copy Conv2's weights and biases to local memory
    for(int i = 0; i < 2400; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_weight[i] = conv2_weight[i];
    }
    for(int i = 0; i < 16; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_bias[i] = conv2_bias[i];
    }

    conv2d<16, 6, 5, 12, 12>(local_pool1_out, local_conv2_out, local_conv2_weight, local_conv2_bias);
    for(int i = 0; i < 1024; i++) {
        #pragma HLS PIPELINE II=1
        conv2_out[i] = local_conv2_out[i];
        // conv2_out[i] = 0;
    }

    avg_pool<2, 2, 16, 8, 8>(local_conv2_out, local_pool2_out);
    for(int i = 0; i < 256; i++) {
        #pragma HLS PIPELINE II=1
        pool2_out[i] = local_pool2_out[i];
        // pool2_out[i] = 0;
    }
    
    // Copy FC1's weights and biases to local memory
    for(int i = 0; i < 30720; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_weight[i] = fc1_weight[i];
    }
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_bias[i] = fc1_bias[i];
    }

    fc<256, 120>(local_pool2_out, local_fc1_out, local_fc1_weight, local_fc1_bias, true);
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        fc1_out[i] = local_fc1_out[i];
        // fc1_out[i] = 0;
    }
    
    // Copy FC2's weights and biases to local memory
    for(int i = 0; i < 10080; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_weight[i] = fc2_weight[i];
    }
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_bias[i] = fc2_bias[i];
    }

    fc<120, 84>(local_fc1_out, local_fc2_out, local_fc2_weight, local_fc2_bias, true);
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        fc2_out[i] = local_fc2_out[i];
        // fc2_out[i] = 0;
    }
    
    // Copy FC3's weights and biases to local memory
    for(int i = 0; i < 840; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_weight[i] = fc3_weight[i];
    }
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_bias[i] = fc3_bias[i];
    }

    fc<84, 10>(local_fc2_out, local_fc3_out, local_fc3_weight, local_fc3_bias, false);
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        fc3_out[i] = local_fc3_out[i];
        // fc3_out[i] = 0;
    }
}
}