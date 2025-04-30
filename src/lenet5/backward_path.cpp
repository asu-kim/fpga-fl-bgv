#include "lenet5/backward_path.h"
#include "lenet5/conv2d_bwd.h"
#include "lenet5/avg_pool_bwd.h"
#include "lenet5/fc_bwd.h"
#include "lenet5/mse_loss.h"

#define lr 1e-3

extern "C" {
void backward_path(
    const float* in_data,             // gmem0
    const float* conv1_weight,        // gmem1
    const float* conv1_bias,          // gmem2
    const float* conv1_out,           // gmem3
    const float* pool1_out,           // gmem4
    const float* conv2_weight,        // gmem5
    const float* conv2_bias,          // gmem6
    const float* conv2_out,           // gmem7
    const float* pool2_out,           // gmem8
    const float* fc1_weight,          // gmem9
    const float* fc1_bias,            // gmem10
    const float* fc1_out,             // gmem11
    const float* fc2_weight,          // gmem12
    const float* fc2_bias,            // gmem13
    const float* fc2_out,             // gmem14
    const float* fc3_weight,          // gmem15
    const float* fc3_bias,            // gmem16
    const float* fc3_out,             // gmem17
    const float* label,               // gmem18
    float* conv1_updated_weight,      // gmem19
    float* conv1_updated_bias,        // gmem20
    float* conv2_updated_weight,      // gmem21
    float* conv2_updated_bias,        // gmem22
    float* fc1_updated_weight,        // gmem23
    float* fc1_updated_bias,          // gmem24
    float* fc2_updated_weight,        // gmem25
    float* fc2_updated_bias,          // gmem26
    float* fc3_updated_weight,        // gmem27
    float* fc3_updated_bias,          // gmem28
    float loss
) {
    // Input data
    #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784

    // Layer 1: Conv1
    #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem1 depth=150
    #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem2 depth=6
    #pragma HLS INTERFACE m_axi port=conv1_out bundle=gmem3 depth=3456
    #pragma HLS INTERFACE m_axi port=pool1_out bundle=gmem4 depth=864

    // Layer 2: Conv2
    #pragma HLS INTERFACE m_axi port=conv2_weight bundle=gmem5 depth=2400
    #pragma HLS INTERFACE m_axi port=conv2_bias bundle=gmem6 depth=16
    #pragma HLS INTERFACE m_axi port=conv2_out bundle=gmem7 depth=1024
    #pragma HLS INTERFACE m_axi port=pool2_out bundle=gmem8 depth=256

    // Layer 3: FC1
    #pragma HLS INTERFACE m_axi port=fc1_weight bundle=gmem9 depth=30720
    #pragma HLS INTERFACE m_axi port=fc1_bias bundle=gmem10 depth=120
    #pragma HLS INTERFACE m_axi port=fc1_out bundle=gmem11 depth=120

    // Layer 4: FC2
    #pragma HLS INTERFACE m_axi port=fc2_weight bundle=gmem12 depth=10080
    #pragma HLS INTERFACE m_axi port=fc2_bias bundle=gmem13 depth=84
    #pragma HLS INTERFACE m_axi port=fc2_out bundle=gmem14 depth=84

    // Layer 5: FC3
    #pragma HLS INTERFACE m_axi port=fc3_weight bundle=gmem15 depth=840
    #pragma HLS INTERFACE m_axi port=fc3_bias bundle=gmem16 depth=10
    #pragma HLS INTERFACE m_axi port=fc3_out bundle=gmem17 depth=10
    #pragma HLS INTERFACE m_axi port=label bundle=gmem18 depth=10

    // Updated weights and biases
    #pragma HLS INTERFACE m_axi port=conv1_updated_weight bundle=gmem19 depth=150
    #pragma HLS INTERFACE m_axi port=conv1_updated_bias bundle=gmem20 depth=6
    #pragma HLS INTERFACE m_axi port=conv2_updated_weight bundle=gmem21 depth=2400
    #pragma HLS INTERFACE m_axi port=conv2_updated_bias bundle=gmem22 depth=16
    #pragma HLS INTERFACE m_axi port=fc1_updated_weight bundle=gmem23 depth=30720
    #pragma HLS INTERFACE m_axi port=fc1_updated_bias bundle=gmem24 depth=120
    #pragma HLS INTERFACE m_axi port=fc2_updated_weight bundle=gmem25 depth=10080
    #pragma HLS INTERFACE m_axi port=fc2_updated_bias bundle=gmem26 depth=84
    #pragma HLS INTERFACE m_axi port=fc3_updated_weight bundle=gmem27 depth=840
    #pragma HLS INTERFACE m_axi port=fc3_updated_bias bundle=gmem28 depth=10

    // Controls for all parameters
    #pragma HLS INTERFACE s_axilite port=in_data bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_out bundle=control
    #pragma HLS INTERFACE s_axilite port=pool1_out bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_out bundle=control
    #pragma HLS INTERFACE s_axilite port=pool2_out bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_out bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_out bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_out bundle=control
    #pragma HLS INTERFACE s_axilite port=label bundle=control
    
    #pragma HLS INTERFACE s_axilite port=conv1_updated_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=conv1_updated_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_updated_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=conv2_updated_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_updated_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc1_updated_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_updated_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc2_updated_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_updated_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=fc3_updated_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=loss bundle=control

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
    
    // Local memory for updated weights and biases
    float local_updated_conv1_weight[6*1*5*5];
    float local_updated_conv1_bias[6];
    float local_updated_conv2_weight[16*6*5*5];
    float local_updated_conv2_bias[16];
    float local_updated_fc1_weight[256*120];
    float local_updated_fc1_bias[120];
    float local_updated_fc2_weight[120*84];
    float local_updated_fc2_bias[84];
    float local_updated_fc3_weight[84*10];
    float local_updated_fc3_bias[10];
    

    // FC3 Backward
    float out_grad[10];
    // Copy FC3's weights and biases to local memory
    for(int i = 0; i < 84*10; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_weight[i] = fc3_weight[i];
    }
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_bias[i] = fc3_bias[i];
    }
    mse_loss<10>(fc3_out, label, loss, out_grad);

    float fc3_dX[84];
    float fc3_dW[84*10];
    float fc3_dB[10];
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_out[i] = fc2_out[i];
    }
    fc_backward<84, 10>(local_fc2_out, out_grad, local_fc3_weight, fc3_dX, fc3_dW, fc3_dB, false);
    for(int i = 0; i < 84*10; i++) {
        #pragma HLS PIPELINE II=1
        fc3_updated_weight[i] = local_fc3_weight[i] - lr * fc3_dW[i];
    }
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        fc3_updated_bias[i] = local_fc3_bias[i] - lr * fc3_dB[i];
    }

    // FC2 Backward
    float fc2_dX[120];
    float fc2_dW[120*84];
    float fc2_dB[84];
    // Copy FC2's weights and biases to local memory
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_out[i] = fc1_out[i];
    }
    for(int i = 0; i < 120*84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_weight[i] = fc2_weight[i];
    }
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_bias[i] = fc2_bias[i];
    }
    fc_backward<120, 84>(local_fc1_out, fc3_dX, local_fc2_weight, fc2_dX, fc2_dW, fc2_dB, true);
    for(int i = 0; i < 120*84; i++) {
        #pragma HLS PIPELINE II=1
        fc2_updated_weight[i] = local_fc2_weight[i] - lr * fc2_dW[i];
    }
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        fc2_updated_bias[i] = local_fc2_bias[i] - lr * fc2_dB[i];
    }

    // FC1 Backward
    float fc1_dX[256];
    float fc1_dW[256*120];
    float fc1_dB[120];
    // Copy FC1's weights and biases to local memory
    for(int i = 0; i < 256; i++) {
        #pragma HLS PIPELINE II=1
        local_pool2_out[i] = pool2_out[i];
    }
    for(int i = 0; i < 256*120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_weight[i] = fc1_weight[i];
    }
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_bias[i] = fc1_bias[i];
    }
    fc_backward<256, 120>(local_pool2_out, fc2_dX, local_fc1_weight, fc1_dX, fc1_dW, fc1_dB, true);
    for(int i = 0; i < 256*120; i++) {
        #pragma HLS PIPELINE II=1
        fc1_updated_weight[i] = local_fc1_weight[i] - lr * fc1_dW[i];
    }
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        fc1_updated_bias[i] = local_fc1_bias[i] - lr * fc1_dB[i];
    }

    // Pool2 Backward
    float pool2_dX[16*8*8];
    avg_pool_backward<2, 2, 16, 8, 8>(fc1_dX, pool2_dX);
    
    // Conv2 Backward
    float conv2_dX[6*12*12];
    float conv2_dW[16*6*5*5];
    float conv2_dB[16];
    // Copy Conv2's weights and biases to local memory
    for(int i = 0; i < 6*12*12; i++) {
        #pragma HLS PIPELINE II=1
        local_pool1_out[i] = pool1_out[i];
    }
    for(int i = 0; i < 16*6*5*5; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_weight[i] = conv2_weight[i];
    }
    for(int i = 0; i < 16; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_bias[i] = conv2_bias[i];
    }
    conv2d_backward<16, 6, 5, 12, 12>(local_pool1_out, pool2_dX, local_conv2_weight, conv2_dX, conv2_dW, conv2_dB);
    for(int i = 0; i < 16*6*5*5; i++) {
        #pragma HLS PIPELINE II=1
        conv2_updated_weight[i] = local_conv2_weight[i] - lr * conv2_dW[i];
    }
    for(int i = 0; i < 16; i++) {
        #pragma HLS PIPELINE II=1
        conv2_updated_bias[i] = local_conv2_bias[i] - lr * conv2_dB[i];
    }

    // Pool1 Backward
    float pool1_dX[6*24*24];
    avg_pool_backward<2, 2, 6, 24, 24>(conv2_dX, pool1_dX);
    
    // Conv1 Backward
    float conv1_dX[1*28*28];
    float conv1_dW[6*1*5*5];
    float conv1_dB[6];
    // Copy Conv1's weights and biases to local memory
    for(int i = 0; i < 1*28*28; i++) {
        #pragma HLS PIPELINE II=1
        local_in_data[i] = in_data[i];
    }
    for(int i = 0; i < 6*1*5*5; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_weight[i] = conv1_weight[i];
    }
    for(int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_bias[i] = conv1_bias[i];
    }
    conv2d_backward<6, 1, 5, 28, 28>(local_in_data, pool1_dX, local_conv1_weight, conv1_dX, conv1_dW, conv1_dB);
    for(int i = 0; i < 6*1*5*5; i++) {
        #pragma HLS PIPELINE II=1
        conv1_updated_weight[i] = local_conv1_weight[i] - lr * conv1_dW[i];
    }
    for(int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=1
        conv1_updated_bias[i] = local_conv1_bias[i] - lr * conv1_dB[i];
    }
}
}