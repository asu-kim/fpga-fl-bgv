#include "lenet5/backward_path.h"
#include "lenet5/conv2d_bwd.h"
#include "lenet5/avg_pool_bwd.h"
#include "lenet5/fc_bwd.h"
#include "lenet5/mse_loss.h"

#include "constants.hpp"

extern "C" {
void backward_path(
    const data_ap_fixed_t* in_data,             // gmem0
    const data_ap_fixed_t* weights,             // gmem1 - consolidated weights
    const data_ap_fixed_t* biases,              // gmem2 - consolidated biases
    const data_ap_fixed_t* outputs,             // gmem3 - consolidated outputs
    const data_ap_fixed_t* label,               // gmem4

    data_ap_fixed_t* updated_weights,           // gmem5 - consolidated updated weights
    data_ap_fixed_t* updated_biases,            // gmem6 - consolidated updated biases
    data_ap_fixed_t& loss
) {
    // Input data
    #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784

    // Combined weights, biases and outputs
    #pragma HLS INTERFACE m_axi port=weights bundle=gmem1 depth=TOTAL_WEIGHTS_SIZE
    #pragma HLS INTERFACE m_axi port=biases bundle=gmem2 depth=TOTAL_BIASES_SIZE
    #pragma HLS INTERFACE m_axi port=outputs bundle=gmem3 depth=TOTAL_OUTS_SIZE // Sum of all output sizes
    #pragma HLS INTERFACE m_axi port=label bundle=gmem4 depth=10

    // Combined updated weights and biases
    #pragma HLS INTERFACE m_axi port=updated_weights bundle=gmem5 depth=TOTAL_WEIGHTS_SIZE
    #pragma HLS INTERFACE m_axi port=updated_biases bundle=gmem6 depth=TOTAL_BIASES_SIZE

    // Controls
    #pragma HLS INTERFACE s_axilite port=in_data bundle=control
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=biases bundle=control
    #pragma HLS INTERFACE s_axilite port=outputs bundle=control
    #pragma HLS INTERFACE s_axilite port=label bundle=control
    #pragma HLS INTERFACE s_axilite port=updated_weights bundle=control
    #pragma HLS INTERFACE s_axilite port=updated_biases bundle=control
    #pragma HLS INTERFACE s_axilite port=loss bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local memory for each stage input and output
    data_ap_fixed_t local_in_data[1*28*28];
    data_ap_fixed_t local_conv1_out[6*24*24];
    data_ap_fixed_t local_pool1_out[6*12*12];
    data_ap_fixed_t local_conv2_out[16*8*8];
    data_ap_fixed_t local_pool2_out[16*4*4];
    data_ap_fixed_t local_fc1_out[120];
    data_ap_fixed_t local_fc2_out[84];
    data_ap_fixed_t local_fc3_out[10];
    
    // Local memory for weights and biases
    data_ap_fixed_t local_conv1_weight[6*1*5*5];
    data_ap_fixed_t local_conv1_bias[6];
    data_ap_fixed_t local_conv2_weight[16*6*5*5];
    data_ap_fixed_t local_conv2_bias[16];
    data_ap_fixed_t local_fc1_weight[256*120];
    data_ap_fixed_t local_fc1_bias[120];
    data_ap_fixed_t local_fc2_weight[120*84];
    data_ap_fixed_t local_fc2_bias[84];
    data_ap_fixed_t local_fc3_weight[84*10];
    data_ap_fixed_t local_fc3_bias[10];
    
    // FC3 Backward
    data_ap_fixed_t out_grad[10];
    // Copy FC3's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < 84*10; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_weight[i] = weights[FC3_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_bias[i] = biases[FC3_BIAS_OFFSET + i];
    }
    
    // Get FC3 output from consolidated outputs array
    data_ap_fixed_t fc3_output[10];
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        fc3_output[i] = outputs[FC3_OUT_OFFSET + i];
    }
    
    mse_loss<10>(fc3_output, label, loss, out_grad);

    data_ap_fixed_t fc3_dX[84];
    data_ap_fixed_t fc3_dW[84*10];
    data_ap_fixed_t fc3_dB[10];
    
    // Get FC2 output from consolidated outputs array
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_out[i] = outputs[FC2_OUT_OFFSET + i];
    }
    
    fc_backward<84, 10>(local_fc2_out, out_grad, local_fc3_weight, fc3_dX, fc3_dW, fc3_dB, false);
    
    // Store updated FC3 weights and biases to consolidated arrays
    for(int i = 0; i < 84*10; i++) {
        #pragma HLS PIPELINE II=1
        updated_weights[FC3_WEIGHT_OFFSET + i] = local_fc3_weight[i] - lr * fc3_dW[i];
    }
    for(int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        updated_biases[FC3_BIAS_OFFSET + i] = local_fc3_bias[i] - lr * fc3_dB[i];
    }

    // FC2 Backward
    data_ap_fixed_t fc2_dX[120];
    data_ap_fixed_t fc2_dW[120*84];
    data_ap_fixed_t fc2_dB[84];
    
    // Copy FC2's weights and biases from consolidated arrays
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_out[i] = outputs[FC1_OUT_OFFSET + i];
    }
    for(int i = 0; i < 120*84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_weight[i] = weights[FC2_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_bias[i] = biases[FC2_BIAS_OFFSET + i];
    }
    
    fc_backward<120, 84>(local_fc1_out, fc3_dX, local_fc2_weight, fc2_dX, fc2_dW, fc2_dB, true);
    
    // Store updated FC2 weights and biases to consolidated arrays
    for(int i = 0; i < 120*84; i++) {
        #pragma HLS PIPELINE II=1
        updated_weights[FC2_WEIGHT_OFFSET + i] = local_fc2_weight[i] - lr * fc2_dW[i];
    }
    for(int i = 0; i < 84; i++) {
        #pragma HLS PIPELINE II=1
        updated_biases[FC2_BIAS_OFFSET + i] = local_fc2_bias[i] - lr * fc2_dB[i];
    }

    // FC1 Backward
    data_ap_fixed_t fc1_dX[256];
    data_ap_fixed_t fc1_dW[256*120];
    data_ap_fixed_t fc1_dB[120];
    
    // Copy FC1's weights and biases from consolidated arrays
    for(int i = 0; i < 256; i++) {
        #pragma HLS PIPELINE II=1
        local_pool2_out[i] = outputs[POOL2_OUT_OFFSET + i];
    }
    for(int i = 0; i < 256*120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_weight[i] = weights[FC1_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_bias[i] = biases[FC1_BIAS_OFFSET + i];
    }
    
    fc_backward<256, 120>(local_pool2_out, fc2_dX, local_fc1_weight, fc1_dX, fc1_dW, fc1_dB, true);
    
    // Store updated FC1 weights and biases to consolidated arrays
    for(int i = 0; i < 256*120; i++) {
        #pragma HLS PIPELINE II=1
        updated_weights[FC1_WEIGHT_OFFSET + i] = local_fc1_weight[i] - lr * fc1_dW[i];
    }
    for(int i = 0; i < 120; i++) {
        #pragma HLS PIPELINE II=1
        updated_biases[FC1_BIAS_OFFSET + i] = local_fc1_bias[i] - lr * fc1_dB[i];
    }

    // Pool2 Backward
    data_ap_fixed_t pool2_dX[16*8*8];
    avg_pool_backward<2, 2, 16, 8, 8>(fc1_dX, pool2_dX);
    
    // Conv2 Backward
    data_ap_fixed_t conv2_dX[6*12*12];
    data_ap_fixed_t conv2_dW[16*6*5*5];
    data_ap_fixed_t conv2_dB[16];
    
    // Copy Conv2's weights and biases from consolidated arrays
    for(int i = 0; i < 6*12*12; i++) {
        #pragma HLS PIPELINE II=1
        local_pool1_out[i] = outputs[POOL1_OUT_OFFSET + i];
    }
    for(int i = 0; i < 16*6*5*5; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_weight[i] = weights[CONV2_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 16; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_bias[i] = biases[CONV2_BIAS_OFFSET + i];
    }
    
    conv2d_backward<16, 6, 5, 12, 12>(local_pool1_out, pool2_dX, local_conv2_weight, conv2_dX, conv2_dW, conv2_dB);
    
    // Store updated Conv2 weights and biases to consolidated arrays
    for(int i = 0; i < 16*6*5*5; i++) {
        #pragma HLS PIPELINE II=1
        updated_weights[CONV2_WEIGHT_OFFSET + i] = local_conv2_weight[i] - lr * conv2_dW[i];
    }
    for(int i = 0; i < 16; i++) {
        #pragma HLS PIPELINE II=1
        updated_biases[CONV2_BIAS_OFFSET + i] = local_conv2_bias[i] - lr * conv2_dB[i];
    }

    // Pool1 Backward
    data_ap_fixed_t pool1_dX[6*24*24];
    avg_pool_backward<2, 2, 6, 24, 24>(conv2_dX, pool1_dX);
    
    // Conv1 Backward
    data_ap_fixed_t conv1_dX[1*28*28];
    data_ap_fixed_t conv1_dW[6*1*5*5];
    data_ap_fixed_t conv1_dB[6];
    
    // Copy Conv1's weights and biases from consolidated arrays
    for(int i = 0; i < 1*28*28; i++) {
        #pragma HLS PIPELINE II=1
        local_in_data[i] = in_data[i];
    }
    for(int i = 0; i < 6*1*5*5; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_weight[i] = weights[CONV1_WEIGHT_OFFSET + i];
    }
    for(int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_bias[i] = biases[CONV1_BIAS_OFFSET + i];
    }
    
    conv2d_backward<6, 1, 5, 28, 28>(local_in_data, pool1_dX, local_conv1_weight, conv1_dX, conv1_dW, conv1_dB);
    
    // Store updated Conv1 weights and biases to consolidated arrays
    for(int i = 0; i < 6*1*5*5; i++) {
        #pragma HLS PIPELINE II=1
        updated_weights[CONV1_WEIGHT_OFFSET + i] = local_conv1_weight[i] - lr * conv1_dW[i];
    }
    for(int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=1
        updated_biases[CONV1_BIAS_OFFSET + i] = local_conv1_bias[i] - lr * conv1_dB[i];
    }
}
}