#include "lenet5/forward_path.h"
#include "lenet5/conv2d.h"
#include "lenet5/avg_pool.h"
#include "lenet5/fc_layer.h"
#include "constants.hpp"

extern "C" {
void forward_path(
    data_ap_fixed_t* in_data,
    data_ap_fixed_t* weights,       // Single array for all weights
    data_ap_fixed_t* biases,        // Single array for all biases
    data_ap_fixed_t* outs
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
    data_ap_fixed_t local_in_data[CONV1_IN_CH][CONV1_IN_ROWS][CONV1_IN_COLS]; // [1*28*28];
    data_ap_fixed_t local_conv1_out[CONV1_OUT_CH][CONV1_OUT_ROWS][CONV1_OUT_COLS];// [6*24*24];
    data_ap_fixed_t local_pool1_out[CONV2_IN_CH][CONV2_IN_ROWS][CONV2_IN_COLS]; // [6*12*12];
    data_ap_fixed_t local_conv2_out[CONV2_OUT_CH][CONV2_OUT_ROWS][CONV2_OUT_COLS]; // [16*8*8];
    data_ap_fixed_t local_pool2_out[CONV2_OUT_CH][CONV2_OUT_ROWS/2][CONV2_OUT_COLS/2]; // [16*4*4];
    data_ap_fixed_t local_fc1_out[FC1_OUT_DIM]; // [120];
    data_ap_fixed_t local_fc2_out[FC2_OUT_DIM]; // [84];
    data_ap_fixed_t local_fc3_out[FC3_OUT_DIM]; // [10];
    
    // Local memory for weights and biases
    data_ap_fixed_t local_conv1_weight[CONV1_OUT_CH][CONV1_IN_CH][KERNEL_SIZE][KERNEL_SIZE]; // [6*1*5*5];
    data_ap_fixed_t local_conv1_bias[CONV1_OUT_CH]; // [6];
    data_ap_fixed_t local_conv2_weight[CONV2_OUT_CH][CONV2_IN_CH][KERNEL_SIZE][KERNEL_SIZE]; // [16*6*5*5];
    data_ap_fixed_t local_conv2_bias[CONV2_OUT_CH]; // [16];
    data_ap_fixed_t local_fc1_weight[FC1_IN_DIM][FC1_OUT_DIM]; // [256*120];
    data_ap_fixed_t local_fc1_bias[FC1_OUT_DIM]; // [120];
    data_ap_fixed_t local_fc2_weight[FC2_IN_DIM][FC2_OUT_DIM]; // [120*84];
    data_ap_fixed_t local_fc2_bias[FC2_OUT_DIM]; // [84];
    data_ap_fixed_t local_fc3_weight[FC3_IN_DIM][FC3_OUT_DIM]; // [84*10];
    data_ap_fixed_t local_fc3_bias[FC3_OUT_DIM]; // [10];
    
    // Copy input data to local memory
    for(int i = 0; i < CONV1_IN_CH; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < CONV1_IN_ROWS; j++) {
            for(int k = 0; k < CONV1_IN_COLS; k++) {
                int idx = i * (CONV1_IN_ROWS*CONV1_IN_COLS) + j * (CONV1_IN_COLS) + k;
                local_in_data[i][j][k] = in_data[idx];
            }
        }
    }
    
    // Copy Conv1's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < CONV1_OUT_CH; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < CONV1_IN_CH; j++) {
            for(int k = 0; k < KERNEL_SIZE; k++) {
                for(int l = 0; l < KERNEL_SIZE; l++) {
                    int idx = i*(CONV1_IN_CH*KERNEL_SIZE*KERNEL_SIZE) + j*(KERNEL_SIZE*KERNEL_SIZE) + k*KERNEL_SIZE + l;
                    local_conv1_weight[i][j][k][l] = weights[CONV1_WEIGHT_OFFSET + idx];
                }
            }
        }
    }
    for(int i = 0; i < CONV1_OUT_CH; i++) {
        #pragma HLS PIPELINE II=1
        local_conv1_bias[i] = biases[CONV1_BIAS_OFFSET + i];
    }

    conv2d<CONV1_OUT_CH, CONV1_IN_CH, KERNEL_SIZE, CONV1_IN_ROWS, CONV1_IN_COLS>(local_in_data, local_conv1_out, local_conv1_weight, local_conv1_bias);

    avg_pool<2, 2, CONV1_OUT_CH, CONV1_OUT_ROWS, CONV1_OUT_COLS>(local_conv1_out, local_pool1_out);
    
    // Copy Conv2's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < CONV2_OUT_CH; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < CONV2_IN_CH; j++) {
            for(int k = 0; k < KERNEL_SIZE; k++) {
                for(int l = 0; l < KERNEL_SIZE; l++) {
                    int idx = i*(CONV2_IN_CH*KERNEL_SIZE*KERNEL_SIZE) + j*(KERNEL_SIZE*KERNEL_SIZE) + k*KERNEL_SIZE + l;
                    local_conv2_weight[i][j][k][l] = weights[CONV2_WEIGHT_OFFSET + idx];
                }
            }
        }
    }
    for(int i = 0; i < CONV2_OUT_CH; i++) {
        #pragma HLS PIPELINE II=1
        local_conv2_bias[i] = biases[CONV2_BIAS_OFFSET + i];
    }

    conv2d<CONV2_OUT_CH, CONV2_IN_CH, KERNEL_SIZE, CONV2_IN_ROWS, CONV2_IN_COLS>(local_pool1_out, local_conv2_out, local_conv2_weight, local_conv2_bias);

    avg_pool<2, 2, CONV2_OUT_CH, CONV2_OUT_ROWS, CONV2_OUT_COLS>(local_conv2_out, local_pool2_out);

    // Flattening
    data_ap_fixed_t local_fc1_in[FC1_IN_DIM];
    for(int i = 0; i < CONV2_OUT_CH; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < CONV2_OUT_ROWS/2; j++) {
            for(int k = 0; k < CONV2_OUT_COLS/2; k++) {
                int idx = i*(CONV2_OUT_ROWS/2*CONV2_OUT_COLS/2) + j*(CONV2_OUT_COLS/2) + k;
                local_fc1_in[idx] = local_pool2_out[i][j][k];
            }
        }
    }
    
    // Copy FC1's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < FC1_IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < FC1_OUT_DIM; j++) {
            int idx = i * FC1_OUT_DIM + j;
            local_fc1_weight[i][j] = weights[FC1_WEIGHT_OFFSET + idx];
        }
    }
    for(int i = 0; i < FC1_OUT_DIM; i++) {
        #pragma HLS PIPELINE II=1
        local_fc1_bias[i] = biases[FC1_BIAS_OFFSET + i];
    }
    fc<256, 120>(local_fc1_in, local_fc1_out, local_fc1_weight, local_fc1_bias, true);
    
    // Copy FC2's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < FC2_IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < FC2_OUT_DIM; j++) {
            int idx = i * FC2_OUT_DIM + j;
            local_fc2_weight[i][j] = weights[FC2_WEIGHT_OFFSET + idx];
        }
    }
    for(int i = 0; i < FC2_OUT_DIM; i++) {
        #pragma HLS PIPELINE II=1
        local_fc2_bias[i] = biases[FC2_BIAS_OFFSET + i];
    }
    fc<120, 84>(local_fc1_out, local_fc2_out, local_fc2_weight, local_fc2_bias, true);
    
    // Copy FC3's weights and biases to local memory from consolidated arrays
    for(int i = 0; i < FC3_IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < FC3_OUT_DIM; j++) {
            int idx = i * FC3_OUT_DIM + j;
            local_fc3_weight[i][j] = weights[FC3_WEIGHT_OFFSET + idx];
        }
    }
    for(int i = 0; i < FC3_OUT_DIM; i++) {
        #pragma HLS PIPELINE II=1
        local_fc3_bias[i] = biases[FC3_BIAS_OFFSET + i];
    }
    fc<84, 10>(local_fc2_out, local_fc3_out, local_fc3_weight, local_fc3_bias, false);

    // conv1 output
    for (int ch = 0; ch < CONV1_OUT_CH; ch++) {
        for (int row = 0; row < CONV1_OUT_ROWS; row++) {
            for (int col = 0; col < CONV1_OUT_COLS; col++) {
                #pragma HLS PIPELINE II=1
                int idx = CONV1_OUT_OFFSET + ch * (CONV1_OUT_ROWS * CONV1_OUT_COLS) + row * CONV1_OUT_COLS + col;
                outs[idx] = local_conv1_out[ch][row][col];
            }
        }
    }

    // pool1 output
    for (int ch = 0; ch < CONV2_IN_CH; ch++) {
        for (int row = 0; row < CONV2_IN_ROWS; row++) {
            for (int col = 0; col < CONV2_IN_COLS; col++) {
                #pragma HLS PIPELINE II=1
                int idx = POOL1_OUT_OFFSET + ch * (CONV2_IN_ROWS * CONV2_IN_COLS) + row * CONV2_IN_COLS + col;
                outs[idx] = local_pool1_out[ch][row][col];
            }
        }
    }

    // conv2 output
    for (int ch = 0; ch < CONV2_OUT_CH; ch++) {
        for (int row = 0; row < CONV2_OUT_ROWS; row++) {
            for (int col = 0; col < CONV2_OUT_COLS; col++) {
                #pragma HLS PIPELINE II=1
                int idx = CONV2_OUT_OFFSET + ch * (CONV2_OUT_ROWS * CONV2_OUT_COLS) + row * CONV2_OUT_COLS + col;
                outs[idx] = local_conv2_out[ch][row][col];
            }
        }
    }

    // pool2 output
    for (int ch = 0; ch < CONV2_OUT_CH; ch++) {
        for (int row = 0; row < CONV2_OUT_ROWS / 2; row++) {
            for (int col = 0; col < CONV2_OUT_COLS / 2; col++) {
                #pragma HLS PIPELINE II=1
                int idx = POOL2_OUT_OFFSET + ch * ((CONV2_OUT_ROWS/2) * (CONV2_OUT_COLS/2)) + row * (CONV2_OUT_COLS/2) + col;
                outs[idx] = local_pool2_out[ch][row][col];
            }
        }
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