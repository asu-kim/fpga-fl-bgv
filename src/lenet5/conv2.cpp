#include <hls_math.h>
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"

extern "C" {
    void conv2(
        data_ap_fixed_t* in_data,
        data_ap_fixed_t* out_data,
        data_ap_fixed_t* conv1_weight,
        data_ap_fixed_t* conv1_bias
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=864
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=1024
        #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem2 depth=2400
        #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=16
        // #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=144
        // #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=64
        // #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem2 depth=25
        // #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=1
        #pragma HLS INTERFACE s_axilite port=return bundle=control

        // data_ap_fixed_t local_weight[2400];
        // data_ap_fixed_t local_bias[16];
        // for(int i = 0; i < 2400; i++) {
        //     local_weight[i] = conv1_weight[i];
        // }
        // for(int i = 0; i < 16; i++) {
        //     local_bias[i] = conv1_bias[i];
        // }

        // conv2d<16, 6, 5, 12, 12>(
        //     in_data,             // Same input data for all channels
        //     out_data,       // Output pointer offset for this batch
        //     local_weight,  // Sliced weights for this batch
        //     local_bias     // Sliced biases for this batch
        // );


        // Constants for convenience
        const int OUT_C_TOTAL = 16;
        const int OUT_C_PER_CALL = 1;  // Process 4 output channels per call
        const int NUM_CALLS = OUT_C_TOTAL / OUT_C_PER_CALL;  // 4 calls
        const int IN_C = 6;
        const int KERNEL_SIZE = 5;
        const int IN_ROWS = 12;
        const int IN_COLS = 12;
        const int OUT_ROWS = IN_ROWS - KERNEL_SIZE + 1; // 8
        const int OUT_COLS = IN_COLS - KERNEL_SIZE + 1; // 8
        
        // Process in batches of 4 output channels
        for (int call_idx = 0; call_idx < NUM_CALLS; call_idx++) {
            // Calculate starting output channel for this batch
            int oc_start = call_idx * OUT_C_PER_CALL;
            
            // Create local arrays for the current batch of output channels
            data_ap_fixed_t local_weight_slice[OUT_C_PER_CALL*IN_C*KERNEL_SIZE*KERNEL_SIZE]; // 4*6*5*5 = 600 weights
            data_ap_fixed_t local_bias_slice[OUT_C_PER_CALL]; // 4 bias values
            
            // Copy weights and biases for the current batch of output channels
            for (int oc_offset = 0; oc_offset < OUT_C_PER_CALL; oc_offset++) {
                int oc = oc_start + oc_offset;
                
                // Copy bias for this output channel
                local_bias_slice[oc_offset] = conv1_bias[oc];
                
                // Copy weights for this output channel
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            // Calculate source index in the original weight array
                            int weight_idx = oc*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + 
                                            ic*(KERNEL_SIZE*KERNEL_SIZE) + 
                                            kh*KERNEL_SIZE + kw;
                            
                            // Calculate destination index in the sliced weight array
                            int slice_idx = oc_offset*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + 
                                           ic*(KERNEL_SIZE*KERNEL_SIZE) + 
                                           kh*KERNEL_SIZE + kw;
                            
                            local_weight_slice[slice_idx] = conv1_weight[weight_idx];
                        }
                    }
                }
            }
            
            // Calculate output pointer offset for this batch
            data_ap_fixed_t* out_batch_ptr = out_data + (oc_start * OUT_ROWS * OUT_COLS);
            
            // Call conv2d for this batch of output channels
            conv2d<OUT_C_PER_CALL, IN_C, KERNEL_SIZE, IN_ROWS, IN_COLS>(
                in_data,             // Same input data for all channels
                out_batch_ptr,       // Output pointer offset for this batch
                local_weight_slice,  // Sliced weights for this batch
                local_bias_slice     // Sliced biases for this batch
            );
        }
    }
}