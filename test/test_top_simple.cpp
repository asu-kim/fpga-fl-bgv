#include <iostream>
#include <cmath>
#include "hls_math.h"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"
#include "keys.h"
#include "weights_bias.h"
#include "encrypted_weights_bias.h"
#include "test_utils.h"
#include "top.hpp"

#define IN_ROWS 28
#define IN_COLS 28
#define KERNEL_SIZE 5
#define IN_C 1
#define OUT_C 1

void conv1_golden(
    const data_t in_data[IN_C][IN_ROWS][IN_COLS],
    data_t out_data[OUT_C * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)],
    const data_t weights[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE],
    const data_t bias[OUT_C],
    float act_out_scale = 1.0f, 
    int act_out_zp = 0
) {
    // Loop over each output channel
    for (int oc = 0; oc < OUT_C; oc++) {
        // Loop over each output row
        for (int oh = 0; oh < IN_ROWS - KERNEL_SIZE + 1; oh++) {
            // Loop over each output column
            for (int ow = 0; ow < IN_COLS - KERNEL_SIZE + 1; ow++) {
                // Initialize accumulator with bias
                ap_int<128> acc = bias[oc];
                
                // Calculate convolution for current output position
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            // Input position
                            int ih = oh + kh;
                            int iw = ow + kw;
                            
                            // Accumulate weighted input
                            data_t in_val = in_data[ic][ih][iw];
                            data_t w_val = weights[oc][ic][kh][kw];
                            acc += (ap_int<64>)in_val * (ap_int<64>)w_val;
                        }
                    }
                }
                
                // Quantize output
                float acc_float = float(acc);
                float scaled = acc_float * act_out_scale + (float)act_out_zp;
                float rounded = floor(scaled + 0.5f);
                
                // Clip to data type range
                data_t result = (data_t)rounded;
                result = hls::max(hls::numeric_limits<data_t>::min(), 
                            hls::min(hls::numeric_limits<data_t>::max(), result));
                
                // Calculate output index and store result
                int out_idx = oc * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)
                            + oh * (IN_COLS - KERNEL_SIZE + 1)
                            + ow;
                out_data[out_idx] = result;
            }
        }
    }
}

int main() {
    // hls::stream<data_t> in_stream, out_stream;

    data_t in_data[IN_C * IN_ROWS * IN_COLS];
    data_t in_data_ref[IN_C][IN_ROWS][IN_COLS];
    data_t out_data[OUT_C * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)];
    data_t out_data_ref[OUT_C * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)];
    data_t weights[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE];
    data_t bias[OUT_C];

    for(int channel=0; channel<OUT_C; ++channel) {
        for(int ic=0; ic<IN_C; ++ic) {
            bias[channel] = 0.0;
            for(int i=0; i<KERNEL_SIZE; ++i) {
                for(int j=0; j<KERNEL_SIZE; ++j) {
                    data_t weight = CONV1_WEIGHT_INT8_DATA[channel*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + ic*(KERNEL_SIZE*KERNEL_SIZE) + i*(KERNEL_SIZE) + j];
                    weights[channel][ic][i][j] = weight;
                }
            }
        }
    }
    std::cout << std::endl;

    std::cout << "weights_golden = [";
    for(int i=0; i<6; i++) {
        for(int j=0; j<1; j++) {
            for(int k=0; k<5; k++) {
                for(int l=0; l<5; l++) {
                    std::cout << weights[i][j][k][l] << ", ";
                }
            }
        }
    }
    std::cout << "]" << std::endl;

    for(int i = 0; i < OUT_C; i++) {
        bias[i] = CONV1_BIAS_INT8_DATA[i];
    }

    std::cout << "bias_golden = [";
    for(int i=0; i<6; i++) {
        std::cout << bias[i] << ", ";
    }
    std::cout << "]" << std::endl;

    for(int i=0; i<IN_ROWS; i++) {
        for(int j = 0; j < IN_COLS; j++) {
            in_data[i * IN_COLS + j] = 1;
            in_data_ref[0][i][j] = 1;
        }
    }

    // Load encrypted weights.
    data_t private_key[POLYNOMIAL_DEGREE];
    data_t encrypted_conv1_weight0_0[POLYNOMIAL_DEGREE];
    data_t encrypted_conv1_weight0_1[POLYNOMIAL_DEGREE];
    data_t encrypted_conv1_weight1_0[POLYNOMIAL_DEGREE];
    data_t encrypted_conv1_weight1_1[POLYNOMIAL_DEGREE];
    data_t encrypted_conv1_bias0[POLYNOMIAL_DEGREE];
    data_t encrypted_conv1_bias1[POLYNOMIAL_DEGREE];

    for(int i = 0; i < POLYNOMIAL_DEGREE; i++) {
        private_key[i] = PRIVATE_KEY[i];
        encrypted_conv1_weight0_0[i] = CONV1_WEIGHT_INT8_DATA0_ENC1[i];
        encrypted_conv1_weight0_1[i] = CONV1_WEIGHT_INT8_DATA0_ENC2[i];
        encrypted_conv1_weight1_0[i] = CONV1_WEIGHT_INT8_DATA1_ENC1[i];
        encrypted_conv1_weight1_1[i] = CONV1_WEIGHT_INT8_DATA1_ENC2[i];
        encrypted_conv1_bias0[i] = CONV1_BIAS_INT8_DATA_ENC1[i];
        encrypted_conv1_bias1[i] = CONV1_BIAS_INT8_DATA_ENC2[i];
    }

    // run conv2d
    // conv1(in_stream, out_stream, flatten_weights, flatten_bias);
    top(private_key, encrypted_conv1_weight0_0, encrypted_conv1_weight0_1, encrypted_conv1_weight1_0,
        encrypted_conv1_weight1_1, encrypted_conv1_bias0, encrypted_conv1_bias1, in_data, out_data);
    conv1_golden(in_data_ref, out_data_ref, weights, bias);

    int errors = 0;
    const int output_size = (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1) * OUT_C;
    std::cout << "out_data_ref = [";
    for(int i=0; i<output_size; ++i) {
        std::cout << out_data_ref[i] << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    std::cout << "out_data = [";
    for(int i=0; i<output_size; ++i) {
        std::cout << out_data[i] << ", ";
    }
    std::cout << "]" << std::endl;
    // std::cout << "Error indexes: ";
    // for(int i=0; i<output_size; ++i) {
    //     if(out_data[i] != out_data_ref[i]) {
    //         errors = 1;
    //         std::cout << i << ", ";
    //     }
    // }
    std::cout << std::endl;

    return errors;
}
