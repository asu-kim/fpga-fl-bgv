#include <iostream>
#include <cmath>
#include "hls_math.h"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"
#include "weights_bias.h"
#include "test_utils.h"

#define IN_ROWS 28
#define IN_COLS 28
#define KERNEL_SIZE 5
#define IN_C 1
#define OUT_C 6

void conv1_golden(
    const data_ap_fixed_t in_data[IN_C][IN_ROWS][IN_COLS],
    data_ap_fixed_t out_data[OUT_C * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)],
    const data_ap_fixed_t weights[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE],
    const data_ap_fixed_t bias[OUT_C]
) {
    // Loop over each output channel
    for (int oc = 0; oc < OUT_C; oc++) {
        // Loop over each output row
        for (int oh = 0; oh < IN_ROWS - KERNEL_SIZE + 1; oh++) {
            // Loop over each output column
            for (int ow = 0; ow < IN_COLS - KERNEL_SIZE + 1; ow++) {
                // Initialize accumulator with bias
                data_ap_fixed_t acc = bias[oc];
                
                // Calculate convolution for current output position
                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                            // Input position
                            int ih = oh + kh;
                            int iw = ow + kw;
                            
                            // Accumulate weighted input
                            data_ap_fixed_t in_val = in_data[ic][ih][iw];
                            data_ap_fixed_t w_val = weights[oc][ic][kh][kw];
                            acc += in_val * w_val;
                        }
                    }
                }
                
                // Quantize output
                // data_ap_fixed_t acc_float = data_ap_fixed_t(acc);
                // data_ap_fixed_t scaled = acc_float * act_out_scale + (data_ap_fixed_t)act_out_zp;
                // data_ap_fixed_t rounded = floor(scaled + 0.5f);
                
                // // Clip to data type range
                // data_ap_fixed_t result = (data_ap_fixed_t)rounded;
                // result = hls::max(hls::numeric_limits<data_ap_fixed_t>::min(), 
                //             hls::min(hls::numeric_limits<data_ap_fixed_t>::max(), result));
                
                // Calculate output index and store result
                int out_idx = oc * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)
                            + oh * (IN_COLS - KERNEL_SIZE + 1)
                            + ow;
                out_data[out_idx] = acc;
            }
        }
    }
}

int main() {
    // hls::stream<data_ap_fixed_t> in_stream, out_stream;

    data_ap_fixed_t in_data[IN_C * IN_ROWS * IN_COLS];
    data_ap_fixed_t in_data_ref[IN_C][IN_ROWS][IN_COLS];
    data_ap_fixed_t out_data[OUT_C * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)];
    data_ap_fixed_t out_data_ref[OUT_C * (IN_ROWS - KERNEL_SIZE + 1) * (IN_COLS - KERNEL_SIZE + 1)];
    data_ap_fixed_t weights[OUT_C][IN_C][KERNEL_SIZE][KERNEL_SIZE];
    data_ap_fixed_t flatten_weights[128 * ((OUT_C*IN_C*KERNEL_SIZE*KERNEL_SIZE + 127)/128)];
    data_ap_fixed_t bias[OUT_C];
    data_ap_fixed_t flatten_bias[128 * ((OUT_C + 127)/128)];

    for(int channel=0; channel<OUT_C; ++channel) {
        for(int ic=0; ic<IN_C; ++ic) {
            bias[channel] = 0.0;
            flatten_bias[channel] = 0.0;
            for(int i=0; i<KERNEL_SIZE; ++i) {
                for(int j=0; j<KERNEL_SIZE; ++j) {
                    data_ap_fixed_t weight = CONV1_WEIGHT_INT8_DATA[channel*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + ic*(KERNEL_SIZE*KERNEL_SIZE) + i*(KERNEL_SIZE) + j];
                    weights[channel][ic][i][j] = weight;
                    flatten_weights[channel*(IN_C*KERNEL_SIZE*KERNEL_SIZE) + ic*(KERNEL_SIZE*KERNEL_SIZE) + i*(KERNEL_SIZE) + j] = weight;
                }
            }
        }
    }
    std::cout << std::endl;

    std::cout << "weights_golden = [";
    for(int i=0; i<OUT_C; i++) {
        for(int j=0; j<IN_C; j++) {
            for(int k=0; k<KERNEL_SIZE; k++) {
                for(int l=0; l<KERNEL_SIZE; l++) {
                    std::cout << weights[i][j][k][l] << ", ";
                }
            }
        }
    }
    std::cout << "]" << std::endl;

    for(int i = 0; i < OUT_C; i++) {
        bias[i] = CONV1_BIAS_INT8_DATA[i];
        flatten_bias[i] = CONV1_BIAS_INT8_DATA[i];
    }

    std::cout << "bias_golden = [";
    for(int i=0; i<OUT_C; i++) {
        std::cout << bias[i] << ", ";
    }
    std::cout << "]" << std::endl;

    for(int i=0; i<IN_ROWS; i++) {
        for(int j = 0; j < IN_COLS; j++) {
            in_data[i * IN_COLS + j] = 1;
            in_data_ref[0][i][j] = 1;
        }
    }

    // run conv2d
    // conv1(in_stream, out_stream, flatten_weights, flatten_bias);
    conv1(in_data, out_data, flatten_weights, flatten_bias);
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
    std::cout << "]" << std::endl << std::endl;
    std::cout << "Error indexes: ";
    for(int i=0; i<output_size; ++i) {
        if(out_data[i] != out_data_ref[i]) {
            errors = 1;
            std::cout << i << ", ";
        }
    }
    std::cout << std::endl;

    return errors;
}
