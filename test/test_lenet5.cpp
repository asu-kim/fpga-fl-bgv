#include <iostream>
#include "lenet5.h"
#include "test_utils.h"


int main() {
    hls::stream<data_t> in_stream, out_stream;
    data_t conv1_weight[6][1][5][5], conv1_bias[6];
    data_t conv2_weight[16][6][5][5], conv2_bias[16];
    data_t fc1_weight[120][256], fc1_bias[120];
    data_t fc2_weight[84][120], fc2_bias[84];
    data_t fc3_weight[10][84], fc3_bias[10];

    // channel, kernel size
    // bias size
    load_conv2d_weight<6, 1, 5>(CONV1_WEIGHT_BIN, conv1_weight);
    load_bias<6>(CONV1_BIAS_BIN, conv1_bias);

    load_conv2d_weight<16, 6, 5>(CONV2_WEIGHT_BIN, conv2_weight);
    load_bias<16>(CONV2_BIAS_BIN, conv2_bias);

    // out dim, in dim
    // bias size
    load_fc_weight<120, 256>(FC1_WEIGHT_BIN, fc1_weight);
    load_bias<120>(FC1_BIAS_BIN, fc1_bias);

    load_fc_weight<84, 120>(FC2_WEIGHT_BIN, fc2_weight);
    load_bias<84>(FC2_BIAS_BIN, fc2_bias);

    load_fc_weight<10, 84>(FC3_WEIGHT_BIN, fc3_weight);
    load_bias<10>(FC3_BIAS_BIN, fc3_bias);

    // input dimension
    load_input_to_stream<28>(DATASET, in_stream);

    lenet5(in_stream, out_stream, conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias);

    int predicted_class = -1;
    data_t max_val = -999;

    for(int i=0; i<10; ++i) {
        data_t val = out_stream.read();
        if(val > max_val) {
            max_val = val;
            predicted_class = i;
        }
    }

    std::cout << "Predicted class: " << predicted_class << std::endl;
    return check_against_golden(predicted_class);
}
