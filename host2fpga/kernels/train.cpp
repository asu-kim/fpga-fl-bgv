#include "hls_stream.h"
#include "aes_utils.h"
#include "utils.h"

#include "reader.h"

// foward
#include "conv2d.h"
#include "avg_pool.h"
#include "flatten.h"
#include "fc.h"

// backward
#include "conv2d_bwd.h"
#include "avg_pool_bwd.h"
#include "flatten_bwd.h"
#include "fc_bwd.h"

// update
#include "softmax.h"
#include "mse_loss.h"
#include "update.h"


const int POOL_SIZE=2, STRIDE=2;

const int IMG_DIM=28;
const int CONV1_OC=6, CONV1_IC=1, CONV1_K=5;
const int CONV1_OH=IMG_DIM-CONV1_K+1, CONV1_OW=IMG_DIM-CONV1_K+1;

const int CONV2_IN_DIM=(CONV1_OH/POOL_SIZE);
const int CONV2_OC=16, CONV2_IC=CONV1_OC, CONV2_K=5;
const int CONV2_OH=CONV2_IN_DIM-CONV2_K+1, CONV2_OW=CONV2_IN_DIM-CONV2_K+1;

const int FLAT_IH=(CONV2_OH/POOL_SIZE), FLAT_IW=(CONV2_OW/POOL_SIZE);

const int FC1_IN=FLAT_IH * FLAT_IW * CONV2_OC, FC1_OUT=120;
const int FC2_IN=120, FC2_OUT=84;
const int FC3_IN=84, FC3_OUT=10;


extern "C" {
    void train_lenet5_top(
            const float *image,
            const float *conv1_weight,
            const float *conv1_bias,
            const float *conv2_in,
            const float *conv2_weight,
            const float *conv2_bias,
            const float *fc1_in,
            const float *fc1_weight,
            const float *fc1_bias,
            const float *fc2_in,
            const float *fc2_weight,
            const float *fc2_bias,
            const float *fc3_in,
            const float *fc3_weight,
            const float *fc3_bias,
            float *probs,
            const float *label
            ) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=HBM0 depth=784

#pragma HLS INTERFACE m_axi port=conv1_weight offset=slave bundle=HBM1 depth=150
#pragma HLS INTERFACE m_axi port=conv1_bias offset=slave bundle=HBM1 depth=6

#pragma HLS INTERFACE m_axi port=conv2_in offset=slave bundle=HBM2 depth=3456
#pragma HLS INTERFACE m_axi port=conv2_weight offset=slave bundle=HBM2 depth=2400
#pragma HLS INTERFACE m_axi port=conv2_bias offset=slave bundle=HBM2 depth=16

#pragma HLS INTERFACE m_axi port=fc1_in offset=slave bundle=HBM3 depth=256
#pragma HLS INTERFACE m_axi port=fc1_weight offset=slave bundle=HBM3 depth=30720
#pragma HLS INTERFACE m_axi port=fc1_bias offset=slave bundle=HBM3 depth=120

#pragma HLS INTERFACE m_axi port=fc2_in offset=slave bundle=HBM4 depth=120
#pragma HLS INTERFACE m_axi port=fc2_weight offset=slave bundle=HBM4 depth=10080
#pragma HLS INTERFACE m_axi port=fc2_bias offset=slave bundle=HBM4 depth=84

#pragma HLS INTERFACE m_axi port=fc3_in offset=slave bundle=HBM5 depth=84
#pragma HLS INTERFACE m_axi port=fc3_weight offset=slave bundle=HBM5 depth=840
#pragma HLS INTERFACE m_axi port=fc3_bias offset=slave bundle=HBM5 depth=10

#pragma HLS INTERFACE m_axi port=probs offset=slave bundle=HBM6
#pragma HLS INTERFACE m_axi port=label offset=slave bundle=HBM6
        
#pragma HLS INTERFACE s_axilite port=image bundle=control

#pragma HLS INTERFACE s_axilite port=conv1_weight bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_bias bundle=control

#pragma HLS INTERFACE s_axilite port=conv2_in bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_weight bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_bias bundle=control

#pragma HLS INTERFACE s_axilite port=fc1_in bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_weight bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_bias bundle=control

#pragma HLS INTERFACE s_axilite port=fc2_in bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_weight bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_bias bundle=control

#pragma HLS INTERFACE s_axilite port=fc3_in bundle=control
#pragma HLS INTERFACE s_axilite port=fc3_weight bundle=control
#pragma HLS INTERFACE s_axilite port=fc3_bias bundle=control

#pragma HLS INTERFACE s_axilite port=probs bundle=control
#pragma HLS INTERFACE s_axilite port=label bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

        // LOSS
        float loss, grads[FC3_OUT];
        mse_loss<FC3_OUT>(probs, label, loss, grads);

        // FULLY CONNECTED 3 BACKPROP
        // template params: in_activation, grad from prev, grad generate by this layer, weight, dW, dB
        float fc3_grads[FC3_IN];
        float fc3_dW[FC3_IN][FC3_OUT];
        float fc3_dB[FC3_OUT];
        fc_backward<FC3_IN, FC3_OUT>(fc3_in, grads, fc3_grads, fc3_weight, fc3_dW, fc3_dB, false);

        // FULLY CONNECTED 2 BACKPROP
        float fc2_grads[FC2_IN];
        float fc2_dW[FC2_IN][FC2_OUT];
        float fc2_dB[FC2_OUT];
        fc_backward<FC2_IN, FC2_OUT>(fc2_in, fc3_grads, fc2_grads, fc2_weight, fc2_dW, fc2_dB, true);

        // FULLY CONNECTED 1 BACKPROP
        float fc1_grads[FC1_IN];
        float fc1_dW[FC1_IN][FC1_OUT];
        float fc1_dB[FC1_OUT];
        fc_backward<FC1_IN, FC1_OUT>(fc1_in, fc2_grads, fc1_grads, fc1_weight, fc1_dW, fc1_dB, true);

        // FLATTEN BACKPROP
        float flatten_grads[FLAT_IH*FLAT_IW*CONV2_OC];
        flatten_backward<FLAT_IH, FLAT_IW, CONV2_OC>(fc1_grads, flatten_grads); 

        // POOL2 BACKPROP
        float pool2_grads[CONV2_OC*CONV2_OH*CONV2_OW];
        avg_pool_backward<POOL_SIZE, STRIDE, CONV2_OH, CONV2_OW, CONV2_OC>(flatten_grads, pool2_grads);

        // CONV2 BACKPROP
        float conv2_grads[CONV1_OC*CONV2_IN_DIM*CONV2_IN_DIM];
        float conv2_dW[CONV2_OC][CONV2_IC][CONV2_K][CONV2_K];
        float conv2_dB[CONV2_OC];

        conv2d_backward<CONV2_OC, CONV2_IC, CONV2_K, CONV2_OH, CONV2_OW>(conv2_in, pool2_grads, conv2_grads, conv2_weight, conv2_dW, conv2_dB);

        // POOL1 BACKPROP
        float pool1_grads[CONV1_OC*CONV1_OH*CONV1_OW];
        avg_pool_backward<POOL_SIZE, STRIDE, CONV1_OH, CONV1_OW, CONV1_OC>(conv2_grads, pool1_grads);

        // CONV1 BACKPROP
        float conv1_grads[IMG_DIM*IMG_DIM];
        float conv1_dW[CONV1_OC][CONV1_IC][CONV1_K][CONV1_K];
        float conv1_dB[CONV1_OC];

        conv2d_backward<CONV1_OC, CONV1_IC, CONV1_K, CONV1_OH, CONV1_OW>(image, pool1_grads, conv1_grads, conv1_weight, conv1_dW, conv1_dB);

        // update
        float conv1_updated_weight[CONV1_OC*CONV1_IC*CONV1_K*CONV1_K];
        float conv1_updated_bias[CONV1_OC];
        conv2d_update<CONV1_OC, CONV1_IC, CONV1_K>(conv1_weight, conv1_dW, conv1_updated_weight);
        bias_update<CONV1_OC>(conv1_bias, conv1_dB, conv1_updated_bias);

        float conv2_updated_weight[CONV2_OC*CONV2_IC*CONV2_K*CONV2_K];
        float conv2_updated_bias[CONV2_OC];
        conv2d_update<CONV2_OC, CONV2_IC, CONV2_K>(conv2_weight, conv2_dW, conv2_updated_weight);
        bias_update<CONV2_OC>(conv2_bias, conv2_dB, conv2_updated_bias);
        
        float fc1_updated_weight[FC1_IN*FC1_OUT];
        float fc1_updated_bias[FC1_OUT];
        general_update<FC1_IN, FC1_OUT>(fc1_weight, fc1_dW, fc1_updated_weight);
        bias_update<FC1_OUT>(fc1_bias, fc1_dB, fc1_updated_bias);

        float fc2_updated_weight[FC2_IN*FC2_OUT];
        float fc2_updated_bias[FC2_OUT];
        general_update<FC2_IN, FC2_OUT>(fc2_weight, fc2_dW, fc2_updated_weight);
        bias_update<FC2_OUT>(fc2_bias, fc2_dB, fc2_updated_bias);

        float fc3_updated_weight[FC3_IN*FC3_OUT];
        float fc3_updated_bias[FC3_OUT];
        general_update<FC3_IN, FC3_OUT>(fc3_weight, fc3_dW, fc3_updated_weight);
        bias_update<FC3_OUT>(fc3_bias, fc3_dB, fc3_updated_bias);
    }
}
