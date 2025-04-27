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

#define aes_word_t float

extern "C" {
    void train_lenet5_top(
            const aes_word_t *image,
            const aes_word_t *arg_conv1_weight,
            const aes_word_t *arg_conv1_bias,
            const aes_word_t *arg_conv2_weight,
            const aes_word_t *arg_conv2_bias,
            const aes_word_t *arg_fc1_weight,
            const aes_word_t *arg_fc1_bias,
            const aes_word_t *arg_fc2_weight,
            const aes_word_t *arg_fc2_bias,
            const aes_word_t *arg_fc3_weight,
            const aes_word_t *arg_fc3_bias,
            aes_word_t *logits,
            const aes_word_t *label
            ) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=HBM0 depth=784

#pragma HLS INTERFACE m_axi port=arg_conv1_weight offset=slave bundle=HBM1 depth=150
#pragma HLS INTERFACE m_axi port=arg_conv1_bias offset=slave bundle=HBM1 depth=6
#pragma HLS INTERFACE m_axi port=arg_conv2_weight offset=slave bundle=HBM2 depth=2400
#pragma HLS INTERFACE m_axi port=arg_conv2_bias offset=slave bundle=HBM2 depth=16

#pragma HLS INTERFACE m_axi port=arg_fc1_weight offset=slave bundle=HBM3 depth=30720
#pragma HLS INTERFACE m_axi port=arg_fc1_bias offset=slave bundle=HBM3 depth=120
#pragma HLS INTERFACE m_axi port=arg_fc2_weight offset=slave bundle=HBM4 depth=10080
#pragma HLS INTERFACE m_axi port=arg_fc2_bias offset=slave bundle=HBM4 depth=84
#pragma HLS INTERFACE m_axi port=arg_fc3_weight offset=slave bundle=HBM5 depth=840
#pragma HLS INTERFACE m_axi port=arg_fc3_bias offset=slave bundle=HBM5 depth=10

#pragma HLS INTERFACE m_axi port=logits offset=slave bundle=HBM6
#pragma HLS INTERFACE m_axi port=label offset=slave bundle=HBM6
        
#pragma HLS INTERFACE s_axilite port=image bundle=control

#pragma HLS INTERFACE s_axilite port=arg_conv1_weight bundle=control
#pragma HLS INTERFACE s_axilite port=arg_conv1_bias bundle=control
#pragma HLS INTERFACE s_axilite port=arg_conv2_weight bundle=control
#pragma HLS INTERFACE s_axilite port=arg_conv2_bias bundle=control

#pragma HLS INTERFACE s_axilite port=arg_fc1_weight bundle=control
#pragma HLS INTERFACE s_axilite port=arg_fc1_bias bundle=control
#pragma HLS INTERFACE s_axilite port=arg_fc2_weight bundle=control
#pragma HLS INTERFACE s_axilite port=arg_fc2_bias bundle=control
#pragma HLS INTERFACE s_axilite port=arg_fc3_weight bundle=control
#pragma HLS INTERFACE s_axilite port=arg_fc3_bias bundle=control

#pragma HLS INTERFACE s_axilite port=logits bundle=control
#pragma HLS INTERFACE s_axilite port=label bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

        // inter FIFO
        hls::stream<float> raw_image("raw_image");
        hls::stream<float> conv1_out("conv1_out");
        hls::stream<float> pool1_out("pool1_out");
        hls::stream<float> conv2_out("conv2_out");
        hls::stream<float> pool2_out("pool2_out");
        hls::stream<float> flat_out("flat_out");
        hls::stream<float> fc1_out("fc1_out");
        hls::stream<float> fc2_out("fc2_out");
        hls::stream<float> fc3_out("fc3_out");
        // backprop
        hls::stream<float> loss("loss");
        hls::stream<float> grads("grads");
#pragma HLS STREAM variable=raw_image depth=512
#pragma HLS STREAM variable=conv1_out depth=512
#pragma HLS STREAM variable=conv2_out depth=512
#pragma HLS STREAM variable=pool1_out depth=512
#pragma HLS STREAM variable=pool2_out depth=512
#pragma HLS STREAM variable=flat_out depth=512
#pragma HLS STREAM variable=fc1_out depth=512
#pragma HLS STREAM variable=fc2_out depth=512
#pragma HLS STREAM variable=fc3_out depth=512

#pragma HLS STREAM variable=loss depth=512
#pragma HLS STREAM variable=grads depth=512

        /* prepare image */
        mem2stream<IMG_DIM*IMG_DIM>(image, raw_image);

        /* CONV1 */
        hls::stream<float> conv1_input_activation("conv1_input_activation"); // for conv2d forward
        hls::stream<float> conv1_activation("conv1_activation"); // for backprop
#pragma HLS STREAM variable=conv1_input_activation depth=512
#pragma HLS STREAM variable=conv1_activation depth=512
        stream_buffer(raw_image, conv1_input_activation, conv1_activation);

        float conv1_weight[CONV1_OC][CONV1_IC][CONV1_K][CONV1_K];
        float conv1_bias[CONV1_OC];

        load_conv_params<CONV1_OC, CONV1_IC, CONV1_K, CONV1_K>(arg_conv1_weight, arg_conv1_bias, conv1_weight, conv1_bias);
        conv2d<CONV1_OC, CONV1_IC, CONV1_K, IMG_DIM>(conv1_input_activation, conv1_out, conv1_weight, conv1_bias);

        /* POOL1 */
        hls::stream<float> pool1_input_activation("pool1_input_activation");
        hls::stream<float> pool1_activation("poo1_activation");
#pragma HLS STREAM variable=pool1_input_activation depth=512
#pragma HLS STREAM variable=pool1_activation depth=512
        stream_buffer(conv1_out, pool1_input_activation, pool1_activation);

        avg_pool<POOL_SIZE, STRIDE, CONV1_OH, CONV1_OW, CONV1_OC>(pool1_input_activation, pool1_out);

        /* CONV2 */
        hls::stream<float> conv2_input_activation("conv2_input_activation");
        hls::stream<float> conv2_activation("conv2_activation");
#pragma HLS STREAM variable=conv2_input_activation depth=512
#pragma HLS STREAM variable=conv2_activation depth=512
        stream_buffer(pool1_out, conv2_input_activation, conv2_activation);

        float conv2_weight[CONV2_OC][CONV2_IC][CONV2_K][CONV2_K];
        float conv2_bias[CONV2_OC];
        load_conv_params<CONV2_OC, CONV2_IC, CONV2_K, CONV2_K>(arg_conv2_weight, arg_conv2_bias, conv2_weight, conv2_bias);
        conv2d<CONV2_OC, CONV2_IC, CONV2_K, CONV2_IN_DIM>(conv2_input_activation, conv2_out, conv2_weight, conv2_bias);

        /* POOL2 */
        hls::stream<float> pool2_input_activation("pool2_input_activation");
        hls::stream<float> pool2_activation("pool2_activation");
#pragma HLS STREAM variable=pool2_input_activation depth=512
#pragma HLS STREAM variable=pool2_activation depth=512
        stream_buffer(conv2_out, pool2_input_activation, pool2_activation);

        avg_pool<POOL_SIZE, STRIDE, CONV2_OH, CONV2_OW, CONV2_OC>(pool2_input_activation, pool2_out);

        /* FLATTEN */
        hls::stream<float> flatten_input_activation("flatten_input_activation");
        hls::stream<float> flatten_activation("flatten_activation");
#pragma HLS STREAM variable=pool2_input_activation depth=512
#pragma HLS STREAM variable=pool2_activation depth=512
        stream_buffer(pool2_out, flatten_input_activation, flatten_activation);
        
        flatten<FLAT_IH, FLAT_IW, CONV2_OC>(flatten_input_activation, flat_out);

        /* FULLY CONNECTED LAYER 1 */ 
        hls::stream<float> fc1_input_activation("fc1_input_activation");
        hls::stream<float> fc1_activation("fc1_activation");
#pragma HLS STREAM variable=fc1_input_activation depth=512
#pragma HLS STREAM variable=fc1_activation depth=512
        stream_buffer(flat_out, fc1_input_activation, fc1_activation);

        float fc1_weight[FC1_IN][FC1_OUT];
        float fc1_bias[FC1_OUT];
        load_fc_params<FC1_IN, FC1_OUT>(arg_fc1_weight, arg_fc1_bias, fc1_weight, fc1_bias);
        fc<FC1_IN, FC1_OUT>(fc1_input_activation, fc1_out, fc1_weight, fc1_bias, true);

        /* FULLY CONNECTED LAYER 2 */ 
        hls::stream<float> fc2_input_activation("fc2_input_activation");
        hls::stream<float> fc2_activation("fc2_activation");
#pragma HLS STREAM variable=fc2_input_activation depth=512
#pragma HLS STREAM variable=fc2_activation depth=512
        stream_buffer(fc1_out, fc2_input_activation, fc2_activation);

        float fc2_weight[FC2_IN][FC2_OUT];
        float fc2_bias[FC2_OUT];
        load_fc_params<FC2_IN, FC2_OUT>(arg_fc2_weight, arg_fc2_bias, fc2_weight, fc2_bias);
        fc<FC2_IN, FC2_OUT>(fc2_input_activation, fc2_out, fc2_weight, fc2_bias, true);

        /* FULLY CONNECTED LAYER 3 */ 
        hls::stream<float> fc3_input_activation("fc3_input_activation");
        hls::stream<float> fc3_activation("fc3_activation");
#pragma HLS STREAM variable=fc3_input_activation depth=512
#pragma HLS STREAM variable=fc3_activation depth=512
        stream_buffer(fc2_out, fc3_input_activation, fc3_activation);

        float fc3_weight[FC3_IN][FC3_OUT];
        float fc3_bias[FC3_OUT];
        load_fc_params<FC3_IN, FC3_OUT>(arg_fc3_weight, arg_fc3_bias, fc3_weight, fc3_bias);
        fc<FC3_IN, FC3_OUT>(fc3_input_activation, fc3_out, fc3_weight, fc3_bias, false);

        /* Get final output */
        for(int i=0; i<FC3_OUT; ++i) {
#pragma HLS PIPELINE II=1
            logits[i] = (aes_word_t)fc3_out.read();
        }

        // LOSS
        hls::stream<float> y_pred("y_pred");
        hls::stream<float> y_true("y_true");
#pragma HLS STREAM variable=y_pred depth=512
#pragma HLS STREAM variable=y_true depth=512
        stream_from_ddr<FC3_OUT>(logits, y_pred); 
        stream_from_ddr<FC3_OUT>(label, y_true); 

        mse_loss<FC3_OUT>(y_pred, y_true, loss, grads);

        // FULLY CONNECTED 3 BACKPROP
        hls::stream<float> fc3_grads("fc3_grads");
#pragma HLS STREAM variable=fc3_grads depth=512

        float fc3_dW[FC3_IN][FC3_OUT];
        float fc3_dB[FC3_OUT];
        fc_backward<FC3_IN, FC3_OUT>(fc3_activation, grads, fc3_grads, fc3_weight, fc3_dW, fc3_dB, false);

        // FULLY CONNECTED 2 BACKPROP
        hls::stream<float> fc2_grads("fc2_grads");
#pragma HLS STREAM variable=fc2_grads depth=512

        float fc2_dW[FC2_IN][FC2_OUT];
        float fc2_dB[FC2_OUT];
        fc_backward<FC2_IN, FC2_OUT>(fc2_activation, fc3_grads, fc2_grads, fc2_weight, fc2_dW, fc2_dB, true);

        // FULLY CONNECTED 1 BACKPROP
        hls::stream<float> fc1_grads("fc1_grads");
#pragma HLS STREAM variable=fc1_grads depth=512

        float fc1_dW[FC1_IN][FC1_OUT];
        float fc1_dB[FC1_OUT];
        fc_backward<FC1_IN, FC1_OUT>(fc1_activation, fc2_grads, fc1_grads, fc1_weight, fc1_dW, fc1_dB, true);

        // FLATTEN BACKPROP
        hls::stream<float> flatten_grads("flatten_grads");
#pragma HLS STREAM variable=flatten_grads depth=512

        flatten_backward<FLAT_IH, FLAT_IW, CONV2_OC>(fc1_grads, flatten_grads); 

        // POOL2 BACKPROP
        hls::stream<float> pool2_grads("pool2_grads");
#pragma HLS STREAM variable=pool2_grads depth=512

        avg_pool_backward<POOL_SIZE, STRIDE, CONV2_OH, CONV2_OW, CONV2_OC>(flatten_grads, pool2_grads);

        // CONV2 BACKPROP
        hls::stream<float> conv2_grads("conv2_grads");
#pragma HLS STREAM variable=conv2_grads depth=512

        float conv2_dW[CONV2_OC][CONV2_IC][CONV2_K][CONV2_K];
        float conv2_dB[CONV2_OC];

        conv2d_backward<CONV2_OC, CONV2_IC, CONV2_K, CONV2_OH, CONV2_OW>(conv2_activation, pool2_grads, conv2_grads, conv2_weight, conv2_dW, conv2_dB);

        // POOL1 BACKPROP
        hls::stream<float> pool1_grads("pool1_grads");
#pragma HLS STREAM variable=pool1_grads depth=512

        avg_pool_backward<POOL_SIZE, STRIDE, CONV1_OH, CONV1_OW, CONV1_OC>(conv2_grads, pool1_grads);

        // CONV1 BACKPROP
        hls::stream<float> conv1_grads("conv1_grads");
#pragma HLS STREAM variable=conv1_grads depth=512

        float conv1_dW[CONV1_OC][CONV1_IC][CONV1_K][CONV1_K];
        float conv1_dB[CONV1_OC];

        conv2d_backward<CONV1_OC, CONV1_IC, CONV1_K, CONV1_OH, CONV1_OW>(conv1_activation, pool1_grads, conv1_grads, conv1_weight, conv1_dW, conv1_dB);

        // update
        conv2d_update<CONV1_OC, CONV1_IC, CONV1_K>(conv1_weight, conv1_dW);
        bias_update<CONV1_OC>(conv1_bias, conv1_dB);
        conv2d_update<CONV2_OC, CONV2_IC, CONV2_K>(conv2_weight, conv2_dW);
        bias_update<CONV2_OC>(conv2_bias, conv2_dB);
        
        general_update<FC1_IN, FC1_OUT>(fc1_weight, fc1_dW);
        bias_update<FC1_OUT>(fc1_bias, fc1_dB);
        general_update<FC2_IN, FC2_OUT>(fc2_weight, fc2_dW);
        bias_update<FC2_OUT>(fc2_bias, fc2_dB);
        general_update<FC3_IN, FC3_OUT>(fc3_weight, fc3_dW);
        bias_update<FC3_OUT>(fc3_bias, fc3_dB);
    }
}
