#include "aes_utils.h"
#include "hls_stream.h"
#include "utils.h"

#include "conv2d.h"
#include "avg_pool.h"
#include "flatten.h"
#include "fc.h"

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

typedef ap_int<32> data_t;

extern "C" {
    void lenet5_top(
            const aes_word_t *image,
            const aes_word_t *enc_conv1_weight,
            const aes_word_t *enc_conv1_bias,
            const aes_word_t *enc_conv2_weight,
            const aes_word_t *enc_conv2_bias,
            const aes_word_t *enc_fc1_weight,
            const aes_word_t *enc_fc1_bias,
            const aes_word_t *enc_fc2_weight,
            const aes_word_t *enc_fc2_bias,
            const aes_word_t *enc_fc3_weight,
            const aes_word_t *enc_fc3_bias,
            aes_word_t *logits
            ) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=HBM0 depth=784

#pragma HLS INTERFACE m_axi port=enc_conv1_weight offset=slave bundle=HBM1 depth=150
#pragma HLS INTERFACE m_axi port=enc_conv1_bias offset=slave bundle=HBM1 depth=6
#pragma HLS INTERFACE m_axi port=enc_conv2_weight offset=slave bundle=HBM2 depth=2400
#pragma HLS INTERFACE m_axi port=enc_conv2_bias offset=slave bundle=HBM2 depth=16

#pragma HLS INTERFACE m_axi port=enc_fc1_weight offset=slave bundle=HBM3 depth=30720
#pragma HLS INTERFACE m_axi port=enc_fc1_bias offset=slave bundle=HBM3 depth=120
#pragma HLS INTERFACE m_axi port=enc_fc2_weight offset=slave bundle=HBM4 depth=10080
#pragma HLS INTERFACE m_axi port=enc_fc2_bias offset=slave bundle=HBM4 depth=84
#pragma HLS INTERFACE m_axi port=enc_fc3_weight offset=slave bundle=HBM5 depth=840
#pragma HLS INTERFACE m_axi port=enc_fc3_bias offset=slave bundle=HBM5 depth=10

#pragma HLS INTERFACE m_axi port=logits offset=slave bundle=HBM6
        
#pragma HLS INTERFACE s_axilite port=image bundle=control

#pragma HLS INTERFACE s_axilite port=enc_conv1_weight bundle=control
#pragma HLS INTERFACE s_axilite port=enc_conv1_bias bundle=control
#pragma HLS INTERFACE s_axilite port=enc_conv2_weight bundle=control
#pragma HLS INTERFACE s_axilite port=enc_conv2_bias bundle=control

#pragma HLS INTERFACE s_axilite port=enc_fc1_weight bundle=control
#pragma HLS INTERFACE s_axilite port=enc_fc1_bias bundle=control
#pragma HLS INTERFACE s_axilite port=enc_fc2_weight bundle=control
#pragma HLS INTERFACE s_axilite port=enc_fc2_bias bundle=control
#pragma HLS INTERFACE s_axilite port=enc_fc3_weight bundle=control
#pragma HLS INTERFACE s_axilite port=enc_fc3_bias bundle=control

#pragma HLS INTERFACE s_axilite port=logits bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

        // inter FIFO
        hls::stream<data_t> raw_image("raw_image");
        hls::stream<data_t> conv1_out("conv1_out");
        hls::stream<data_t> pool1_out("pool1_out");
        hls::stream<data_t> conv2_out("conv2_out");
        hls::stream<data_t> pool2_out("pool2_out");
        hls::stream<data_t> flat_out("flat_out");
        hls::stream<data_t> fc1_out("fc1_out");
        hls::stream<data_t> fc2_out("fc2_out");
        hls::stream<data_t> fc3_out("fc3_out");
#pragma HLS STREAM variable=raw_image depth=512
#pragma HLS STREAM variable=conv1_out depth=512
#pragma HLS STREAM variable=conv2_out depth=512
#pragma HLS STREAM variable=pool1_out depth=512
#pragma HLS STREAM variable=pool2_out depth=512
#pragma HLS STREAM variable=flat_out depth=512
#pragma HLS STREAM variable=fc1_out depth=512
#pragma HLS STREAM variable=fc2_out depth=512
#pragma HLS STREAM variable=fc3_out depth=512

#pragma HLS DATAFLOW
        for(int i=0; i<IMG_DIM * IMG_DIM; ++i) {
#pragma HLS PIPELINE II=1
            raw_image.write(image[i]);
        }

        /* CONV1 */
            // declarations
        data_t conv1_weight[CONV1_OC][CONV1_IC][CONV1_K][CONV1_K];
        data_t conv1_bias[CONV1_OC];
            // load data from HBM
        load_conv_params<CONV1_OC, CONV1_IC, CONV1_K, CONV1_K>(enc_conv1_weight, enc_conv1_bias, conv1_weight, conv1_bias);
            // call conv2d kernel
        conv2d<CONV1_OC, CONV1_IC, CONV1_K, IMG_DIM>(raw_image, conv1_out, conv1_weight, conv1_bias);

        /* POOL1 */
        avg_pool<POOL_SIZE, STRIDE, CONV1_OH, CONV1_OW, CONV1_OC>(conv1_out, pool1_out);

        /* CONV2 */
            // declarations
        data_t conv2_weight[CONV2_OC][CONV2_IC][CONV2_K][CONV2_K];
        data_t conv2_bias[CONV2_OC];
            // load data from HBM
        load_conv_params<CONV2_OC, CONV2_IC, CONV2_K, CONV2_K>(enc_conv2_weight, enc_conv2_bias, conv2_weight, conv2_bias);
            // call conv2d kernel
        conv2d<CONV2_OC, CONV2_IC, CONV2_K, CONV2_IN_DIM>(pool1_out, conv2_out, conv2_weight, conv2_bias);

        /* POOL2 */
        avg_pool<POOL_SIZE, STRIDE, CONV2_OH, CONV2_OW, CONV2_OC>(conv2_out, pool2_out);

        /* FLATTEN */
        flatten<FLAT_IH, FLAT_IW, CONV2_OC>(pool2_out, flat_out);

        /* FULLY CONNECTED LAYER 1 */ 
            // declarations
        data_t fc1_weight[FC1_IN][FC1_OUT];
        data_t fc1_bias[FC1_OUT];
            // load from HBM
        load_fc_params<FC1_IN, FC1_OUT>(enc_fc1_weight, enc_fc1_bias, fc1_weight, fc1_bias);
            // call fc kernel
        fc<FC1_IN, FC1_OUT>(flat_out, fc1_out, fc1_weight, fc1_bias, true);

        /* FULLY CONNECTED LAYER 2 */ 
            // declarations
        data_t fc2_weight[FC2_IN][FC2_OUT];
        data_t fc2_bias[FC2_OUT];
            // load from HBM
        load_fc_params<FC2_IN, FC2_OUT>(enc_fc2_weight, enc_fc2_bias, fc2_weight, fc2_bias);
            // call fc kernel
        fc<FC2_IN, FC2_OUT>(fc1_out, fc2_out, fc2_weight, fc2_bias, true);

        /* FULLY CONNECTED LAYER 3 */ 
            // declarations
        data_t fc3_weight[FC3_IN][FC3_OUT];
        data_t fc3_bias[FC3_OUT];
            // load from HBM
        load_fc_params<FC3_IN, FC3_OUT>(enc_fc3_weight, enc_fc3_bias, fc3_weight, fc3_bias);
            // call fc kernel
        fc<FC3_IN, FC3_OUT>(fc2_out, fc3_out, fc3_weight, fc3_bias, false);

        /* Get final output */
        for(int i=0; i<FC3_OUT; ++i) {
#pragma HLS PIPELINE II=1
            logits[i] = (aes_word_t)fc3_out.read();
        }
    }
}// extern "C"
