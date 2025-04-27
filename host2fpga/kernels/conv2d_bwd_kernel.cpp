#include "hls_stream.h"
#include "aes_utils.h"
#include "conv2d_bwd.h"

#define OUT_C 6
#define IN_C 1
#define K 5
#define H 28
#define W 28

extern "C" {
    void conv2d_backward_kernel(
            const aes_word_t *x, /* original input */
            const aes_word_t *w, /* original weight */
            const aes_word_t *grads, /* grads from next layer */
            aes_word_t *dX,
            aes_word_t *dW,
            aes_word_t *dB,
            int rows, int cols
            ) {
#pragma HLS INTERFACE m_axi port=x    offset=slave bundle=HBM0
#pragma HLS INTERFACE m_axi port=grads offset=slave bundle=HBM1
#pragma HLS INTERFACE m_axi port=dX   offset=slave bundle=HBM2
#pragma HLS INTERFACE m_axi port=dW   offset=slave bundle=HBM3
#pragma HLS INTERFACE m_axi port=dB   offset=slave bundle=HBM3

#pragma HLS INTERFACE s_axilite port=x    bundle=control
#pragma HLS INTERFACE s_axilite port=grads bundle=control
#pragma HLS INTERFACE s_axilite port=dX   bundle=control
#pragma HLS INTERFACE s_axilite port=dW   bundle=control
#pragma HLS INTERFACE s_axilite port=dB   bundle=control
#pragma HLS INTERFACE s_axilite port=rows     bundle=control
#pragma HLS INTERFACE s_axilite port=cols     bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

        // stream activation
        hls::stream<float> X("x");
#pragma HLS STREAM variable=X depth=1024
        int total_act = IN_C * H * W;
        for(int i=0; i<total_act; ++i) {
#pragma HLS PIPELINE II=1
            X.write(x[i]);
        }

        // stream grads
        hls::stream<float> grads_stream("gs");
#pragma HLS STREAM varibale=grads_stream depth=1024
        int gradH = rows - K + 1, gradW = cols - K + 1;
        int total_grads = OUT_C * gradH * gradW;
        for(int i=0; i<total_grads; ++i) {
            grads_stream.write(grads[i]);
        }

        // param buffers
        static float weight_buffer[OUT_C][IN_C][K][K];
        static float dB_buffer[OUT_C];
        static float dW_buffer[OUT_C][IN_C][K][K];
#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=dB_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=dW_buffer complete dim=1

        int total_weights = OUT_C * IN_C * K * K;
        for(int i=0; i<total_weights; ++i) {
#pragma HLS PIPELINE II=1
            weight_buffer[0][0][0][0];
            ((float*)weight_buffer)[i] = w[i];
        } 

        conv2d_backward<OUT_C, IN_C, K, H, W>(X, grads_stream, X, weight_buffer, dB_buffer, dW_buffer);

        for(int i=0; i<total_act; ++i) {
            dX[i] = (float)X.read();
        }
        for(int i=0; i<total_weights; ++i) {
            dW[i] = dW_buffer[i];
        }
        for(int i=0; i<OUT_C; ++i) {
            dB[i] = dB_buffer[i];
        }
    }
}
