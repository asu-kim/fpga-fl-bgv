#include "hls_stream.h"
#include "aes_utils.h"
#include "conv2d.h"

#define OUT_C 6
#define IN_C  1
#define KSIZE 5
#define IN_DIM 28

extern "C" {
    void conv2d_kernel(
            const aes_word_t *enc_weights,   // HBM[0]
            const aes_word_t *enc_bias,      // HBM[1] (after weights)
            const aes_word_t *enc_input,     // HBM[2]
            aes_word_t       *enc_output,    // HBM[3]
            int rows,
            int cols)
    {
#pragma HLS INTERFACE m_axi port=enc_weights  offset=slave bundle=HBM0
#pragma HLS INTERFACE m_axi port=enc_bias     offset=slave bundle=HBM0
#pragma HLS INTERFACE m_axi port=enc_input    offset=slave bundle=HBM1
#pragma HLS INTERFACE m_axi port=enc_output   offset=slave bundle=HBM2

#pragma HLS INTERFACE s_axilite port=enc_weights  bundle=control
#pragma HLS INTERFACE s_axilite port=enc_bias  bundle=control
#pragma HLS INTERFACE s_axilite port=enc_input bundle=control
#pragma HLS INTERFACE s_axilite port=enc_output bundle=control

#pragma HLS INTERFACE s_axilite port=rows  bundle=control
#pragma HLS INTERFACE s_axilite port=cols  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

        // Decrypt weights + bias into internal BRAMs
        static data_t weight[OUT_C][IN_C][KSIZE][KSIZE];
        static data_t bias  [OUT_C];
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias   complete dim=1

        // total weights = OUT_C*IN_C*KSIZE*KSIZE words
        const int W_TOTAL = OUT_C*IN_C*KSIZE*KSIZE;
        for(int i=0;i<W_TOTAL;i++) {
#pragma HLS PIPELINE II=1
            int w = decrypt_word(enc_weights[i]);
            ((data_t*)weight)[i] = (data_t)w;
        }
        for(int i=0;i<OUT_C;i++) {
#pragma HLS PIPELINE II=1
            bias[i] = (data_t)decrypt_word(enc_bias[i]);
        }

        // Stream‑in activations, run conv2d, stream‑out
        hls::stream<data_t> in_stream("in");
        hls::stream<data_t> out_stream("out");
#pragma HLS STREAM variable=in_stream  depth=1024
#pragma HLS STREAM variable=out_stream depth=1024

        int ACT_TOTAL = rows * cols * IN_C;
        // decryption & push to stream
        for(int i=0;i<ACT_TOTAL;i++) {
#pragma HLS PIPELINE II=1
            in_stream.write((data_t)decrypt_word(enc_input[i]));
        }

        // core convolution (template instantiated)
        conv2d<OUT_C,IN_C,KSIZE,IN_DIM>(in_stream,
                out_stream,
                weight,
                bias);

        // encrypt and write output
        int OUT_PIX = (rows - KSIZE + 1) * (cols - KSIZE + 1) * OUT_C;
        for(int i=0;i<OUT_PIX;i++) {
#pragma HLS PIPELINE II=1
            data_t pv = out_stream.read();
            enc_output[i] = encrypt_word((aes_word_t)pv);
        }
    }
} // extern "C"
