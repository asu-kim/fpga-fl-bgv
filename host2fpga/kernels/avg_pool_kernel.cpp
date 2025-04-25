#include "hls_stream.h"
#include "aes_utils.h"
#include "avg_pool.h"

#define STRIDE 2
#define POOLSIZE 2
#define IN_HEIGHT 24
#define IN_WIDTH 24
#define IN_C 6

extern "C" {
    void avg_pool_kernel(
            const aes_word_t *enc_input,     // HBM[0]
            aes_word_t       *enc_output     // HBM[1]
            )
    {
#pragma HLS INTERFACE m_axi port=enc_input    offset=slave bundle=HBM1
#pragma HLS INTERFACE m_axi port=enc_output   offset=slave bundle=HBM2

#pragma HLS INTERFACE s_axilite port=enc_input bundle=control
#pragma HLS INTERFACE s_axilite port=enc_output bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

        // Stream‑in activations, run fc, stream‑out
        hls::stream<data_t> in_stream("in");
        hls::stream<data_t> out_stream("out");
#pragma HLS STREAM variable=in_stream  depth=1024
#pragma HLS STREAM variable=out_stream depth=1024

        int ACT_TOTAL = IN_HEIGHT * IN_WIDTH * IN_C;
        // decryption & push to stream
        for(int i=0;i<ACT_TOTAL;i++) {
#pragma HLS PIPELINE II=1
            in_stream.write((data_t)decrypt_word(enc_input[i]));
        }

        // core convolution (template instantiated)
        avg_pool<POOLSIZE, STRIDE, IN_HEIGHT, IN_WIDTH, IN_C>(in_stream, out_stream);

        // encrypt and write output
        int OUT_PIX = (IN_HEIGHT / POOLSIZE) * (IN_WIDTH / POOLSIZE) * IN_C;
        for(int i=0;i<OUT_PIX;i++) {
#pragma HLS PIPELINE II=1
            data_t pv = out_stream.read();
            enc_output[i] = encrypt_word((aes_word_t)pv);
        }
    }
} // extern "C"
