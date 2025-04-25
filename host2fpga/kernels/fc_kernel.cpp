#include "hls_stream.h"
#include "aes_utils.h"
#include "fc.h"

#define IN_DIM  256
#define OUT_DIM 120
#define PRV_C 16
#define MAX_ROWS 28
#define MAX_COLS 28

extern "C" {
    void fc_kernel(
            const aes_word_t *enc_weights,   // HBM[0]
            const aes_word_t *enc_bias,      // HBM[1] (after weights)
            const aes_word_t *enc_input,     // HBM[2]
            aes_word_t       *enc_output,    // HBM[3]
            bool use_relu
            )
    {
#pragma HLS INTERFACE m_axi port=enc_weights  offset=slave bundle=HBM0
#pragma HLS INTERFACE m_axi port=enc_bias     offset=slave bundle=HBM0
#pragma HLS INTERFACE m_axi port=enc_input    offset=slave bundle=HBM1
#pragma HLS INTERFACE m_axi port=enc_output   offset=slave bundle=HBM2

#pragma HLS INTERFACE s_axilite port=enc_weights  bundle=control
#pragma HLS INTERFACE s_axilite port=enc_bias  bundle=control
#pragma HLS INTERFACE s_axilite port=enc_input bundle=control
#pragma HLS INTERFACE s_axilite port=enc_output bundle=control

#pragma HLS INTERFACE s_axilite port=use_relu bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

        // Decrypt weights + bias into internal BRAMs
        static data_t weight[IN_DIM][OUT_DIM];
        static data_t bias  [OUT_DIM];
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias   complete dim=1

        // total weights for fc layers = IN_DIM * OUT_DIM words
        const int W_TOTAL = IN_DIM * OUT_DIM;
        for(int i=0;i<W_TOTAL;i++) {
#pragma HLS PIPELINE II=1
            int w = decrypt_word(enc_weights[i]); // 120 * 256
            // transpose to IN_DIM * OUT_DIM
            int pos = (i%OUT_DIM) * IN_DIM + (i/OUT_DIM);
            ((data_t*)weight)[pos] = (data_t)w;
        }
        for(int i=0;i<OUT_DIM;i++) {
#pragma HLS PIPELINE II=1
            bias[i] = (data_t)decrypt_word(enc_bias[i]);
        }

        // Stream‑in activations, run fc, stream‑out
        hls::stream<data_t> in_stream("in");
        hls::stream<data_t> out_stream("out");
#pragma HLS STREAM variable=in_stream  depth=1024
#pragma HLS STREAM variable=out_stream depth=1024

        int ACT_TOTAL = IN_DIM;
        // decryption & push to stream
        for(int i=0;i<ACT_TOTAL;i++) {
#pragma HLS PIPELINE II=1
            in_stream.write((data_t)decrypt_word(enc_input[i]));
        }

        // core convolution (template instantiated)
        fc<IN_DIM, OUT_DIM>(in_stream,
                out_stream,
                weight,
                bias,
                use_relu=use_relu);

        // encrypt and write output
        for(int i=0;i<OUT_DIM;i++) {
#pragma HLS PIPELINE II=1
            data_t pv = out_stream.read();
            enc_output[i] = encrypt_word((aes_word_t)pv);
        }
    }
} // extern "C"
