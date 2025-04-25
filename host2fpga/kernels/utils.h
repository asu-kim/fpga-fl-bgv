#ifndef UTILS_H
#define UTILS_H

#include "aes_utils.h"

typedef ap_int<32> data_t;

template<int OUT_C, int IN_C, int KH, int KW>
void load_conv_params(const aes_word_t *enc_w,
                      const aes_word_t *enc_b,
                      data_t(&dst_w)[OUT_C][IN_C][KH][KW],
                      data_t (&dst_b)[OUT_C]) {
    for(int oc=0; oc<OUT_C; ++oc) {
        for(int ic=0; ic<IN_C; ++ic) {
            for(int r=0; r<KH; ++r) {
                for(int c=0; c<KW; ++c) {
#pragma HLS PIPELINE II=1
                    int src = oc * IN_C * KH * KW + ic * KH * KW + r * KW + c;
                    dst_w[oc][ic][r][c] = enc_w[src];
                }
            }
        }
    }
    for(int i=0; i<OUT_C; ++i) {
#pragma HLS PIPELINE II=1
        dst_b[i] = enc_b[i];
    }
}

template<int IN_DIM, int OUT_DIM>
void load_fc_params(const aes_word_t *enc_w,
                    const aes_word_t *enc_b,
                    data_t (&dst_w)[IN_DIM][OUT_DIM],
                    data_t (&dst_b)[OUT_DIM]) {
    // note.
    // input layout for fc is OUT_DIM * IN_DIM
    // so we transpose weights for all fc_layres.
    for(int oc=0; oc<OUT_DIM; ++oc) {
        for(int ic=0; ic<IN_DIM; ++ic) {
            int src = oc*IN_DIM + ic;
            dst_w[ic][oc] = enc_w[src];
        }
    }
    for(int i=0; i<OUT_DIM; ++i) {
#pragma HLS PIPELINE II=1
        dst_b[i] = enc_b[i];
    }
}

#endif
