// simple encrypt helper
#ifndef AES_UTILS_H
#define AES_UTILS_H

#include <ap_int.h>
#include "stdint.h"

typedef ap_uint<32> aes_word_t;
const aes_word_t AES_KEY = 0x55AA55AAu;

inline aes_word_t encrypt_word(aes_word_t in) {
    #pragma HLS inline
    return in ^ AES_KEY;
}

inline aes_word_t decrypt_word(aes_word_t in) {
    #pragma HLS inline
    return in ^ AES_KEY;
}

#endif
