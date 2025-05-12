#ifndef AVG_POOL_BWD_H
#define AVG_POOL_BWD_H

#include "data_type.hpp"

template<int POOL_SIZE, int STRIDE, int IN_C, int IN_ROW, int IN_COL>
void avg_pool_backward(
        const data_ap_fixed_t grads[IN_C*((IN_ROW-POOL_SIZE)/STRIDE+1)*((IN_COL-POOL_SIZE)/STRIDE+1)],
        data_ap_fixed_t dX[IN_C*IN_ROW*IN_COL]
        ) {
// #pragma HLS INLINE off

    constexpr int pooled_row = (IN_ROW-POOL_SIZE)/STRIDE+1;
    constexpr int pooled_col = (IN_COL-POOL_SIZE)/STRIDE+1;
    const data_ap_fixed_t inverse = data_ap_fixed_t(1.0) / (data_ap_fixed_t)(POOL_SIZE*POOL_SIZE);

    data_ap_fixed_t x_buffer[IN_C*IN_ROW*IN_COL];
    for(int i=0; i<IN_C*IN_ROW*IN_COL; i++) {
        x_buffer[i] = 0;
    }

    for(int k=0; k<IN_C; ++k) {
        for(int r=0; r<pooled_row; ++r) {
            for(int c=0; c<pooled_col; ++c) {
                data_ap_fixed_t grad = grads[k*(pooled_row*pooled_col) + r*pooled_col + c] * inverse;
                
                for(int pr=0; pr<POOL_SIZE; ++pr) {
                    for(int pc=0; pc<POOL_SIZE; ++pc) {
                        int row_idx = r*STRIDE+pr;
                        int col_idx = c*STRIDE+pc;
                        
                        if(row_idx < IN_ROW && col_idx < IN_COL) {
                            int idx = k*(IN_ROW*IN_COL) + row_idx*IN_COL + col_idx;
                            x_buffer[idx] += grad;
                        }
                    }
                }
            }
        }
    }

    for(int i=0; i<IN_C*IN_ROW*IN_COL; i++) {
        dX[i] = x_buffer[i];
    }
}

#endif