#include <iostream>
#include <cmath>
#include "avg_pool.h"
#include "test_utils.h"

#define IN_ROWS 28
#define IN_COLS 28
#define CHANNELS 6
#define POOL_SIZE 4

#ifndef __SYNTHESIS__
int main() {
    hls::stream<data_t> in_stream, out_stream;

    for(int c=0; c<CHANNELS; ++c) {
        for(int i=0; i<IN_ROWS; ++i) {
            for(int j=0; j<IN_COLS; ++j) {
                data_t val = (j%POOL_SIZE) + 1;
                in_stream.write(val); // 1,2,3,-4,1,2,3,-4

            }
        }
    }

    avg_pool<POOL_SIZE>(in_stream, out_stream, IN_ROWS, IN_COLS, CHANNELS);

    int errs = 0;
    for(int i=0; i<(IN_ROWS/POOL_SIZE) * (IN_COLS/POOL_SIZE) * CHANNELS; ++i) {
        data_t out_val = out_stream.read();
        if (fabs(static_cast<double>(out_val) - 0.5) > 0.01) {
            errs++;
            std::cout << "Error at output " << i << ": got " << out_val << ", expected 2.5" << std::endl;
        }
    }

    std::cout << "Pooling Test: " << (errs ? "Fail" : "Pass") 
        << "(" << errs << " errors)\n";

    return errs;
}

#endif
