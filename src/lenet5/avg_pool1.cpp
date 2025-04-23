// #include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/avg_pool.h"
#include "lenet5/avg_pool1.h"

extern "C" {
    void avg_pool1(
        // hls::stream<data_t>& in_stream,
        // hls::stream<data_t>& out_stream,
        data_t* in_data,
        data_t* out_data
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=3456
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=864

        avg_pool<2, 6, 24, 24>(in_data, out_data);
    }
}