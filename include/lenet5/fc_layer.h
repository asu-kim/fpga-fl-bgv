#ifndef FC_H
#define FC_H

//----------------------
// fully connected layer
//----------------------
template<int IN_DIM, int OUT_DIM>
void fc(
        const data_ap_fixed_t in_data[IN_DIM],
        data_ap_fixed_t out_data[OUT_DIM],
        const data_ap_fixed_t weight[IN_DIM*OUT_DIM],
        const data_ap_fixed_t bias[OUT_DIM],
        bool use_relu = true
        ) {
    // #pragma HLS INLINE OFF
    // #pragma HLS ARRAY_PARTITION variable=weight cyclic factor=4 dim=1
    // #pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    data_ap_fixed_t line_buffer[OUT_DIM];
    // #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    for(int j=0; j<OUT_DIM; ++j) {
        #pragma HLS PIPELINE II=1
        line_buffer[j] = bias[j];
    }

    for(int i=0; i<IN_DIM; ++i) {
        data_ap_fixed_t val = in_data[i];
        for(int j=0; j<OUT_DIM; ++j) {
            #pragma HLS PIPELINE II=1
            line_buffer[j] += val * weight[i*OUT_DIM + j];
        }
    }

    // post activation
    for(int j=0; j<OUT_DIM; ++j) {
        #pragma HLS PIPELINE II=1
        out_data[j] = use_relu ? (line_buffer[j] > 0 ? line_buffer[j] : data_ap_fixed_t(0)) : line_buffer[j];
    }
}

#endif
