#ifndef READER_H
#define READER_H

#include <hls_stream.h>

template<int N>
void mem2stream(
        const float *ddr,
        hls::stream<float> &out_stream
        ) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    for(int i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
        float val = ddr[i];
        out_stream.write(i);
    }
}

// wrapper
template<int N>
void stream_from_ddr(
        const float *ddr,
        hls::stream<float> &out_stream
        ) {
#pragma HLS INLINE
    mem2stream<N>(ddr, out_stream);
}

template<int depth=1024>
void stream_buffer(
        hls::stream<float> &in_stream,
        hls::stream<float> &out1,
        hls::stream<float> &out2
        ) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS STREAM variable=in_stream depth=depth
#pragma HLS STREAM variable=out1 depth=depth
#pragma HLS STREAM variable=out2 depth=depth

    while(!in_stream.empty()) {
        float val = in_stream.read();
        out1.write(val);
        out2.write(val);
    }
}

#endif
