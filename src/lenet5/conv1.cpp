#include "hls_stream.h"
#include "constants.hpp"
#include "data_type.hpp"
#include "lenet5/conv2d.h"
#include "lenet5/conv1.h"

extern "C" {
    // void conv1
    // // (
    // //     const data_t in_data[28*28],
	// // 	data_t out_data[6*24*24],
	// // 	const data_t conv1_weight[6*5*5],
	// // 	const data_t conv1_bias[6]
	// // )
    // (
    //     data_t in_data[28*28],
    //     data_t out_data[6*24*24],
    //     data_t conv1_weight[6*5*5],
    //     data_t conv1_bias[6]
    // )
    // {
    //     float act_out_scale=1;
    //     float act_out_zp=0;
    //     data_t IBRAM[28][28];
    //     data_t WBRAM[6][5][5];
    //     data_t biasBRAM[6];
    //     data_t OBRAM[6][24*24];
    // #pragma HLS array_partition variable=WBRAM complete dim=1
    // #pragma HLS array_partition variable=biasBRAM complete dim=0
    // #pragma HLS array_partition variable=OBRAM complete dim=2

    //     copy_input_1:
    //     for(int i=0;i<28;i++){
    //         copy_input_2 :
    //         for(int j=0;j<28;j++){
    //         #pragma HLS PIPELINE II=1
    //             IBRAM[i][j] = in_data[i*28 + j];
    //         }
    //     }

    //     // load weights & bias at first iteration only
    //     copy_kernel_1:
    //     for(int i=0;i<6;i++){
    //         copy_kernel_2:
    //         for(int j=0;j<5;j++){
    //             for(int k=0;k<5;k++){
    //             #pragma HLS PIPELINE II=1
    //                 WBRAM[i][j][k] = conv1_weight[i*25+j*5+k];
    //             }
    //         }
    //     }


    //     copy_bias:
    //     for(int i=0;i<6;i++){
    // #pragma HLS PIPELINE II=1
    //         biasBRAM[i] = conv1_bias[i];
    //     }

    //     //////////////////////////////////////////////////////////////////////
    //     //						   Convolution								//
    //     //////////////////////////////////////////////////////////////////////
    //     ROW_K:
    //     for(int row_k=0;row_k<5;row_k++){
    //         COL_K:
    //         for(int col_k=0;col_k<5;col_k++){
    //             ROW :
    //             for (int row = 0; row < 24; row++) {
    //                 COL :
    //                 for (int col = 0; col < 24; col++) {
    //                 #pragma HLS PIPELINE II=1
    //                     data_t input_pixel = IBRAM[row+row_k][col+col_k];
    //                     data_t mult[6];
    //                     #pragma HLS array_partition variable=mult complete dim=0
    //                     D_OUT:
    //                     for(int co=0;co<6;co++){
    //                     #pragma HLS unroll
    //                         mult[co] = input_pixel*WBRAM[co][row_k][col_k];
    //                         if(row_k==0&&col_k==0)
    //                             OBRAM[co][row*24+col] = mult[co];
    //                         else
    //                             OBRAM[co][row*24+col] += mult[co];
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     copy_output:
    //     for(int i=0; i<6; i++){
    //         for(int j=0; j<24*24; j++){
    // #pragma HLS PIPELINE II=2
    //             data_t sum = OBRAM[i][j]+biasBRAM[i];
    //             float sum_float_val = float(sum);
    //             float sum_scaled = sum_float_val * act_out_scale + act_out_zp;
    //             float sum_rounded = hls::floor(sum_scaled+0.5f);
    //             data_t sum_clipped = (data_t)sum_rounded;
    //             sum_clipped = hls::max(MIN_VAL, hls::min(MAX_VAL, sum_clipped));
    //             out_data[i*784+j]=sum_clipped;
    //         }
    //     }
    // }
    void conv1(
        // hls::stream<data_t>& in_stream,
        // hls::stream<data_t>& out_stream,
        data_t* in_data,
        data_t* out_data,
        data_t* conv1_weight,
        data_t* conv1_bias
    ) { 
        #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784
        #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=3456
        #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem3 depth=256
        #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=128

        // Create local copy
        data_t local_weight[150];
        data_t local_bias[6];

        // Copy data
        for(int i=0; i<150; i++) {
            local_weight[i] = conv1_weight[i];
        }
        for(int i=0; i<6; i++) {
            local_bias[i] = conv1_bias[i];
        }

        conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
    }
    // void conv1(
    //     // hls::stream<data_t>& in_stream,
    //     // hls::stream<data_t>& out_stream,
    //     data_t* in_data,
    //     data_t* out_data,
    //     data_t* conv1_weight,
    //     data_t* conv1_bias
    // ) { 
    //     #pragma HLS INTERFACE m_axi port=in_data bundle=gmem0 depth=784
    //     #pragma HLS INTERFACE m_axi port=out_data bundle=gmem1 depth=2304
    //     #pragma HLS INTERFACE m_axi port=conv1_weight bundle=gmem3 depth=128
    //     #pragma HLS INTERFACE m_axi port=conv1_bias bundle=gmem3 depth=128

    //     // Create local copy
    //     data_t weight_ch1[25];
    //     data_t weight_ch2[25];
    //     data_t weight_ch3[25];
    //     data_t weight_ch4[25];
    //     data_t bias_ch1[1];
    //     data_t bias_ch2[1];
    //     data_t bias_ch3[1];
    //     data_t bias_ch4[1];

    //     // Copy data
    //     // for(int i=0; i<4; i++) {
    //     //     local_bias[i] = conv1_bias[i];
    //     // }

    //     bias_ch1[0] = conv1_bias[0];
    //     bias_ch2[0] = conv1_bias[1];
    //     bias_ch3[0] = conv1_bias[2];
    //     bias_ch4[0] = conv1_bias[3];

    //     for(int j=0; j<1; j++) {
    //         for(int k=0; k<5; k++) {
    //             for(int l=0; l<5; l++) {
    //                 weight_ch1[k*5+l] = conv1_weight[j * 25 + k * 5 + l];
    //             }
    //         }
    //     }

    //     for(int j=0; j<1; j++) {
    //         for(int k=0; k<5; k++) {
    //             for(int l=0; l<5; l++) {
    //                 weight_ch2[k*5+l] = conv1_weight[25 + j * 25 + k * 5 + l];
    //             }
    //         }
    //     }

    //     for(int j=0; j<1; j++) {
    //         for(int k=0; k<5; k++) {
    //             for(int l=0; l<5; l++) {
    //                 weight_ch3[k*5+l] = conv1_weight[50 + j * 25 + k * 5 + l];
    //             }
    //         }
    //     }

    //     for(int j=0; j<1; j++) {
    //         for(int k=0; k<5; k++) {
    //             for(int l=0; l<5; l++) {
    //                 weight_ch4[k*5+l] = conv1_weight[75 + j * 25 + k * 5 + l];
    //             }
    //         }
    //     }

    //     data_t out_ch1[576];
    //     data_t out_ch2[576];
    //     data_t out_ch3[576];
    //     data_t out_ch4[576];

    //     // conv2d<6, 1, 5, 28, 28>(in_data, out_data, local_weight, local_bias);
    //     conv2d<1, 1, 5, 28, 28>(in_data, out_ch1, weight_ch1, bias_ch1);
    //     conv2d<1, 1, 5, 28, 28>(in_data, out_ch2, weight_ch2, bias_ch2);
    //     conv2d<1, 1, 5, 28, 28>(in_data, out_ch3, weight_ch3, bias_ch3);
    //     conv2d<1, 1, 5, 28, 28>(in_data, out_ch4, weight_ch4, bias_ch4);

    //     for(int i = 0; i < 576; i++) {
    //         out_data[i] = out_ch1[i];
    //     }
    //     for(int i = 0; i < 576; i++) {
    //         out_data[i + 576] = out_ch2[i];
    //     }
    //     for(int i = 0; i < 576; i++) {
    //         out_data[i + 1152] = out_ch3[i];
    //     }
    //     for(int i = 0; i < 576; i++) {
    //         out_data[i + 1728] = out_ch4[i];
    //     }
    // }
}