#include "hls.h"

template <int K, int N>
static void gemm_tile_relu(
    int8_t  *in,
    int8_t  *W,
    int32_t *bias,
    int8_t   out[N]
) {
    for (int n0 = 0; n0 < N; n0 += TILE_N) {

        int32_t acc[TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
            acc[n] = 0;
        }

        for (int k0 = 0; k0 < K; k0 += TILE_K) {

            int8_t W_buf[TILE_K][TILE_N];
#pragma HLS ARRAY_PARTITION variable=W_buf dim=2 complete


            for (int k = 0; k < TILE_K; k++) {
                for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
                    int gk = k0 + k;
                    int gn = n0 + n;
                    W_buf[k][n] = (gk < K && gn < N) ? W[gk * N + gn] : (int8_t)0;
                }
            }


            for (int k = 0; k < TILE_K; k++) {
#pragma HLS PIPELINE II=1
                int gk = k0 + k;
                int8_t a = (gk < K) ? in[gk] : (int8_t)0;
                for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
                    acc[n] += (int32_t)a * (int32_t)W_buf[k][n];
                }
            }
        } 

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
            int gn = n0 + n;
            if (gn < N) {
                int32_t v = acc[n] + bias[gn];
                out[gn] = (int8_t)(v > 127 ? 127 : (v < 0 ? 0 : v));
            }
        }
    } 
}


template <int K, int N>
static void gemm_tile_logits(
    int8_t  *in,
    int8_t  *W,
    int32_t *bias,
    int32_t  out[N]
) {
    for (int n0 = 0; n0 < N; n0 += TILE_N) {

        int32_t acc[TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
            acc[n] = 0;
        }

        for (int k0 = 0; k0 < K; k0 += TILE_K) {

            int8_t W_buf[TILE_K][TILE_N];
#pragma HLS ARRAY_PARTITION variable=W_buf dim=2 complete

            for (int k = 0; k < TILE_K; k++) {
                for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
                    int gk = k0 + k;
                    int gn = n0 + n;
                    W_buf[k][n] = (gk < K && gn < N) ? W[gk * N + gn] : (int8_t)0;
                }
            }

            for (int k = 0; k < TILE_K; k++) {
#pragma HLS PIPELINE II=1
                int gk = k0 + k;
                int8_t a = (gk < K) ? in[gk] : (int8_t)0;
                for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
                    acc[n] += (int32_t)a * (int32_t)W_buf[k][n];
                }
            }
        }

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
            int gn = n0 + n;
            if (gn < N)
                out[gn] = acc[n] + bias[gn];
        }
    }
}


void mlp(
    int8_t  *A,
    int8_t  *W1, int32_t *b1,
    int8_t  *W2, int32_t *b2,
    int8_t  *W3, int32_t *b3,
    int32_t *C
) {

#pragma HLS INTERFACE m_axi port=A  depth=784    offset=slave bundle=in_a
#pragma HLS INTERFACE m_axi port=W1 depth=100352 offset=slave bundle=in_w1
#pragma HLS INTERFACE m_axi port=b1 depth=128    offset=slave bundle=in_w1
#pragma HLS INTERFACE m_axi port=W2 depth=8192   offset=slave bundle=in_w2
#pragma HLS INTERFACE m_axi port=b2 depth=64     offset=slave bundle=in_w2
#pragma HLS INTERFACE m_axi port=W3 depth=640    offset=slave bundle=in_w3
#pragma HLS INTERFACE m_axi port=b3 depth=10     offset=slave bundle=in_w3
#pragma HLS INTERFACE m_axi port=C  depth=10     offset=slave bundle=out_c

#pragma HLS INTERFACE s_axilite port=A      bundle=CTL
#pragma HLS INTERFACE s_axilite port=W1     bundle=CTL
#pragma HLS INTERFACE s_axilite port=b1     bundle=CTL
#pragma HLS INTERFACE s_axilite port=W2     bundle=CTL
#pragma HLS INTERFACE s_axilite port=b2     bundle=CTL
#pragma HLS INTERFACE s_axilite port=W3     bundle=CTL
#pragma HLS INTERFACE s_axilite port=b3     bundle=CTL
#pragma HLS INTERFACE s_axilite port=C      bundle=CTL
#pragma HLS INTERFACE s_axilite port=return bundle=CTL


    int8_t act1[FC1_OUT];
    int8_t act2[FC2_OUT];
#pragma HLS ARRAY_PARTITION variable=act1 complete
#pragma HLS ARRAY_PARTITION variable=act2 complete

    // FC1
    gemm_tile_relu<FC1_IN, FC1_OUT>(A, W1, b1, act1);

    // FC2
    gemm_tile_relu<FC2_IN, FC2_OUT>(act1, W2, b2, act2);

    // FC3
    gemm_tile_logits<FC3_IN, FC3_OUT>(act2, W3, b3, C);
}