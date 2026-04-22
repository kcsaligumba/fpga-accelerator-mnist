#include "hls.h"
#include <string.h>

template <int K, int N, int32_t M0, int SHIFT>
static void gemm_tile_relu(
    int8_t        *in,
    const int8_t  *W,
    const int32_t *bias,
    int8_t         out[N]
) {
#pragma HLS INLINE
    for (int n0 = 0; n0 < N; n0 += TILE_N) {

        int32_t acc[TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
            acc[n] = 0;
        }

        // Fused MAC: weights are already resident in partitioned on-chip BRAM
        for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
            int8_t a = in[k];
            for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
                int gn = n0 + n;
                int8_t w = (gn < N) ? W[k * N + gn] : (int8_t)0;
                acc[n] += (int32_t)a * (int32_t)w;
            }
        }

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
            int gn = n0 + n;
            if (gn < N) {
                int32_t v = acc[n] + bias[gn];
                if (v < 0) v = 0; // ReLU
                int64_t scaled = ((int64_t)v * (int64_t)M0
                                  + ((int64_t)1 << (SHIFT - 1))) >> SHIFT; // round half up
                out[gn] = (int8_t)(scaled > 127 ? 127 : scaled);
            }
        }
    }
}


template <int K, int N>
static void gemm_tile_logits(
    int8_t        *in,
    const int8_t  *W,
    const int32_t *bias,
    int32_t        out[N]
) {
#pragma HLS INLINE
    for (int n0 = 0; n0 < N; n0 += TILE_N) {

        int32_t acc[TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
            acc[n] = 0;
        }

        for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
            int8_t a = in[k];
            for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL
                int gn = n0 + n;
                int8_t w = (gn < N) ? W[k * N + gn] : (int8_t)0;
                acc[n] += (int32_t)a * (int32_t)w;
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


    // On-chip mirrors of weights/biases: partitioned so TILE_N=32 reads per cycle can happen inside the fused MAC loop.
    int8_t  W1_bram[FC1_IN * FC1_OUT];
    int8_t  W2_bram[FC2_IN * FC2_OUT];
    int8_t  W3_bram[FC3_IN * FC3_OUT];
    int32_t b1_bram[FC1_OUT];
    int32_t b2_bram[FC2_OUT];
    int32_t b3_bram[FC3_OUT];
#pragma HLS ARRAY_PARTITION variable=W1_bram cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=W2_bram cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=W3_bram cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=b1_bram complete
#pragma HLS ARRAY_PARTITION variable=b2_bram complete
#pragma HLS ARRAY_PARTITION variable=b3_bram complete

    // One-time burst load of weights/biases from DRAM into on-chip BRAM.
    memcpy(W1_bram, W1, sizeof(W1_bram));
    memcpy(b1_bram, b1, sizeof(b1_bram));
    memcpy(W2_bram, W2, sizeof(W2_bram));
    memcpy(b2_bram, b2, sizeof(b2_bram));
    memcpy(W3_bram, W3, sizeof(W3_bram));
    memcpy(b3_bram, b3, sizeof(b3_bram));

    int8_t act1[FC1_OUT];
    int8_t act2[FC2_OUT];
#pragma HLS ARRAY_PARTITION variable=act1 complete
#pragma HLS ARRAY_PARTITION variable=act2 complete

    // FC1
    gemm_tile_relu<FC1_IN, FC1_OUT, FC1_M0, REQUANT_SHIFT>(A, W1_bram, b1_bram, act1);

    // FC2
    gemm_tile_relu<FC2_IN, FC2_OUT, FC2_M0, REQUANT_SHIFT>(act1, W2_bram, b2_bram, act2);

    // FC3
    gemm_tile_logits<FC3_IN, FC3_OUT>(act2, W3_bram, b3_bram, C);
}