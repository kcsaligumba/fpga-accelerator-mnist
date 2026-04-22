
#include "hardcode.h"
#include "hardcode_params.h"


template <int K, int N, int32_t M0, int SHIFT>
static void gemm_tile_relu(
    int8_t        *in,
    const int8_t  *W_rom,
    const int32_t *bias,
    int8_t         out[N]
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
                    W_buf[k][n] = (gk < K && gn < N)
                                    ? W_rom[gk * N + gn]
                                    : (int8_t)0;
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
    const int8_t  *W_rom,
    const int32_t *bias,
    int32_t        out[N]
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
                    W_buf[k][n] = (gk < K && gn < N)
                                    ? W_rom[gk * N + gn]
                                    : (int8_t)0;
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


void mlp(int8_t *A, int32_t *C) {
#pragma HLS INTERFACE m_axi port=A depth=784 offset=slave bundle=in_a
#pragma HLS INTERFACE m_axi port=C depth=10  offset=slave bundle=out_c

#pragma HLS INTERFACE s_axilite port=A      bundle=CTL
#pragma HLS INTERFACE s_axilite port=C      bundle=CTL
#pragma HLS INTERFACE s_axilite port=return bundle=CTL

   
    int8_t act1[FC1_OUT];
    int8_t act2[FC2_OUT];
#pragma HLS ARRAY_PARTITION variable=act1 complete
#pragma HLS ARRAY_PARTITION variable=act2 complete

    // FC1
    gemm_tile_relu<FC1_IN, FC1_OUT, FC1_M0, REQUANT_SHIFT>(A, W1, b1, act1);

    // FC2
    gemm_tile_relu<FC2_IN, FC2_OUT, FC2_M0, REQUANT_SHIFT>(act1, W2, b2, act2);

    // FC3
    gemm_tile_logits<FC3_IN, FC3_OUT>(act2, W3, b3, C);
}