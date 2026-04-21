/**
 * @file gemm.cpp
 * A parameterizable tiled GEMM accelerator for Vitis HLS.
 *
 * This accelerator compues C = A * B for arbitrary matrix dimensions
 * M x K (A), K x N (B), producing M x N (C). It uses a tiled approach:
 * the hardware works on small fixed-sized sub-matrices (tiles) that fit
 * in on-chip BRAM, while the overall problem size is set at runtime
 * via AXI-Lite registers.
 */

#include "gemm.h"
#include <string.h>

/*
 * Burst-read one tile of matrix A from DRAM into on-chip BRAM.
 * @param A pointer to the full A matrix in DRAM
 * @param A_buffer on-chip BRAM buffer to hold one TILE_M x TILE_K sub-matrix
 * @param row_start row index in the full matrix where this tile begins
 * @param col_start column index in the full matrix where this tile begins
 * @param M, K actual dimensions of the full A matrix
 */
void read_A(
    int8_t *A,
    int8_t A_buffer[TILE_M][TILE_K],
    int row_start,
    int col_start,
    int M,
    int K
) {
    for (int m = 0; m < TILE_M; m++) {
        for (int k = 0; k < TILE_K; k++) {
#pragma HLS PIPELINE II=1
            int r = row_start + m;
            int c = col_start + k;
            // Handle out-of-bounds with zero-padding
            A_buffer[m][k] = (r < M && c < K) ? A[r * K + c] : 0;
        }
    }
}

/*
 * Burst-read one tile of matrix B from DRAM into on-chip BRAM.
 * @param B pointer to the full B matrix in DRAM
 * @param B_buffer on-chip BRAM buffer to hold one TILE_K x TILE_N sub-matrix
 * @param row_start row index in the full matrix where this tile begins
 * @param col_start column index in the full matrix where this tile begins
 * @param K, N actual dimensions of the full A matrix
 */
void read_B(
    int8_t *B,
    int8_t B_buffer[TILE_K][TILE_N],
    int row_start,
    int col_start,
    int K,
    int N
) {
    for (int k = 0; k < TILE_K; k++) {
        for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
            int r = row_start + k;
            int c = col_start + n;
            // Handle out-of-bounds with zero-padding
            B_buffer[k][n] = (r < K && c < N) ? B[r * N + c] : 0;
        }
    }
}

/*
 * Burst-write one completed tile of C from on-chip BRAM back to DRAM.
 * @param A pointer to the full A matrix in DRAM
 * @param A_buffer on-chip BRAM buffer to hold one TILE_M x TILE_K sub-matrix
 * @param row_start row index in the full matrix where this tile begins
 * @param col_start column index in the full matrix where this tile begins
 * @param M, K actual dimensions of the full A matrix
 */
void write_C(
    int *C,
    int C_buffer[TILE_M][TILE_N],
    int row_start,
    int col_start,
    int M,
    int N
) {
    for (int m = 0; m < TILE_M; m++) {
        for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
            int r = row_start + m;
            int c = col_start + n;
            // Only write elements within matrix bounds
            if (r < M && c < N) {
                C[r * N + c] = C_buffer[m][n];
            }
        }
    }
}

/*
 * Computes a partial GEMM for one tile: C_buffer += A_buffer * B_buffer.
 * or C_buffer = A_buffer * B_buffer if this is the first k-tile (i.e. first_k_tile is true)
 */
void compute(
    int8_t A_buffer[TILE_M][TILE_K], 
    int8_t B_buffer[TILE_K][TILE_N], 
    int C_buffer[TILE_M][TILE_N],
    bool first_k_tile
) {
// Every column index of B_buffer and C_buffer gets its own physical BRAM port
#pragma HLS ARRAY_PARTITION variable=B_buffer dim=2 complete
#pragma HLS ARRAY_PARTITION variable=C_buffer dim=2 complete

    for (int m = 0; m < TILE_M; m++) {     
        for (int k = 0; k < TILE_K; k++) {
#pragma HLS PIPELINE II=1
            // This value of a is used by all n iterations
            int a_val = A_buffer[m][k];
            for (int n = 0; n < TILE_N; n++) {
// Fully unroll the n-loop so all TILE_N multiplications happen in the same clock cycle
#pragma HLS UNROLL
                // Start at 0 if this is the first k-tile and k is 0
                int prev = (first_k_tile && k == 0) ? 0 : C_buffer[m][n];
                // Mulitply a_val by one column of B, add to the running sum, and write back
                C_buffer[m][n] = prev + a_val * B_buffer[k][n];
            }
        }
    }
}

/*
 * Top-level accelerator function.
 */
void multiply(int8_t *A, int8_t *B, int *C, int M, int K, int N) {
// Create AXI4 master ports for A, B, and C for reading/writing into/from DRAM
// offset=slave: Base address in DRAM is provided by the host via an AXI-Lite register
// depth: determines the maximum number of elements the port might access
#pragma HLS INTERFACE m_axi port=A depth=16384 offset=slave bundle=in1
#pragma HLS INTERFACE m_axi port=B depth=16384 offset=slave bundle=in2
#pragma HLS INTERFACE m_axi port=C depth=16384 offset=slave bundle=out1

// Exposes the DRAM base-address registers for A, B, C on the AXI-Lite control bus
#pragma HLS INTERFACE s_axilite port=A bundle=CTL
#pragma HLS INTERFACE s_axilite port=B bundle=CTL
#pragma HLS INTERFACE s_axilite port=C bundle=CTL

// Exposes M, K, N as runtime-configurable 32-bit registers on the AXI-Lite control bus
#pragma HLS INTERFACE s_axilite port=M bundle=CTL
#pragma HLS INTERFACE s_axilite port=K bundle=CTL
#pragma HLS INTERFACE s_axilite port=N bundle=CTL

// Maps the accelerator's control signals onto AXI-Lite registers
#pragma HLS INTERFACE s_axilite port=return bundle=CTL

    int8_t A_buffer[TILE_M][TILE_K];
    int8_t B_buffer[TILE_K][TILE_N];
    int C_buffer[TILE_M][TILE_N];

    for (int m = 0; m < M; m += TILE_M) {
        for (int n = 0; n < N; n += TILE_N) {
            for (int k = 0; k < K; k += TILE_K) {
                bool first_k_tile = (k == 0);

                read_A(A, A_buffer, m, k, M, K);
                read_B(B, B_buffer, k, n, K, N);
                compute(A_buffer, B_buffer, C_buffer, first_k_tile);
            }
            write_C(C, C_buffer, m, n, M, N);
        }
    }
}