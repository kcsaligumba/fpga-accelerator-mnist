#include <stdint.h>

#ifndef GEMM_H
#define GEMM_H

// Tile dimensions, tuned to fit in BRAM on the xc7z020
#define TILE_M 32
#define TILE_K 32
#define TILE_N 32

// Maximum supported matrix dimension
#define MAX_DIM 512

void multiply(int8_t *A, int8_t *B, int *C, int M, int K, int N);

#endif