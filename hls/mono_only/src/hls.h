#ifndef HLS_H
#define HLS_H

#include <stdint.h>

#define TILE_K  32
#define TILE_N  32

#define FC1_IN  784
#define FC1_OUT 128
#define FC2_IN  128
#define FC2_OUT  64
#define FC3_IN   64
#define FC3_OUT  10

void mlp(
    int8_t  *A,
    int8_t  *W1, int32_t *b1,
    int8_t  *W2, int32_t *b2,
    int8_t  *W3, int32_t *b3,
    int32_t *C
);

#endif