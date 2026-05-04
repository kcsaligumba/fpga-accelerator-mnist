/**
 * @file hardcode.cpp
 * Monolithic MLP inference accelerator with weights hardcoded into the bitstream as ROM.
 *
 * Implements a 3-layer fully-connected network (784->128->64->10) for MNIST digit
 * classification using symmetric INT8 quantization throughout. All weight and bias
 * arrays are declared as static const in hardcode_params.h and synthesized directly
 * into on-chip BRAM at compile time, no weight DMA occurs at runtime. The top-level
 * function accepts a batch of M input images and writes Mx10 INT32 logit vectors.
 */

#include "hardcode.h"        // Layer dimensions, tile sizes, and quantization constants
#include "hardcode_params.h" // Static const INT8 weight ROMs and INT32 bias arrays for all three layers


/*
 * Compute one fully-connected layer with ReLU activation and INT8 requantization.
 * Processes the output dimension in tiles of TILE_N neurons so that TILE_N parallel
 * MAC units operate every clock cycle at II=1.
 *
 * @param in    Pointer to the INT8 input activation vector of length K
 * @param W_rom Pointer to the INT8 weight matrix of shape K×N, stored row-major
 * @param bias  Pointer to the INT32 bias vector of length N (pre-scaled to INT32 domain)
 * @param out   Output buffer of length N that receives the requantized INT8 activations
 *
 * Template parameters:
 *   K     - number of input neurons (reduction dimension)
 *   N     - number of output neurons
 *   M0    - fixed-point requantization multiplier (replaces a float division)
 *   SHIFT - right-shift amount applied after the M0 multiply to land back in INT8 range
 */
template <int K, int N, int32_t M0, int SHIFT>
static void gemm_tile_relu(
    int8_t        *in,      // INT8 input activations, length K
    const int8_t  *W_rom,   // INT8 weight ROM, shape K×N row-major
    const int32_t *bias,    // INT32 pre-scaled biases, length N
    int8_t         out[N]   // INT8 output activations, length N
) {
#pragma HLS INLINE // Inline into the caller so HLS can schedule across layer boundaries without a separate II budget
    for (int n0 = 0; n0 < N; n0 += TILE_N) { // Iterate over output neurons in TILE_N-wide strips

        int32_t acc[TILE_N]; // On-chip accumulator registers, one per output neuron in the current tile
#pragma HLS ARRAY_PARTITION variable=acc complete // Map each acc[n] to its own register so all TILE_N values can be written in the same cycle

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL // Unroll fully: generate TILE_N parallel zero-assignments in hardware
            acc[n] = 0; // Zero the accumulator before beginning the dot product for this tile
        }

        // Fused MAC: read weights directly from the partitioned on-chip ROM
        // TILE_N parallel reads per cycle land in TILE_N different banks
        for (int k = 0; k < K; k++) { // Iterate over every input activation (the reduction dimension)
#pragma HLS PIPELINE II=1 // Accept a new value of k every clock cycle; inner n-loop is unrolled into combinational logic
            int8_t a = in[k]; // Load one input activation; broadcast to all TILE_N multipliers this cycle
            for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL // Unroll fully: TILE_N weight reads and multiplications are issued in the same cycle
                int gn = n0 + n; // Compute the global output-neuron index within the full N dimension
                int8_t w = (gn < N) ? W_rom[k * N + gn] : (int8_t)0; // Read weight from ROM; zero-pad if the tile extends past the true output width
                acc[n] += (int32_t)a * (int32_t)w; // Widen both operands to INT32 before multiplying to prevent overflow, then accumulate
            }
        }

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1 // Process one output neuron per cycle during the post-accumulation pass
            int gn = n0 + n; // Global output-neuron index for this iteration
            if (gn < N) { // Skip any padding lanes that fall outside the true output width
                int32_t v = acc[n] + bias[gn]; // Add the pre-scaled INT32 bias to the INT32 accumulator
                if (v < 0) v = 0; // ReLU activation: clamp any negative result to zero
                int64_t scaled = ((int64_t)v * (int64_t)M0
                                  + ((int64_t)1 << (SHIFT - 1))) >> SHIFT; // Multiply by fixed-point scale M0, add half-LSB for rounding, then right-shift SHIFT bits to map back to INT8 range
                out[gn] = (int8_t)(scaled > 127 ? 127 : scaled); // Saturate at INT8 maximum (127); negatives are already eliminated by ReLU
            }
        }
    }
}


/*
 * Compute the final fully-connected layer (no activation, no requantization).
 * Returns raw INT32 logits so the host can apply argmax or softmax in software
 * without any precision loss from an additional quantization step.
 *
 * @param in    Pointer to the INT8 input activation vector of length K
 * @param W_rom Pointer to the INT8 weight matrix of shape K×N, stored row-major
 * @param bias  Pointer to the INT32 bias vector of length N
 * @param out   Output buffer of length N that receives the INT32 logits
 *
 * Template parameters:
 *   K - number of input neurons
 *   N - number of output classes
 */
template <int K, int N>
static void gemm_tile_logits(
    int8_t        *in,      // INT8 input activations, length K
    const int8_t  *W_rom,   // INT8 weight ROM, shape K×N row-major
    const int32_t *bias,    // INT32 pre-scaled biases, length N
    int32_t        out[N]   // INT32 output logits, length N
) {
#pragma HLS INLINE // Inline into the caller; avoids a separate hardware module boundary
    for (int n0 = 0; n0 < N; n0 += TILE_N) { // Iterate over output classes in TILE_N-wide strips

        int32_t acc[TILE_N]; // On-chip accumulator registers, one per output class in the current tile
#pragma HLS ARRAY_PARTITION variable=acc complete // Map each acc[n] to its own register so all TILE_N values can be updated in the same cycle

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL // Unroll fully: TILE_N parallel zero-assignments in hardware
            acc[n] = 0; // Zero the accumulator before starting the dot product for this tile
        }

        for (int k = 0; k < K; k++) { // Iterate over every input activation (the reduction dimension)
#pragma HLS PIPELINE II=1 // Accept a new value of k every clock cycle; inner n-loop is unrolled
            int8_t a = in[k]; // Load one input activation; broadcast to all TILE_N multipliers this cycle
            for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL // Unroll fully: TILE_N weight reads and multiplications happen in the same cycle
                int gn = n0 + n; // Global output-class index within the full N dimension
                int8_t w = (gn < N) ? W_rom[k * N + gn] : (int8_t)0; // Read weight from ROM; zero-pad if the tile extends past the true output width
                acc[n] += (int32_t)a * (int32_t)w; // Widen to INT32 before multiplying to prevent overflow, then accumulate
            }
        }

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1 // Write one output logit per cycle
            int gn = n0 + n; // Global output-class index
            if (gn < N) // Skip padding lanes that fall outside the true number of classes
                out[gn] = acc[n] + bias[gn]; // Add the INT32 bias to produce the final INT32 logit; no ReLU or requantization
        }
    }
}


/*
 * Top-level accelerator function: batch MLP inference with hardcoded weights.
 *
 * Runs M samples through the 3-layer MLP (FC1→ReLU→FC2→ReLU→FC3) and writes
 * M×10 INT32 logit vectors. Weights and biases for all layers are read directly
 * from on-chip ROM arrays declared in hardcode_params.h; no DRAM weight traffic
 * occurs during inference, so every invocation pays only the cost of reading
 * M×784 input bytes and writing M×10 INT32 output words.
 *
 * @param A Pointer to the flattened INT8 input batch in DRAM, shape M×784
 * @param C Pointer to the INT32 output logit buffer in DRAM, shape M×10
 * @param M Number of samples in the batch (1–128)
 */
void mlp(int8_t *A, int32_t *C, int M) {
#pragma HLS INTERFACE m_axi port=A depth=100352 offset=slave bundle=in_a // AXI4-Master port for reading the input batch from DRAM; depth covers 128 samples × 784 bytes
#pragma HLS INTERFACE m_axi port=C depth=1280   offset=slave bundle=out_c // AXI4-Master port for writing logits to DRAM; depth covers 128 samples × 10 INT32 words

#pragma HLS INTERFACE s_axilite port=A      bundle=CTL // AXI-Lite register: base address of the input buffer in DRAM
#pragma HLS INTERFACE s_axilite port=C      bundle=CTL // AXI-Lite register: base address of the output buffer in DRAM
#pragma HLS INTERFACE s_axilite port=M      bundle=CTL // AXI-Lite register: number of samples in this batch (written by host before AP_START)
#pragma HLS INTERFACE s_axilite port=return bundle=CTL // AXI-Lite register: AP_START, AP_DONE, AP_IDLE control signals

// Cyclic-partition each weight ROM by TILE_N so the inner unrolled N-loop can perform TILE_N parallel reads per cycle at II=1
#pragma HLS ARRAY_PARTITION variable=W1 cyclic factor=32 dim=1 // Split W1 across 32 BRAM banks so 32 consecutive columns can be read simultaneously
#pragma HLS ARRAY_PARTITION variable=W2 cyclic factor=32 dim=1 // Split W2 across 32 BRAM banks for the same reason
#pragma HLS ARRAY_PARTITION variable=W3 cyclic factor=32 dim=1 // Split W3 across 32 BRAM banks for the same reason
#pragma HLS ARRAY_PARTITION variable=b1 complete // Give each b1 element its own register so all 128 bias reads happen in one cycle
#pragma HLS ARRAY_PARTITION variable=b2 complete // Give each b2 element its own register so all 64 bias reads happen in one cycle
#pragma HLS ARRAY_PARTITION variable=b3 complete // Give each b3 element its own register so all 10 bias reads happen in one cycle

    int8_t act1[FC1_OUT]; // Intermediate INT8 activations after FC1+ReLU, length 128
    int8_t act2[FC2_OUT]; // Intermediate INT8 activations after FC2+ReLU, length 64
#pragma HLS ARRAY_PARTITION variable=act1 complete // All 128 act1 elements get their own register for single-cycle read/write
#pragma HLS ARRAY_PARTITION variable=act2 complete // All 64 act2 elements get their own register for single-cycle read/write

    for (int m = 0; m < M; m++) { // Process each sample in the batch sequentially
#pragma HLS LOOP_TRIPCOUNT min=1 max=128 avg=32 // Inform HLS of the expected iteration range for latency/resource reporting; does not affect logic
        gemm_tile_relu<FC1_IN, FC1_OUT, FC1_M0, REQUANT_SHIFT>(A + m * FC1_IN, W1, b1, act1); // FC1: 784→128, reads from DRAM at stride FC1_IN, writes INT8 activations into act1
        gemm_tile_relu<FC2_IN, FC2_OUT, FC2_M0, REQUANT_SHIFT>(act1, W2, b2, act2); // FC2: 128→64, reads act1, writes INT8 activations into act2
        gemm_tile_logits<FC3_IN, FC3_OUT>(act2, W3, b3, C + m * FC3_OUT); // FC3: 64→10, reads act2, writes 10 INT32 logits into the output buffer at stride FC3_OUT
    }
}
