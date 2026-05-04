/**
 * @file dynamic.cpp
 * Monolithic MLP inference accelerator with weights loaded dynamically from DRAM.
 *
 * Implements the same 3-layer fully-connected network (784→128→64→10) as the
 * hardcoded variant, but receives all weight and bias arrays as AXI pointer
 * arguments at runtime. On each invocation the weights are burst-copied from DRAM
 * into partitioned on-chip BRAM once, then reused across all M samples in the batch.
 * This amortizes the weight-load cost over the batch while keeping the per-sample
 * compute path identical to the hardcoded design.
 */

#include "dynamic.h" // Layer dimensions, tile sizes, and quantization constants
#include <string.h>  // memcpy for burst-loading weights from DRAM into on-chip BRAM


/*
 * Compute one fully-connected layer with ReLU activation and INT8 requantization.
 * Processes the output dimension in tiles of TILE_N neurons so that TILE_N parallel
 * MAC units operate every clock cycle at II=1.
 *
 * @param in   Pointer to the INT8 input activation vector of length K
 * @param W    Pointer to the INT8 weight matrix of shape K×N, stored row-major in on-chip BRAM
 * @param bias Pointer to the INT32 bias vector of length N (pre-scaled to INT32 domain)
 * @param out  Output buffer of length N that receives the requantized INT8 activations
 *
 * Template parameters:
 *   K     - number of input neurons (reduction dimension)
 *   N     - number of output neurons
 *   M0    - fixed-point requantization multiplier (replaces a float division)
 *   SHIFT - right-shift amount applied after the M0 multiply to land back in INT8 range
 */
template <int K, int N, int32_t M0, int SHIFT>
static void gemm_tile_relu(
    int8_t        *in,    // INT8 input activations, length K
    const int8_t  *W,     // INT8 weight matrix in on-chip BRAM, shape K×N row-major
    const int32_t *bias,  // INT32 pre-scaled biases in on-chip BRAM, length N
    int8_t         out[N] // INT8 output activations, length N
) {
#pragma HLS INLINE // Inline into the caller so HLS can schedule across layer boundaries without a separate II budget
    for (int n0 = 0; n0 < N; n0 += TILE_N) { // Iterate over output neurons in TILE_N-wide strips

        int32_t acc[TILE_N]; // On-chip accumulator registers, one per output neuron in the current tile
#pragma HLS ARRAY_PARTITION variable=acc complete // Map each acc[n] to its own register so all TILE_N values can be written in the same cycle

        for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL // Unroll fully: generate TILE_N parallel zero-assignments in hardware
            acc[n] = 0; // Zero the accumulator before beginning the dot product for this tile
        }

        // Fused MAC: weights are already resident in partitioned on-chip BRAM
        for (int k = 0; k < K; k++) { // Iterate over every input activation (the reduction dimension)
#pragma HLS PIPELINE II=1 // Accept a new value of k every clock cycle; inner n-loop is unrolled into combinational logic
            int8_t a = in[k]; // Load one input activation; broadcast to all TILE_N multipliers this cycle
            for (int n = 0; n < TILE_N; n++) {
#pragma HLS UNROLL // Unroll fully: TILE_N weight reads and multiplications are issued in the same cycle
                int gn = n0 + n; // Compute the global output-neuron index within the full N dimension
                int8_t w = (gn < N) ? W[k * N + gn] : (int8_t)0; // Read weight from BRAM; zero-pad if the tile extends past the true output width
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
 * @param in   Pointer to the INT8 input activation vector of length K
 * @param W    Pointer to the INT8 weight matrix of shape K×N in on-chip BRAM, stored row-major
 * @param bias Pointer to the INT32 bias vector of length N in on-chip BRAM
 * @param out  Output buffer of length N that receives the INT32 logits
 *
 * Template parameters:
 *   K - number of input neurons
 *   N - number of output classes
 */
template <int K, int N>
static void gemm_tile_logits(
    int8_t        *in,    // INT8 input activations, length K
    const int8_t  *W,     // INT8 weight matrix in on-chip BRAM, shape K×N row-major
    const int32_t *bias,  // INT32 pre-scaled biases in on-chip BRAM, length N
    int32_t        out[N] // INT32 output logits, length N
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
                int8_t w = (gn < N) ? W[k * N + gn] : (int8_t)0; // Read weight from BRAM; zero-pad if the tile extends past the true output width
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
 * Top-level accelerator function: batch MLP inference with dynamically loaded weights.
 *
 * Accepts weight and bias pointers for all three layers as AXI arguments. On each
 * invocation, all weights and biases are burst-copied from DRAM into partitioned
 * on-chip BRAM once before the batch loop. This one-time transfer is then amortized
 * across all M samples, so per-sample latency drops as M increases. After loading,
 * each sample is processed identically to the hardcoded design.
 *
 * @param A  Pointer to the flattened INT8 input batch in DRAM, shape M×784
 * @param W1 Pointer to FC1 INT8 weight matrix in DRAM, shape 784×128
 * @param b1 Pointer to FC1 INT32 bias vector in DRAM, length 128
 * @param W2 Pointer to FC2 INT8 weight matrix in DRAM, shape 128×64
 * @param b2 Pointer to FC2 INT32 bias vector in DRAM, length 64
 * @param W3 Pointer to FC3 INT8 weight matrix in DRAM, shape 64×10
 * @param b3 Pointer to FC3 INT32 bias vector in DRAM, length 10
 * @param C  Pointer to the INT32 output logit buffer in DRAM, shape M×10
 * @param M  Number of samples in the batch (1–128)
 */
void mlp(
    int8_t  *A,
    int8_t  *W1, int32_t *b1,
    int8_t  *W2, int32_t *b2,
    int8_t  *W3, int32_t *b3,
    int32_t *C,
    int M
) {

#pragma HLS INTERFACE m_axi port=A  depth=100352 offset=slave bundle=in_a  // AXI4-Master port for reading the input batch; depth covers 128 samples × 784 bytes
#pragma HLS INTERFACE m_axi port=W1 depth=100352 offset=slave bundle=in_w1 // AXI4-Master port for reading FC1 weights (784×128 = 100352 bytes); shares a bus with b1
#pragma HLS INTERFACE m_axi port=b1 depth=128    offset=slave bundle=in_w1 // AXI4-Master port for reading FC1 biases (128 INT32 words); shares a bus with W1
#pragma HLS INTERFACE m_axi port=W2 depth=8192   offset=slave bundle=in_w2 // AXI4-Master port for reading FC2 weights (128×64 = 8192 bytes); shares a bus with b2
#pragma HLS INTERFACE m_axi port=b2 depth=64     offset=slave bundle=in_w2 // AXI4-Master port for reading FC2 biases (64 INT32 words); shares a bus with W2
#pragma HLS INTERFACE m_axi port=W3 depth=640    offset=slave bundle=in_w3 // AXI4-Master port for reading FC3 weights (64×10 = 640 bytes); shares a bus with b3
#pragma HLS INTERFACE m_axi port=b3 depth=10     offset=slave bundle=in_w3 // AXI4-Master port for reading FC3 biases (10 INT32 words); shares a bus with W3
#pragma HLS INTERFACE m_axi port=C  depth=1280   offset=slave bundle=out_c // AXI4-Master port for writing logits to DRAM; depth covers 128 samples × 10 INT32 words

#pragma HLS INTERFACE s_axilite port=A      bundle=CTL // AXI-Lite register: base address of the input batch in DRAM
#pragma HLS INTERFACE s_axilite port=W1     bundle=CTL // AXI-Lite register: base address of the FC1 weight matrix in DRAM
#pragma HLS INTERFACE s_axilite port=b1     bundle=CTL // AXI-Lite register: base address of the FC1 bias vector in DRAM
#pragma HLS INTERFACE s_axilite port=W2     bundle=CTL // AXI-Lite register: base address of the FC2 weight matrix in DRAM
#pragma HLS INTERFACE s_axilite port=b2     bundle=CTL // AXI-Lite register: base address of the FC2 bias vector in DRAM
#pragma HLS INTERFACE s_axilite port=W3     bundle=CTL // AXI-Lite register: base address of the FC3 weight matrix in DRAM
#pragma HLS INTERFACE s_axilite port=b3     bundle=CTL // AXI-Lite register: base address of the FC3 bias vector in DRAM
#pragma HLS INTERFACE s_axilite port=C      bundle=CTL // AXI-Lite register: base address of the output logit buffer in DRAM
#pragma HLS INTERFACE s_axilite port=M      bundle=CTL // AXI-Lite register: number of samples in this batch (written by host before AP_START)
#pragma HLS INTERFACE s_axilite port=return bundle=CTL // AXI-Lite register: AP_START, AP_DONE, AP_IDLE control signals

    // On-chip mirrors of weights/biases: partitioned so TILE_N=32 reads per cycle can happen inside the fused MAC loop.
    int8_t  W1_bram[FC1_IN * FC1_OUT]; // On-chip BRAM mirror of FC1 weights, shape 784×128
    int8_t  W2_bram[FC2_IN * FC2_OUT]; // On-chip BRAM mirror of FC2 weights, shape 128×64
    int8_t  W3_bram[FC3_IN * FC3_OUT]; // On-chip BRAM mirror of FC3 weights, shape 64×10
    int32_t b1_bram[FC1_OUT];           // On-chip BRAM mirror of FC1 biases, length 128
    int32_t b2_bram[FC2_OUT];           // On-chip BRAM mirror of FC2 biases, length 64
    int32_t b3_bram[FC3_OUT];           // On-chip BRAM mirror of FC3 biases, length 10
#pragma HLS ARRAY_PARTITION variable=W1_bram cyclic factor=32 dim=1 // Split W1_bram across 32 BRAM banks so TILE_N=32 consecutive columns can be read simultaneously
#pragma HLS ARRAY_PARTITION variable=W2_bram cyclic factor=32 dim=1 // Split W2_bram across 32 BRAM banks for the same reason
#pragma HLS ARRAY_PARTITION variable=W3_bram cyclic factor=32 dim=1 // Split W3_bram across 32 BRAM banks for the same reason
#pragma HLS ARRAY_PARTITION variable=b1_bram complete // Give each b1_bram element its own register so all 128 bias reads happen in one cycle
#pragma HLS ARRAY_PARTITION variable=b2_bram complete // Give each b2_bram element its own register so all 64 bias reads happen in one cycle
#pragma HLS ARRAY_PARTITION variable=b3_bram complete // Give each b3_bram element its own register so all 10 bias reads happen in one cycle

    // One-time burst load of weights/biases from DRAM into on-chip BRAM, amortized across M samples.
    memcpy(W1_bram, W1, sizeof(W1_bram)); // Burst-transfer all 100352 FC1 weight bytes from DRAM; HLS infers an AXI burst transaction
    memcpy(b1_bram, b1, sizeof(b1_bram)); // Burst-transfer all 128 FC1 bias words from DRAM
    memcpy(W2_bram, W2, sizeof(W2_bram)); // Burst-transfer all 8192 FC2 weight bytes from DRAM
    memcpy(b2_bram, b2, sizeof(b2_bram)); // Burst-transfer all 64 FC2 bias words from DRAM
    memcpy(W3_bram, W3, sizeof(W3_bram)); // Burst-transfer all 640 FC3 weight bytes from DRAM
    memcpy(b3_bram, b3, sizeof(b3_bram)); // Burst-transfer all 10 FC3 bias words from DRAM

    int8_t act1[FC1_OUT]; // Intermediate INT8 activations after FC1+ReLU, length 128
    int8_t act2[FC2_OUT]; // Intermediate INT8 activations after FC2+ReLU, length 64
#pragma HLS ARRAY_PARTITION variable=act1 complete // All 128 act1 elements get their own register for single-cycle read/write
#pragma HLS ARRAY_PARTITION variable=act2 complete // All 64 act2 elements get their own register for single-cycle read/write

    for (int m = 0; m < M; m++) { // Process each sample in the batch sequentially; weights stay in BRAM throughout
#pragma HLS LOOP_TRIPCOUNT min=1 max=128 avg=32 // Inform HLS of the expected iteration range for latency/resource reporting; does not affect logic
        gemm_tile_relu<FC1_IN, FC1_OUT, FC1_M0, REQUANT_SHIFT>(A + m * FC1_IN, W1_bram, b1_bram, act1); // FC1: 784→128, reads from DRAM at stride FC1_IN, writes INT8 activations into act1
        gemm_tile_relu<FC2_IN, FC2_OUT, FC2_M0, REQUANT_SHIFT>(act1, W2_bram, b2_bram, act2); // FC2: 128→64, reads act1, writes INT8 activations into act2
        gemm_tile_logits<FC3_IN, FC3_OUT>(act2, W3_bram, b3_bram, C + m * FC3_OUT); // FC3: 64→10, reads act2, writes 10 INT32 logits into the output buffer at stride FC3_OUT
    }
}
