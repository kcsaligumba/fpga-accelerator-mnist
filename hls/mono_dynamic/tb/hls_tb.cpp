#include <iostream>
#include <cstring>
#include <stdint.h>
#include "hls.h"

extern void mlp(
    int8_t  *A,
    int8_t  *W1, int32_t *b1,
    int8_t  *W2, int32_t *b2,
    int8_t  *W3, int32_t *b3,
    int32_t *C
);

static int8_t  A [FC1_IN];
static int8_t  W1[FC1_IN  * FC1_OUT];
static int8_t  W2[FC2_IN  * FC2_OUT];
static int8_t  W3[FC3_IN  * FC3_OUT];
static int32_t b1[FC1_OUT];
static int32_t b2[FC2_OUT];
static int32_t b3[FC3_OUT];
static int32_t C_hls[FC3_OUT];
static int32_t C_ref[FC3_OUT];

static void ref_mlp() {
    int8_t act1[FC1_OUT];
    for (int n = 0; n < FC1_OUT; n++) {
        int32_t s = b1[n];
        for (int k = 0; k < FC1_IN; k++)
            s += (int32_t)A[k] * (int32_t)W1[k * FC1_OUT + n];
        act1[n] = (int8_t)(s > 127 ? 127 : (s < 0 ? 0 : s));
    }

    int8_t act2[FC2_OUT];
    for (int n = 0; n < FC2_OUT; n++) {
        int32_t s = b2[n];
        for (int k = 0; k < FC2_IN; k++)
            s += (int32_t)act1[k] * (int32_t)W2[k * FC2_OUT + n];
        act2[n] = (int8_t)(s > 127 ? 127 : (s < 0 ? 0 : s));
    }

    for (int n = 0; n < FC3_OUT; n++) {
        int32_t s = b3[n];
        for (int k = 0; k < FC3_IN; k++)
            s += (int32_t)act2[k] * (int32_t)W3[k * FC3_OUT + n];
        C_ref[n] = s;
    }
}

static bool run_test(int seed, const char *label) {
    for (int i = 0; i < FC1_IN;           i++) A [i] = (int8_t)(((i+seed) % 9) - 4);
    for (int i = 0; i < FC1_IN * FC1_OUT; i++) W1[i] = (int8_t)(((i+seed) % 7) - 3);
    for (int i = 0; i < FC2_IN * FC2_OUT; i++) W2[i] = (int8_t)(((i+seed) % 5) - 2);
    for (int i = 0; i < FC3_IN * FC3_OUT; i++) W3[i] = (int8_t)(((i+seed) % 9) - 4);
    for (int i = 0; i < FC1_OUT; i++) b1[i] = (i+seed) % 16 - 8;
    for (int i = 0; i < FC2_OUT; i++) b2[i] = (i+seed) % 16 - 8;
    for (int i = 0; i < FC3_OUT; i++) b3[i] = (i+seed) % 16 - 8;

    ref_mlp();

    memset(C_hls, 0, sizeof(C_hls));
    mlp(A, W1, b1, W2, b2, W3, b3, C_hls);

    bool pass = true;
    for (int n = 0; n < FC3_OUT; n++) {
        if (C_hls[n] != C_ref[n]) {
            std::cerr << "[" << label << "] MISMATCH logit[" << n << "]"
                      << "  ref=" << C_ref[n] << "  got=" << C_hls[n] << "\n";
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "PASS" : "FAIL") << "  [" << label << "]\n";
    return pass;
}

int main() {
    bool ok = true;

    // Basic correctness across different data patterns
    ok &= run_test(0, "seed=0 baseline");
    ok &= run_test(1, "seed=1 shifted");
    ok &= run_test(4, "seed=4 positive-heavy");
    ok &= run_test(8, "seed=8 negative-heavy");

    // Edge case: all-zero input → all outputs should equal bias
    memset(A, 0, sizeof(A));
    memset(W1, 0, sizeof(W1));
    memset(W2, 0, sizeof(W2));
    memset(W3, 0, sizeof(W3));
    for (int i = 0; i < FC1_OUT; i++) b1[i] = i;
    for (int i = 0; i < FC2_OUT; i++) b2[i] = i;
    for (int i = 0; i < FC3_OUT; i++) b3[i] = i;
    ref_mlp();
    memset(C_hls, 0, sizeof(C_hls));
    mlp(A, W1, b1, W2, b2, W3, b3, C_hls);
    {
        bool pass = true;
        for (int n = 0; n < FC3_OUT; n++) {
            if (C_hls[n] != C_ref[n]) {
                std::cerr << "[zero-weights] MISMATCH logit[" << n << "]\n";
                pass = false; break;
            }
        }
        std::cout << (pass ? "PASS" : "FAIL") << "  [zero-weights bias-only]\n";
        ok &= pass;
    }

    std::cout << "\n" << (ok ? "All tests passed." : "SOME TESTS FAILED.") << "\n";
    return ok ? 0 : 1;
}