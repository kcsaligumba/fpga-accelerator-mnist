#include <iostream>
#include <cstring>
#include <stdint.h>
#include "hardcode.h"
#include "hardcode_params.h"

extern void mlp(int8_t *A, int32_t *C);


static void ref_mlp(int8_t *A, int32_t *C_ref) {
    // FC1
    int8_t act1[FC1_OUT];
    for (int n = 0; n < FC1_OUT; n++) {
        int32_t s = b1[n];
        for (int k = 0; k < FC1_IN; k++)
            s += (int32_t)A[k] * (int32_t)W1[k * FC1_OUT + n];
        act1[n] = (int8_t)(s > 127 ? 127 : (s < 0 ? 0 : s));
    }

    // FC2
    int8_t act2[FC2_OUT];
    for (int n = 0; n < FC2_OUT; n++) {
        int32_t s = b2[n];
        for (int k = 0; k < FC2_IN; k++)
            s += (int32_t)act1[k] * (int32_t)W2[k * FC2_OUT + n];
        act2[n] = (int8_t)(s > 127 ? 127 : (s < 0 ? 0 : s));
    }

    // FC3 — no ReLU
    for (int n = 0; n < FC3_OUT; n++) {
        int32_t s = b3[n];
        for (int k = 0; k < FC3_IN; k++)
            s += (int32_t)act2[k] * (int32_t)W3[k * FC3_OUT + n];
        C_ref[n] = s;
    }
}


static bool run_test(int8_t *A, const char *label) {
    int32_t C_hls[FC3_OUT] = {};
    int32_t C_ref[FC3_OUT] = {};

    ref_mlp(A, C_ref);
    mlp(A, C_hls);

    bool pass = true;
    for (int n = 0; n < FC3_OUT; n++) {
        if (C_hls[n] != C_ref[n]) {
            std::cerr << "[" << label << "] MISMATCH logit[" << n << "]"
                      << "  ref=" << C_ref[n] << "  got=" << C_hls[n] << "\n";
            pass = false;
            break;
        }
    }

    // Also print predicted class for convenience
    int pred = 0;
    for (int n = 1; n < FC3_OUT; n++)
        if (C_ref[n] > C_ref[pred]) pred = n;

    std::cout << (pass ? "PASS" : "FAIL")
              << "  [" << label << "]  predicted class=" << pred
              << "  logits:";
    for (int n = 0; n < FC3_OUT; n++)
        std::cout << " " << C_ref[n];
    std::cout << "\n";

    return pass;
}

int main() {
    bool ok = true;

    // Test 1: all-zero input
    {
        int8_t A[FC1_IN] = {};
        ok &= run_test(A, "all-zero input");
    }

    // Test 2: all-one input
    {
        int8_t A[FC1_IN];
        for (int i = 0; i < FC1_IN; i++) A[i] = 1;
        ok &= run_test(A, "all-one input");
    }

    // Test 3: alternating +1/-1
    {
        int8_t A[FC1_IN];
        for (int i = 0; i < FC1_IN; i++) A[i] = (i % 2 == 0) ? 1 : -1;
        ok &= run_test(A, "alternating +1/-1");
    }

    // Test 4: ramp pattern (representative of a quantised MNIST pixel row)
    {
        int8_t A[FC1_IN];
        for (int i = 0; i < FC1_IN; i++) A[i] = (int8_t)(((i % 9) - 4));
        ok &= run_test(A, "ramp pattern");
    }

    // Test 5: max positive activations
    {
        int8_t A[FC1_IN];
        for (int i = 0; i < FC1_IN; i++) A[i] = 127;
        ok &= run_test(A, "max positive (127)");
    }

    // Test 6: max negative activations
    {
        int8_t A[FC1_IN];
        for (int i = 0; i < FC1_IN; i++) A[i] = -128;
        ok &= run_test(A, "max negative (-128)");
    }

    std::cout << "\n" << (ok ? "All tests passed." : "SOME TESTS FAILED.") << "\n";
    return ok ? 0 : 1;
}