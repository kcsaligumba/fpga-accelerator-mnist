#include <iostream>
#include <cstdlib>
#include <stdint.h>
#include "gemm.h"

extern void multiply(int8_t *A, int8_t *B, int *C, int M, int K, int N);

// Static arrays sized to the largest test case
#define MAX_SIZE 128

static int8_t A[MAX_SIZE * MAX_SIZE];
static int8_t B[MAX_SIZE * MAX_SIZE];
static int C[MAX_SIZE * MAX_SIZE];
static int ref[MAX_SIZE * MAX_SIZE];

static bool run_test(int M, int K, int N, const char *label) {
    // Fill with small deterministic values clamped to [-4, 4] for INT8
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)((i % 9) - 4);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)((i % 9) - 4);

    // Reference
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            ref[i * N + j] = sum;
        }
    }

    // Zero output and run HLS
    for (int i = 0; i < M * N; i++) C[i] = 0;
    multiply(A, B, C, M, K, N);

    // Compare
    bool pass = true;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != ref[i]) {
            std::cerr << "[" << label << "] Mismatch at flat index " << i
                      << ": expected " << ref[i] << ", got " << C[i] << "\n";
            pass = false;
            break;
        }
    }

    if (pass) std::cout << "Test (" << label << ") " << M << "x" << K << "x" << N << " passed.\n";
    return pass;
}

int main() {
    bool all_pass = true;

    all_pass &= run_test(32, 32, 32, "small exact");
    all_pass &= run_test(128, 128, 128, "original size");
    all_pass &= run_test(120, 84, 10, "FC layer sized");
    all_pass &= run_test(1, 120, 84, "single row");

    if (all_pass) {
        std::cout << "All tests passed!\n";
        return 0;
    }
    std::cerr << "Some tests failed.\n";
    return 1;
}