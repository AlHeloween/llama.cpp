#pragma once

#include "common.cuh"

// BitNet I2_S: fused MUL_MAT I2_S × F32 → F32 using add/subtract/skip (no dequant to F32).
// Weights are 2-bit ternary: 4 values/byte, 00=-1, 01=0, 10=+1, 11=0 (same as ggml-quants.c dequantize_row_i2_s).

void ggml_cuda_mul_mat_i2_s(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,  // I2_S [M, K] packed
    const ggml_tensor * src1,  // F32 [K, N]
    ggml_tensor * dst);       // F32 [M, N]
