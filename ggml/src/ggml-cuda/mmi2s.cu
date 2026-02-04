#include "mmi2s.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include "common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// BitNet I2_S: 2-bit ternary, 4 values/byte. 00=-1, 01=0, 10=+1, 11=0 (matches ggml-quants.c dequantize_row_i2_s).
static __device__ __forceinline__ float i2_s_byte_to_val(uint8_t b, int shift) {
    const int idx = (b >> shift) & 3;
    const float tbl[4] = { -1.f, 0.f, 1.f, 0.f };
    return tbl[idx];
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
    // Full mask for non-divergent warps (Pascal+).
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

static __device__ __forceinline__ float block_reduce_sum(float v) {
    // Block-wide reduction for 1D blocks (threadIdx.x).
    // Up to 1024 threads → up to 32 warps.
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        warp_sums[wid] = v;
    }
    __syncthreads();

    const int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? warp_sums[lane] : 0.f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
    }
    return v;
}

// Kernel: dst[m, n] = sum_k src0[m,k] * src1[k,n] with src0 I2_S (ternary), src1 F32.
// Each thread computes one output (m, n). Loops over k in steps of 4 (one byte of weights).
__global__ void mul_mat_i2_s_kernel(
    const uint8_t * __restrict__ src0,  // [M, K/4] bytes, row stride src0_row_bytes
    const float   * __restrict__ src1,  // [K, N], row stride src1_row_bytes (bytes)
    float         * __restrict__ dst,   // [M, N], row stride dst_row_bytes (bytes)
    const int64_t M,
    const int64_t K,
    const int64_t N,
    const size_t  src0_row_bytes,
    const size_t  src1_row_bytes,
    const size_t  dst_row_bytes) {

    const int64_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.f;
    const int64_t K4 = K / 4;
    for (int64_t k4 = 0; k4 < K4; k4++) {
        const uint8_t b = src0[m * src0_row_bytes + k4];
        float v0 = i2_s_byte_to_val(b, 0);
        float v1 = i2_s_byte_to_val(b, 2);
        float v2 = i2_s_byte_to_val(b, 4);
        float v3 = i2_s_byte_to_val(b, 6);
        const int64_t k = k4 * 4;
        const float * row0 = (const float *) ((const char *) src1 + (k + 0) * src1_row_bytes);
        const float * row1 = (const float *) ((const char *) src1 + (k + 1) * src1_row_bytes);
        const float * row2 = (const float *) ((const char *) src1 + (k + 2) * src1_row_bytes);
        const float * row3 = (const float *) ((const char *) src1 + (k + 3) * src1_row_bytes);
        acc += v0 * row0[n] + v1 * row1[n] + v2 * row2[n] + v3 * row3[n];
    }
    float * dst_row = (float *) ((char *) dst + m * dst_row_bytes);
    dst_row[n] = acc;
}

// Specialized mat-vec for N=1 (generation): avoid wasting 31/32 threads on the N dimension.
// One block computes one output row (one 'm'). Threads iterate over K/4 bytes and reduce.
__global__ void mul_mat_i2_s_vec_kernel(
    const uint8_t * __restrict__ src0,  // [M, K/4] bytes, row stride src0_row_bytes
    const float   * __restrict__ src1,  // [K, 1], row stride src1_row_bytes (bytes)
    float         * __restrict__ dst,   // [M, 1], row stride dst_row_bytes (bytes)
    const int64_t M,
    const int64_t K,
    const size_t  src0_row_bytes,
    const size_t  src1_row_bytes,
    const size_t  dst_row_bytes) {

    const int64_t m = (int64_t) blockIdx.x;
    if (m >= M) return;

    float acc = 0.f;
    const int64_t K4 = K / 4;
    for (int64_t k4 = (int64_t) threadIdx.x; k4 < K4; k4 += (int64_t) blockDim.x) {
        const uint8_t b = src0[m * src0_row_bytes + k4];
        const int64_t k = k4 * 4;

        const float x0 = *(const float *) ((const char *) src1 + (k + 0) * src1_row_bytes);
        const float x1 = *(const float *) ((const char *) src1 + (k + 1) * src1_row_bytes);
        const float x2 = *(const float *) ((const char *) src1 + (k + 2) * src1_row_bytes);
        const float x3 = *(const float *) ((const char *) src1 + (k + 3) * src1_row_bytes);

        // Decode 2-bit ternary: 00=-1, 01=0, 10=+1, 11=0
        const int q0 = (b >> 0) & 3;
        const int q1 = (b >> 2) & 3;
        const int q2 = (b >> 4) & 3;
        const int q3 = (b >> 6) & 3;

        if (q0 == 0) acc -= x0; else if (q0 == 2) acc += x0;
        if (q1 == 0) acc -= x1; else if (q1 == 2) acc += x1;
        if (q2 == 0) acc -= x2; else if (q2 == 2) acc += x2;
        if (q3 == 0) acc -= x3; else if (q3 == 2) acc += x3;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        float * dst_row = (float *) ((char *) dst + m * dst_row_bytes);
        dst_row[0] = acc;
    }
}

void ggml_cuda_mul_mat_i2_s(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst) {

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I2_S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ne00 % 4 == 0);
    const int64_t M = ne01;
    const int64_t K = ne00;
    const int64_t N = ne11;
    const size_t src0_row_bytes = nb01;
    const size_t src1_row_bytes = nb11;
    const size_t dst_row_bytes  = nb1;

    const uint8_t * src0_d = (const uint8_t *) src0->data;
    const float   * src1_d = (const float *)   src1->data;
    float         * dst_d  = (float *)        dst->data;

    // Batched over ne02, ne03 (batch dimensions)
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const uint8_t * src0_ptr = (const uint8_t *) ((const char *) src0_d + i02*nb02 + i03*nb03);
            const float   * src1_ptr = (const float *)   ((const char *) src1_d + i02*nb12 + i03*nb13);
            float         * dst_ptr  = (float *)        ((char *)       dst_d  + i02*nb2  + i03*nb3);

            if (N == 1) {
                // Generation path: one block per row for better occupancy on small N.
                const int threads = 256;
                dim3 block(threads, 1, 1);
                dim3 grid((unsigned) M, 1, 1);
                mul_mat_i2_s_vec_kernel<<<grid, block, 0, ctx.stream()>>>(
                    src0_ptr,
                    src1_ptr,
                    dst_ptr,
                    M, K,
                    src0_row_bytes,
                    src1_row_bytes,
                    dst_row_bytes);
            } else {
                dim3 block(32, 16);
                dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
                mul_mat_i2_s_kernel<<<grid, block, 0, ctx.stream()>>>(
                    src0_ptr,
                    src1_ptr,
                    dst_ptr,
                    M, K, N,
                    src0_row_bytes,
                    src1_row_bytes,
                    dst_row_bytes);
            }
        }
    }
}
