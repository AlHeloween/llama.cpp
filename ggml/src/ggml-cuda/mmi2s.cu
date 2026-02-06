#include "mmi2s.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include "common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cinttypes>
#include <cstdio>

// BitNet I2_S CUDA:
// Reference CPU path:
// 1) quantize activations x -> int8 v with scale s = 127/maxabs(x) and sum(v)
// 2) dot_q = sum(q * v) where q is raw 2-bit value in {0,1,2} (not q-1)
// 3) out = (dot_q - sum(v)) / s * weight_scale
// Weights are packed per 128 weights into 32 bytes:
//   group_idx = j/32 (0..3), group_pos = j%32 (0..31)
//   byte = w[block*32 + group_pos]
//   q    = (byte >> (6 - 2*group_idx)) & 3
static constexpr int QK_I2_S = 128;

static __device__ __forceinline__ float load_src1_f32(const char * base, int64_t i, size_t nb10, int src1_type) {
    const char * p = base + i * (int64_t) nb10;
    if (src1_type == GGML_TYPE_F32) {
        return *(const float *) p;
    }
    if (src1_type == GGML_TYPE_F16) {
        return __half2float(*(const half *) p);
    }
    return 0.0f;
}

static __device__ __forceinline__ int32_t block_reduce_sum_i32(int32_t v) {
    __shared__ int32_t warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        warp_sums[wid] = v;
    }
    __syncthreads();

    const int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? warp_sums[lane] : 0;
    if (wid == 0) {
        v = warp_reduce_sum(v);
    }
    return v;
}

__global__ void quantize_cols_i8_s_kernel(
    const void  * __restrict__ src1,   // [K, N] (dim0 contiguous), src1_nb11 bytes between columns
    int8_t       * __restrict__ xq,     // [K, N] (column-major: xq[n*K + k])
    float        * __restrict__ scales, // [N]
    int32_t      * __restrict__ sums,   // [N]
    const int     src1_type,           // GGML_TYPE_F32 or GGML_TYPE_F16
    const int64_t K,
    const int64_t N,
    const size_t  src1_nb10,
    const size_t  src1_nb11) {

    const int64_t n = (int64_t) blockIdx.x;
    if (n >= N) return;

    const char * src1_col_bytes = (const char *) src1 + n * src1_nb11;
    int8_t * __restrict__ y      = xq + n * K;

    // 1) maxabs
    float tmax = 1e-5f;
    for (int64_t i = (int64_t) threadIdx.x; i < K; i += (int64_t) blockDim.x) {
        const float a = fabsf(load_src1_f32(src1_col_bytes, i, src1_nb10, src1_type));
        if (a > tmax) tmax = a;
    }

    // reduce max across block
    __shared__ float sh_max[32];
    __shared__ float sh_s;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;

    float vmax = warp_reduce_max(tmax);
    if (lane == 0) {
        sh_max[wid] = vmax;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        float v = (threadIdx.x < nwarps) ? sh_max[threadIdx.x] : 0.0f;
        v = warp_reduce_max(v);
        if (threadIdx.x == 0) {
            sh_max[0] = v;
            sh_s = 127.0f / v;
            scales[n] = sh_s;
        }
    }
    __syncthreads();

    const float s = sh_s;

    // 2) quantize + sum
    int32_t tsum = 0;
    for (int64_t i = (int64_t) threadIdx.x; i < K; i += (int64_t) blockDim.x) {
        int v = __float2int_rn(load_src1_f32(src1_col_bytes, i, src1_nb10, src1_type) * s);
        v = v > 127 ? 127 : v;
        v = v < -128 ? -128 : v;
        y[i] = (int8_t) v;
        tsum += v;
    }

    tsum = block_reduce_sum_i32(tsum);
    if (threadIdx.x == 0) {
        sums[n] = tsum;
    }
}

__global__ void mul_mat_i2_s_packed_kernel(
    const uint8_t * __restrict__ src0,      // packed weights [M, K/4] bytes
    const int8_t  * __restrict__ xq,        // quantized activations [K, N] (xq[n*K + k])
    const float   * __restrict__ act_scales,// [N]
    const int32_t * __restrict__ act_sums,  // [N]
    const float   * __restrict__ w_scale,   // single float
    float         * __restrict__ dst,       // dst [M, N] (column-major: dst_col[n][m])
    const int64_t M,
    const int64_t K,
    const int64_t N,
    const size_t  src0_row_bytes,
    const size_t  dst_nb1) {

    const int64_t n = (int64_t) blockIdx.x;
    const int64_t m = (int64_t) blockIdx.y;
    if (n >= N || m >= M) return;

    const int tid = (int) threadIdx.x; // [0..127]

    const int32_t act_sum = act_sums[n];
    const float   act_s   = act_scales[n];
    const float   ws      = w_scale[0];

    const uint8_t * __restrict__ w_row = src0 + m * src0_row_bytes;
    const int8_t  * __restrict__ v_col = xq  + n * K;

    int32_t acc = 0;
    const int64_t nblk = K / QK_I2_S;

    for (int64_t b = 0; b < nblk; ++b) {
        const uint8_t * __restrict__ w_blk = w_row + b * 32; // 128 weights -> 32 bytes
        const int64_t k = b * QK_I2_S + tid;
        const int group_idx = tid >> 5;  // 0..3
        const int group_pos = tid & 31;  // 0..31
        const uint8_t byte = w_blk[group_pos];
        const int q = (byte >> (6 - 2 * group_idx)) & 3;
        acc += q * (int32_t) v_col[k];
    }

    acc = block_reduce_sum_i32(acc);
    if (tid == 0) {
        float * dst_col = (float *) ((char *) dst + n * dst_nb1);
        dst_col[m] = ((float) (acc - act_sum) / act_s) * ws;
    }
}

void ggml_cuda_mul_mat_i2_s(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst) {

    GGML_TENSOR_BINARY_OP_LOCALS;

    static bool s_logged_once = false;
    if (!s_logged_once) {
        s_logged_once = true;
        fprintf(stderr,
            "%s: I2_S CUDA: ne00=%" PRId64 " ne01=%" PRId64 " ne11=%" PRId64
            " nb00=%zu nb01=%zu nb02=%zu nb03=%zu nb10=%zu nb11=%zu nb12=%zu nb13=%zu nb0=%zu nb1=%zu\n",
            __func__,
            ne00, ne01, ne11,
            (size_t) nb00, (size_t) nb01, (size_t) nb02, (size_t) nb03,
            (size_t) nb10, (size_t) nb11, (size_t) nb12, (size_t) nb13,
            (size_t) nb0, (size_t) nb1);
        fflush(stderr);
    }

    GGML_ASSERT(src0->type == GGML_TYPE_I2_S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ne00 % QK_I2_S == 0);
    const int64_t M = ne01;
    const int64_t K = ne00;
    const int64_t N = ne11;
    const size_t src0_row_bytes = nb01;
    const size_t src1_nb11      = nb11;
    const size_t dst_nb1        = nb1;

    const uint8_t * src0_d = (const uint8_t *) src0->data;
    const void    * src1_d = (const void *)    src1->data;
    float         * dst_d  = (float *)        dst->data;

    // Batched over ne02, ne03 (batch dimensions)
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const uint8_t * src0_ptr = (const uint8_t *) ((const char *) src0_d + i02*nb02 + i03*nb03);
            const void    * src1_ptr = (const void *)    ((const char *) src1_d + i02*nb12 + i03*nb13);
            float         * dst_ptr  = (float *)        ((char *)       dst_d  + i02*nb2  + i03*nb3);

            // Packed weight scale stored after packed weights (K*M/4 bytes).
            const float * w_scale = (const float *) ((const char *) src0_ptr + (K * M) / 4);

            // Quantize src1 columns -> int8 + per-column scale/sum.
            ggml_cuda_pool_alloc<int8_t>  xq(ctx.pool(), (size_t) (K * N));
            ggml_cuda_pool_alloc<float>   act_scales(ctx.pool(), (size_t) N);
            ggml_cuda_pool_alloc<int32_t> act_sums(ctx.pool(), (size_t) N);

            {
                const int threads = 256;
                dim3 block(threads, 1, 1);
                dim3 grid((unsigned) N, 1, 1);
                quantize_cols_i8_s_kernel<<<grid, block, 0, ctx.stream()>>>(
                    src1_ptr,
                    xq.get(),
                    act_scales.get(),
                    act_sums.get(),
                    (int) src1->type,
                    K, N,
                    (size_t) nb10,
                    src1_nb11);
            }

            // dst = (packed I2_S) x (quantized src1) with BitNet correction and weight scale.
            {
                dim3 block(QK_I2_S, 1, 1);
                dim3 grid((unsigned) N, (unsigned) M, 1);
                mul_mat_i2_s_packed_kernel<<<grid, block, 0, ctx.stream()>>>(
                    src0_ptr,
                    xq.get(),
                    act_scales.get(),
                    act_sums.get(),
                    w_scale,
                    dst_ptr,
                    M, K, N,
                    src0_row_bytes,
                    dst_nb1);
            }
        }
    }
}
