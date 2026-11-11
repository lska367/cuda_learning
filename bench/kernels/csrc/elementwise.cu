// 练习 1：fused silu_and_mul
//
// 目标：out[r, c] = silu(x[r, c]) * x[r, C + c]      （C = cols）
// 这是 transformer 里最宽的中间张量（Llama-3-8B: [T, 2*14336]），所以它是纯带宽活：
//   ideal bytes = T*2C*2 (读) + T*C*2 (写)
// 验收：≥ 90% 实测峰值带宽（`make bench` 的 %BW 列）
//
// 三个关键点，值得单独记住：
// 1. 16 字节向量化：float4 = 8 个 half。用 4 字节访问会让 L1/LSU 的请求数翻 4 倍。
// 2. 2D grid (row, col_block)：避免在 kernel 里做 row = idx / C 的整数除法（那是 ~20 个周期的整数指令）。
// 3. 先读后写、读写不同地址：help the compiler 用 LDG + STG 而不是 read-modify-write。

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops.h"

namespace {

constexpr int kThreads = 256;   // 8 个 float4/线程的块，足够覆盖延迟
constexpr int kVec = 8;         // 每个 float4 = 8 个 half

// silu(g) * u，一次算 8 个 half（4 个 half2）
__device__ __forceinline__ float4 silu_mul_8(__half2 g0, __half2 g1, __half2 g2, __half2 g3,
                                             __half2 u0, __half2 u1, __half2 u2, __half2 u3) {
  float4 r;
  __half2* rh = reinterpret_cast<__half2*>(&r);
  const __half2 gs[4] = {g0, g1, g2, g3};
  const __half2 us[4] = {u0, u1, u2, u3};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 gf = __half22float2(gs[j]);
    float2 uf = __half22float2(us[j]);
    // sigmoid 用 __expf 的近似（fp16 输出精度下完全够；--use_fast_math 也会这么做）
    float a = gf.x / (1.0f + __expf(-gf.x)) * uf.x;
    float b = gf.y / (1.0f + __expf(-gf.y)) * uf.y;
    rh[j] = __floats2half2_rn(a, b);
  }
  return r;
}

__global__ void __launch_bounds__(kThreads)
    silu_and_mul_kernel(const __half* __restrict__ x, __half* __restrict__ out,
                        int n_cols, int n_vec_per_row) {
  const int row = blockIdx.x;
  const int vec_idx = blockIdx.y * kThreads + threadIdx.x;   // 第几个 float4
  if (vec_idx >= n_vec_per_row) return;

  const size_t row_base = static_cast<size_t>(row) * 2 * n_cols;
  const float4* g = reinterpret_cast<const float4*>(x + row_base);          // gate
  const float4* u = reinterpret_cast<const float4*>(x + row_base + n_cols); // up
  float4* o = reinterpret_cast<float4*>(out + static_cast<size_t>(row) * n_cols);

  const float4 gv = g[vec_idx];
  const float4 uv = u[vec_idx];
  const __half2* gh = reinterpret_cast<const __half2*>(&gv);
  const __half2* uh = reinterpret_cast<const __half2*>(&uv);
  o[vec_idx] = silu_mul_8(gh[0], gh[1], gh[2], gh[3], uh[0], uh[1], uh[2], uh[3]);
}

// 通用回退：n_cols 不能被 8 整除时用标量版
__global__ void __launch_bounds__(kThreads)
    silu_and_mul_scalar_kernel(const __half* __restrict__ x, __half* __restrict__ out,
                               int n_cols) {
  const int row = blockIdx.x;
  const size_t row_base = static_cast<size_t>(row) * 2 * n_cols;
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < n_cols; c += blockDim.x * gridDim.y) {
    float g = __half2float(x[row_base + c]);
    float u = __half2float(x[row_base + n_cols + c]);
    out[static_cast<size_t>(row) * n_cols + c] = __float2half_rn(g / (1.f + __expf(-g)) * u);
  }
}

}  // namespace

void silu_and_mul_launch(const void* x, void* out, int rows, int cols, cudaStream_t stream) {
  const __half* xh = static_cast<const __half*>(x);
  __half* oh = static_cast<__half*>(out);
  if (cols % kVec == 0) {
    const int n_vec = cols / kVec;
    dim3 grid(rows, (n_vec + kThreads - 1) / kThreads);
    silu_and_mul_kernel<<<grid, kThreads, 0, stream>>>(xh, oh, cols, n_vec);
  } else {
    dim3 grid(rows, (cols + kThreads - 1) / kThreads);
    silu_and_mul_scalar_kernel<<<grid, kThreads, 0, stream>>>(xh, oh, cols);
  }
  CUDA_CHECK_LAUNCH();
}
