// 练习 1：fused silu_and_mul（SwiGLU）
//
//   out[t, i] = silu(x[t, i]) * x[t, I + i]      I = n_cols
//
// 三个实现，按顺序看：
//   v0_scalar  每线程 1 个元素       —— 正确性锚点，性能约 15~25% 带宽
//   v1_vec     每线程 8 个元素       —— 128-bit 访存 + float 计算，目标 ≥85% 带宽
//   v2_*       TODO：给你自己写      —— 见文件末尾的提示
//
// 编译：bench/cuda/build.py（JIT，改完 .cu 后下次 import 自动重编）

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.cuh"

namespace {

// ---------------------------------------------------------------- v0
// 一个线程算一个输出元素。读两半（间隔 I，两条独立的流），写一个。
__global__ void silu_mul_scalar_kernel(const half* __restrict__ x,
                                       half* __restrict__ out,
                                       int n_cols, int n_total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    int row = idx / n_cols;
    int col = idx - row * n_cols;
    const half* base = x + (size_t)row * 2 * n_cols;
    float g = __half2float(base[col]);
    float u = __half2float(base[n_cols + col]);
    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float r = g / (1.0f + __expf(-g)) * u;
    out[idx] = __float2half_rn(r);
}

// ---------------------------------------------------------------- v1
// 每线程 8 个元素：gate/up 各一次 16-byte 读取、out 一次 16-byte 写入。
// 关键三点：
//   1) 用 float4（=16B = 8 个 half）做访存单元，编译器生成 LDG.128/STG.128
//      128 个线程 * 16B = 2KB/CTA 的连续事务，正好塞满一个 memory sector 组；
//   2) 计算必须在 fp32 里做（half 上 exp 精度和范围都不够）；
//   3) 用 grid-stride 循环，使 grid 大小与张量大小解耦（尾块不需要单独 kernel）。
__global__ void silu_mul_vec_kernel(const float4* __restrict__ x,
                                    float4* __restrict__ out,
                                    int n_cols, int n_total_vec8) {
    constexpr int VEC = 8;                       // 8 个 half = 16B
    int cols_vec = n_cols / VEC;                 // 每行的 16B 块数
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_total_vec8; i += stride) {
        int row = i / cols_vec;
        int cv = i - row * cols_vec;
        const float4* base = x + (size_t)row * 2 * cols_vec;
        float4 g4 = __ldg(base + cv);
        float4 u4 = __ldg(base + cols_vec + cv);
        const half* g = reinterpret_cast<const half*>(&g4);
        const half* u = reinterpret_cast<const half*>(&u4);
        half r[VEC];
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float gv = __half2float(g[k]);
            float uv = __half2float(u[k]);
            r[k] = __float2half_rn(gv / (1.0f + __expf(-gv)) * uv);
        }
        out[i] = *reinterpret_cast<float4*>(r);
    }
}

// ---------------------------------------------------------------- v2 (TODO)
// 想再往上推，按收益顺序试这几件事（每步都用 bench 复测）：
//   a) 换成 half2 计算：__h2exp / __hfma2，一条指令算 2 个元素（v1 每线程 8 个
//      float 运算 = 8 条 exp.serial，很容易从带宽受限变成算力受限 —— 先用
//      ncu/nsys 确认瓶颈）；
//   b) 每线程处理 16 或 32 个元素（2~4 次 128-bit load），提高 ILP；
//   c) blockDim 取 128 或 256，grid 取 SM 数的整数倍（80 的倍数），
//      避免最后一个 wave 不满；
//   d) 打开 --use_fast_math（__expf 已经够快，意义有限）。
// 提示：如果 %BW 卡在 60% 上不去，先确认 row 的除法是否成了每线程一条昂贵的
// 整数除法（用 constexpr 幂次或预计算 col_block），再看 sectors/request 指标。

}  // namespace

void silu_mul_v0(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    TORCH_CHECK(x.dtype() == torch::kHalf && out.dtype() == torch::kHalf);
    int64_t rows = x.size(0);
    int64_t cols = x.size(1) / 2;
    TORCH_CHECK(out.numel() == rows * cols, "out size mismatch");
    int n_total = (int)(rows * cols);
    int threads = 256;
    int blocks = (n_total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    silu_mul_scalar_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        (int)cols, n_total);
    CUDA_CHECK_KERNEL();
}

void silu_mul_v1(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
    int64_t rows = x.size(0);
    int64_t cols = x.size(1) / 2;
    TORCH_CHECK(cols % 8 == 0, "I must be a multiple of 8 for the vector path");
    int64_t n_vec = rows * cols / 8;
    int threads = 256;
    int blocks = (int)std::min<int64_t>((n_vec + threads - 1) / threads, 80 * 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    silu_mul_vec_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float4*>(x.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(out.data_ptr<at::Half>()),
        (int)cols, (int)n_vec);
    CUDA_CHECK_KERNEL();
}
