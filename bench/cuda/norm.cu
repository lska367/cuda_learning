// 练习 2 / 3：RMSNorm 与 fused_add_rmsnorm
//
//   rmsnorm(x)          : out = x * rsqrt(mean(x^2) + eps) * w
//   fused_add_rmsnorm   : residual += x; out = rmsnorm(residual)
//
// 三个实现：
//   v0_block     1 CTA / 行，smem 归约，标量访存   —— 正确性锚点
//   v1_warp      1 warp / 行，warp shuffle 归约     —— 练习 2 的目标写法
//   fused        add + rmsnorm 一个 kernel          —— 练习 3 的目标写法
//
// 归约的三层结构（V100，一个 SM 有 4 个 scheduler）：
//   1) 线程内：每线程把若干个元素累加到自己的寄存器 sum
//   2) warp 内：__shfl_xor_sync 做 5 次蝶形，得到 warp 内总和（无 smem、无 barrier）
//   3) CTA 内：每个 warp 的 leader 把结果写 smem，__syncthreads 后第一个 warp 归约 W 个值

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.cuh"

namespace {

// ---------------------------------------------------------------- v0
__global__ void rms_norm_block_kernel(const half* __restrict__ x,
                                      const half* __restrict__ w,
                                      half* __restrict__ out,
                                      int n_cols, float eps, float inv_n) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const half* xr = x + (size_t)row * n_cols;
    half* orow = out + (size_t)row * n_cols;

    float sum = 0.f;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        float v = __half2float(xr[c]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rstd = rsqrtf(sdata[0] * inv_n + eps);   // 所有线程读同一个值，无需再写回
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        orow[c] = __float2half_rn(__half2float(xr[c]) * rstd * __half2float(w[c]));
    }
}

// ---------------------------------------------------------------- v1
// 一个 warp 负责一行：hidden=4096 时每个 lane 负责 128 个元素（fp16 → 8 个 16B 块）。
// 用 16B 访存 + warp shuffle，整个 kernel 只有一条归约路径，没有 __syncthreads。
__global__ void rms_norm_warp_kernel(const half* __restrict__ x,
                                     const half* __restrict__ w,
                                     half* __restrict__ out,
                                     int n_cols, float eps, float inv_n,
                                     int warp_per_block) {
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * warp_per_block) + (threadIdx.x >> 5);
    const half* xr = x + (size_t)warp * n_cols;
    half* orow = out + (size_t)warp * n_cols;

    // 线程内累加：16B 块 = 8 个 half
    float sum = 0.f;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    int nvec = n_cols / 8;
    for (int v = lane; v < nvec; v += 32) {
        float4 t = __ldg(xr4 + v);
        const half* h = reinterpret_cast<const half*>(&t);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float f = __half2float(h[k]);
            sum = fmaf(f, f, sum);
        }
    }
    sum = warp_reduce_sum(sum);
    float rstd = rsqrtf(__shfl_sync(0xffffffff, sum, 0) * inv_n + eps);

    // 写回：w 只有 n_cols 个元素，所有行复用，走 L2/常量路径
    float4* orow4 = reinterpret_cast<float4*>(orow);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (int v = lane; v < nvec; v += 32) {
        float4 xt = __ldg(xr4 + v);
        float4 wt = __ldg(w4 + v);
        half r[8];
        const half* hx = reinterpret_cast<const half*>(&xt);
        const half* hw = reinterpret_cast<const half*>(&wt);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            r[k] = __float2half_rn(__half2float(hx[k]) * rstd * __half2float(hw[k]));
        }
        orow4[v] = *reinterpret_cast<float4*>(r);
    }
}

// ---------------------------------------------------------------- fused
// 练习 3：residual += x 与 rmsnorm 合成一个 kernel，省掉一次 16KB/token 的读-改-写。
// 注意：residual 是 in-place 输出（两输入两输出），这是 vLLM fused_add_rms_norm 的语义。
__global__ void fused_add_rms_norm_kernel(const half* __restrict__ x,
                                          half* __restrict__ residual,
                                          const half* __restrict__ w,
                                          half* __restrict__ out,
                                          int n_cols, float eps, float inv_n,
                                          int warp_per_block) {
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * warp_per_block) + (threadIdx.x >> 5);
    const half* xr = x + (size_t)warp * n_cols;
    half* rr = residual + (size_t)warp * n_cols;
    half* orow = out + (size_t)warp * n_cols;

    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    float4* rr4 = reinterpret_cast<float4*>(rr);
    float4* orow4 = reinterpret_cast<float4*>(orow);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    int nvec = n_cols / 8;

    // 只读一遍 x 和 residual：sum 与写回共用同一份寄存器数据做不到（跨 lane），
    // 所以这里读两次（第二次命中 L2），比"写 residual 再被下个 kernel 读"便宜得多。
    float sum = 0.f;
    for (int v = lane; v < nvec; v += 32) {
        float4 a = __ldg(xr4 + v);
        float4 b = __ldg(reinterpret_cast<const float4*>(rr) + v);
        half s[8];
        const half* ha = reinterpret_cast<const half*>(&a);
        const half* hb = reinterpret_cast<const half*>(&b);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float f = __half2float(ha[k]) + __half2float(hb[k]);
            s[k] = __float2half_rn(f);
            sum = fmaf(f, f, sum);
        }
        rr4[v] = *reinterpret_cast<float4*>(s);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    float rstd = rsqrtf(__shfl_sync(0xffffffff, sum, 0) * inv_n + eps);

    for (int v = lane; v < nvec; v += 32) {
        float4 b = rr4[v];        // 不能用 __ldg：这块地址刚被本 kernel 写过，只读缓存里是旧值
        float4 wt = __ldg(w4 + v);
        half r[8];
        const half* hb = reinterpret_cast<const half*>(&b);
        const half* hw = reinterpret_cast<const half*>(&wt);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            r[k] = __float2half_rn(__half2float(hb[k]) * rstd * __half2float(hw[k]));
        }
        orow4[v] = *reinterpret_cast<float4*>(r);
    }
}

}  // namespace

void rms_norm_v0(torch::Tensor x, torch::Tensor w, torch::Tensor out, double eps) {
    int64_t rows = x.size(0), cols = x.size(1);
    int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_block_kernel<<<(int)rows, threads, threads * sizeof(float), stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        (int)cols, (float)eps, 1.f / (float)cols);
    CUDA_CHECK_KERNEL();
}

void rms_norm_v1(torch::Tensor x, torch::Tensor w, torch::Tensor out, double eps) {
    TORCH_CHECK(x.size(1) % 128 == 0, "hidden must be a multiple of 128");
    int64_t rows = x.size(0), cols = x.size(1);
    int warps_per_block = 4;                 // 128 threads
    int warps_total = (int)rows;
    int blocks = (warps_total + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_warp_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        (int)cols, (float)eps, 1.f / (float)cols, warps_per_block);
    CUDA_CHECK_KERNEL();
}

void fused_add_rms_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor w,
                        torch::Tensor out, double eps) {
    int64_t rows = x.size(0), cols = x.size(1);
    int warps_per_block = 4;
    int blocks = ((int)rows + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_add_rms_norm_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        (int)cols, (float)eps, 1.f / (float)cols, warps_per_block);
    CUDA_CHECK_KERNEL();
}
