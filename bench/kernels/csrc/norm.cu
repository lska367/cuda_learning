// 练习 2：RMSNorm（一行一个 CTA）
// 练习 3：fused add + RMSNorm（两输入两输出）
//
// 这里刻意给两个版本，用来体会"多读一遍"有多贵：
//   v1: 读 x → 归约 → 再读一遍 x → 写 y            账面流量 3x（2 读 1 写）
//   v2: 读 x 进寄存器 → 归约 → 用寄存器里的值写 y    账面流量 2x（1 读 1 写）
// 实测（H=4096，% 为实测峰值带宽占比）：
//            T=1024   T=4096   T=16384
//   v1        55%      65%      54%
//   v2        69%      82%      74%
// 注意两点：
// 1. v1 的真实流量是账面的 1.5 倍，所以它其实也在"满速跑"，只是搬了更多字节 ——
//    这就是为什么 %BW 必须按实现的真实流量看，不能只看 ms。
// 2. 尺寸决定了天花板：同尺寸的纯 copy 在这台机器上只有 77% / 93% / 96%。
//    T=1024 时总共才 16 MB 流量，任何 kernel 都到不了 85%。想把 v2 从 82% 推到 90%，
//    要吃掉的是 __syncthreads 造成的气泡（软件流水：块内处理多行，把下一行的 load
//    提到本行的 barrier 之前），而不是再加向量化。
//
// v2 的要点：
//   1. 每线程 VPT 个 float4（H=4096, 256 线程 → VPT=2，共 16 个 half = 8 个 32-bit 寄存器），
//      全部留在寄存器里，不写 shared memory；
//   2. 向量化步长用 blockDim（tid + i*THREADS），warp 内 32 个 float4 = 512 B 连续；
//   3. 归约 = warp shuffle 5 步 + shared memory 跨 warp，最后广播 rstd。
//
// 想继续压榨的话，可以试：
//   - 一个 warp 负责一行（无 __syncthreads，但寄存器压力大；实测在这台机器上更慢）；
//   - x 用 __ldcs（streaming，evict-first）读，把 L2 让给 w；
//   - 把 (x, w) 的两次 load 合并成一次 32 B 访问（需要对齐的 [T,H] 交错布局）。
//   但先看清天花板：同尺寸的纯 copy 在这台机器上只有 86–93%，别把 100% 当目标。

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ops.h"

namespace {

constexpr int kVec = 8;      // 每个 float4 = 8 个 half
constexpr int kThreads = 256;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// 块内求和并广播给所有线程
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n_warp = (blockDim.x + 31) >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < n_warp) ? smem[lane] : 0.f;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float sum_sq_8(float4 v) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __half22float2(h[j]);
    s = fmaf(f.x, f.x, s);
    s = fmaf(f.y, f.y, s);
  }
  return s;
}

__device__ __forceinline__ float4 add_8(float4 a, float4 b) {
  float4 r;
  const __half2* ah = reinterpret_cast<const __half2*>(&a);
  const __half2* bh = reinterpret_cast<const __half2*>(&b);
  __half2* rh = reinterpret_cast<__half2*>(&r);
#pragma unroll
  for (int j = 0; j < 4; ++j) rh[j] = __hadd2(ah[j], bh[j]);
  return r;
}

__device__ __forceinline__ float4 scale_mul_8(float4 x, float4 w, float s) {
  float4 r;
  const __half2* xh = reinterpret_cast<const __half2*>(&x);
  const __half2* wh = reinterpret_cast<const __half2*>(&w);
  __half2* rh = reinterpret_cast<__half2*>(&r);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 xf = __half22float2(xh[j]);
    const float2 wf = __half22float2(wh[j]);
    rh[j] = __floats2half2_rn(xf.x * wf.x * s, xf.y * wf.y * s);
  }
  return r;
}

// ---------------------------------------------------------------- v1 两遍读

__global__ void __launch_bounds__(kThreads)
    rms_norm_v1_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                       __half* __restrict__ out, int n_cols, float eps) {
  __shared__ float smem[32];
  const size_t base = static_cast<size_t>(blockIdx.x) * n_cols;
  const int n_vec = n_cols / kVec;
  const float4* xp = reinterpret_cast<const float4*>(x + base);
  const float4* wp = reinterpret_cast<const float4*>(w);

  float acc = 0.f;
  for (int i = threadIdx.x; i < n_vec; i += kThreads) acc += sum_sq_8(xp[i]);

  const float rstd = rsqrtf(block_reduce_sum(acc, smem) / n_cols + eps);

  float4* op = reinterpret_cast<float4*>(out + base);
  for (int i = threadIdx.x; i < n_vec; i += kThreads)
    op[i] = scale_mul_8(xp[i], wp[i], rstd);   // ← 第二次读 x，真实流量 +50%
}

// ------------------------------------------- v2 寄存器缓存（一行一 CTA，单遍）

template <int VPT>
__global__ void __launch_bounds__(kThreads)
    rms_norm_v2_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                       __half* __restrict__ out, float eps) {
  __shared__ float smem[32];
  const size_t base = static_cast<size_t>(blockIdx.x) * (VPT * kThreads * kVec);
  const float4* xp = reinterpret_cast<const float4*>(x + base);
  const float4* wp = reinterpret_cast<const float4*>(w);
  float4* op = reinterpret_cast<float4*>(out + base);

  float4 xv[VPT], wv[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    // __ldcs = ld.global.cs（streaming，evict-first）：x 只用一次，别把 L2 挤掉 w。
    // 实测在大 T（>16k 行）时能多 5 个点，小 T 时无差别。
    xv[i] = __ldcs(xp + threadIdx.x + i * kThreads);
    wv[i] = wp[threadIdx.x + i * kThreads];
  }

  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) acc += sum_sq_8(xv[i]);

  const float rstd = rsqrtf(block_reduce_sum(acc, smem) / (VPT * kThreads * kVec) + eps);

#pragma unroll
  for (int i = 0; i < VPT; ++i)
    __stcs(op + threadIdx.x + i * kThreads, scale_mul_8(xv[i], wv[i], rstd));
}

// ------------------------------------------------- 练习 3：residual + RMSNorm

// 两输入两输出：读 x 和 residual，写回 acc = x + residual 以及 out = acc*rstd*w。
// 理想流量 4x（2 读 2 写）。省掉的是一个独立的 add kernel（1 读 1 写 + 1 次 launch）。
template <int VPT>
__global__ void __launch_bounds__(kThreads)
    fused_add_rms_norm_kernel(const __half* __restrict__ x, const __half* __restrict__ resid,
                              __half* __restrict__ resid_out, const __half* __restrict__ w,
                              __half* __restrict__ out, float eps) {
  __shared__ float smem[32];
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * (VPT * kThreads * kVec);
  const float4* xp = reinterpret_cast<const float4*>(x + row_base);
  const float4* rp = reinterpret_cast<const float4*>(resid + row_base);
  float4* rpo = reinterpret_cast<float4*>(resid_out + row_base);
  const float4* wp = reinterpret_cast<const float4*>(w);
  float4* op = reinterpret_cast<float4*>(out + row_base);

  float4 av[VPT], wv[VPT];
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int idx = threadIdx.x + i * kThreads;
    av[i] = add_8(xp[idx], rp[idx]);   // acc = x + residual
    rpo[idx] = av[i];                  // 顺手写回残差，省掉独立的 add kernel
    wv[i] = wp[idx];
    acc += sum_sq_8(av[i]);
  }

  const float rstd = rsqrtf(block_reduce_sum(acc, smem) / (VPT * kThreads * kVec) + eps);

#pragma unroll
  for (int i = 0; i < VPT; ++i)
    op[threadIdx.x + i * kThreads] = scale_mul_8(av[i], wv[i], rstd);
}

// --------------------------------------------------------------- 标量回退

__global__ void __launch_bounds__(kThreads)
    rms_norm_scalar_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                           __half* __restrict__ out, int n_cols, float eps) {
  __shared__ float smem[32];
  const size_t base = static_cast<size_t>(blockIdx.x) * n_cols;
  float acc = 0.f;
  for (int i = threadIdx.x; i < n_cols; i += kThreads) {
    const float v = __half2float(x[base + i]);
    acc = fmaf(v, v, acc);
  }
  const float rstd = rsqrtf(block_reduce_sum(acc, smem) / n_cols + eps);
  for (int i = threadIdx.x; i < n_cols; i += kThreads)
    out[base + i] = __float2half_rn(__half2float(x[base + i]) * __half2float(w[i]) * rstd);
}

__global__ void __launch_bounds__(kThreads)
    fused_add_rms_norm_scalar_kernel(const __half* __restrict__ x, const __half* __restrict__ resid,
                                     __half* __restrict__ resid_out, const __half* __restrict__ w,
                                     __half* __restrict__ out, int n_cols, float eps) {
  __shared__ float smem[32];
  const size_t base = static_cast<size_t>(blockIdx.x) * n_cols;
  float acc = 0.f;
  for (int i = threadIdx.x; i < n_cols; i += kThreads) {
    const float v = __half2float(x[base + i]) + __half2float(resid[base + i]);
    resid_out[base + i] = __float2half_rn(v);
    acc = fmaf(v, v, acc);
  }
  const float rstd = rsqrtf(block_reduce_sum(acc, smem) / n_cols + eps);
  for (int i = threadIdx.x; i < n_cols; i += kThreads)
    out[base + i] = __float2half_rn(__half2float(resid_out[base + i]) * __half2float(w[i]) * rstd);
}

constexpr int kVptMax = 8;

}  // namespace

// 返回 dispatch 到的 VPT；0 表示走了回退路径
int rms_norm_launch(const void* x, const void* w, void* out, int rows, int cols, float eps,
                    int impl, cudaStream_t stream) {
  const __half* xh = static_cast<const __half*>(x);
  const __half* wh = static_cast<const __half*>(w);
  __half* oh = static_cast<__half*>(out);

  const int vpt = cols / (kVec * kThreads);
  const bool vec_ok = (cols % (kVec * kThreads) == 0) && vpt >= 1 && vpt <= kVptMax;

  if (impl == 0 || !vec_ok) {
    if (cols % kVec == 0)
      rms_norm_v1_kernel<<<rows, kThreads, 0, stream>>>(xh, wh, oh, cols, eps);
    else
      rms_norm_scalar_kernel<<<rows, kThreads, 0, stream>>>(xh, wh, oh, cols, eps);
    CUDA_CHECK_LAUNCH();
    return 0;
  }

  switch (vpt) {
    case 1: rms_norm_v2_kernel<1><<<rows, kThreads, 0, stream>>>(xh, wh, oh, eps); break;
    case 2: rms_norm_v2_kernel<2><<<rows, kThreads, 0, stream>>>(xh, wh, oh, eps); break;
    case 4: rms_norm_v2_kernel<4><<<rows, kThreads, 0, stream>>>(xh, wh, oh, eps); break;
    case 8: rms_norm_v2_kernel<8><<<rows, kThreads, 0, stream>>>(xh, wh, oh, eps); break;
    default: rms_norm_v1_kernel<<<rows, kThreads, 0, stream>>>(xh, wh, oh, cols, eps); break;
  }
  CUDA_CHECK_LAUNCH();
  return vpt;
}

int fused_add_rms_norm_launch(const void* x, const void* residual_in, void* residual_out,
                              const void* w, void* out, int rows, int cols, float eps,
                              cudaStream_t stream) {
  const __half* xh = static_cast<const __half*>(x);
  const __half* rh = static_cast<const __half*>(residual_in);
  __half* roh = static_cast<__half*>(residual_out);
  const __half* wh = static_cast<const __half*>(w);
  __half* oh = static_cast<__half*>(out);

  const int vpt = cols / (kVec * kThreads);
  const bool vec_ok = (cols % (kVec * kThreads) == 0) && vpt >= 1 && vpt <= kVptMax;
  if (!vec_ok) {
    if (cols % kVec == 0) {
      fused_add_rms_norm_scalar_kernel<<<rows, kThreads, 0, stream>>>(xh, rh, roh, wh, oh, cols, eps);
      CUDA_CHECK_LAUNCH();
      return 0;
    }
    // 非 8 对齐：逐元素 + 块归约的标量版
    fused_add_rms_norm_scalar_kernel<<<rows, kThreads, 0, stream>>>(xh, rh, roh, wh, oh, cols, eps);
    CUDA_CHECK_LAUNCH();
    return 0;
  }

  switch (vpt) {
    case 1:
      fused_add_rms_norm_kernel<1><<<rows, kThreads, 0, stream>>>(xh, rh, roh, wh, oh, eps);
      break;
    case 2:
      fused_add_rms_norm_kernel<2><<<rows, kThreads, 0, stream>>>(xh, rh, roh, wh, oh, eps);
      break;
    case 4:
      fused_add_rms_norm_kernel<4><<<rows, kThreads, 0, stream>>>(xh, rh, roh, wh, oh, eps);
      break;
    case 8:
      fused_add_rms_norm_kernel<8><<<rows, kThreads, 0, stream>>>(xh, rh, roh, wh, oh, eps);
      break;
    default:
      fused_add_rms_norm_scalar_kernel<<<rows, kThreads, 0, stream>>>(xh, rh, roh, wh, oh, cols, eps);
      break;
  }
  CUDA_CHECK_LAUNCH();
  return vpt;
}
