// 纯 C++/CUDA 接口，不引入 torch 头文件 —— 这样 .cu 的编译时间能压到十几秒。
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK_LAUNCH()                                                     \
  do {                                                                          \
    cudaError_t err_ = cudaGetLastError();                                       \
    if (err_ != cudaSuccess) {                                                   \
      fprintf(stderr, "[cuda] launch failed: %s (%s:%d)\n",                      \
              cudaGetErrorString(err_), __FILE__, __LINE__);                     \
      std::abort();                                                              \
    }                                                                            \
  } while (0)

// out = silu(x[:, :I]) * x[:, I:]，x 布局 [rows, 2*cols]
void silu_and_mul_launch(const void* x, void* out, int rows, int cols,
                         cudaStream_t stream);

// 行归一化：out = x * rsqrt(mean(x^2)+eps) * w，布局 [rows, cols]
// impl: 0 = v1 两遍读（反面教材），1 = v2 寄存器缓存单遍
int rms_norm_launch(const void* x, const void* w, void* out, int rows, int cols,
                    float eps, int impl, cudaStream_t stream);

// acc = x + residual; out = rmsnorm(acc) * w；acc 同时写回 residual_out
int fused_add_rms_norm_launch(const void* x, const void* residual_in,
                              void* residual_out, const void* w, void* out,
                              int rows, int cols, float eps, cudaStream_t stream);
