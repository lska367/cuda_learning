#pragma once
// 所有 CUDA 练习共用的头文件

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// kernel launch 之后立刻查错：非法 grid/block、越界写导致的 launch error 都在这里暴露，
// 而不是等几行之后的某个 sync 报出来（那时候堆栈已经和出错的地方无关了）。
#define CUDA_CHECK_KERNEL()                                                        \
    do {                                                                           \
        cudaError_t err__ = cudaGetLastError();                                     \
        TORCH_CHECK(err__ == cudaSuccess, "CUDA kernel error: ",                    \
                    cudaGetErrorString(err__));                                     \
    } while (0)

// __shfl_xor_sync 的蝶形归约，warp 内 32 个 lane 求和（无 smem、无 barrier）
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
