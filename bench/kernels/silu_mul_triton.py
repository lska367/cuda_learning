"""fused silu_and_mul（SwiGLU）：out = silu(x[:, :I]) * x[:, I:]

练习 1 的参考实现。要点：
- 一次读两半、一次写，读:写 = 2:1；
- BLOCK 之内做连续访问，尾块 mask；
- 用 1D grid + pid 拆 (row, col_block)，避免 T=1 时 grid 太小。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(x_ptr, out_ptr, n_col_blocks, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // n_col_blocks
    cb = pid % n_col_blocks
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row * (2 * n_cols)
    gate = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    # silu(x) = x * sigmoid(x)。fp32 中间量，避免 fp16 上 sigmoid 溢出
    res = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row * n_cols + cols, res.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor, block: int = 1024) -> None:
    """x: [T, 2I]，out: [T, I]（就地写入 out）"""
    T, two_i = x.shape
    I = two_i // 2
    n_col_blocks = triton.cdiv(I, block)
    grid = (T * n_col_blocks,)
    _silu_mul_kernel[grid](x, out, n_col_blocks, I, BLOCK=block, num_warps=4)
