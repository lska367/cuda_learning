"""RMSNorm：一行一个 CTA，warp shuffle + shared memory 两级归约。

练习 2 的参考实现。要点：
- 一行（hidden=4096 fp16 = 8 KB）只读一次、写一次，所以性能上限就是带宽；
- fp32 累加平方和（fp16 直接累加在 4096 长度上会掉精度）；
- H 不是 2 的幂时用 mask，BLOCK 取 next_pow2(H)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * n_cols + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rms_norm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float = 1e-6) -> None:
    T, H = x.shape
    block = triton.next_power_of_2(H)
    num_warps = 8 if block >= 2048 else 4
    _rms_norm_kernel[(T,)](x, w, out, H, eps, BLOCK=block, num_warps=num_warps)
