"""归一化：每层 2 次，纯带宽活，融合收益最容易量化的一类。

同一份 case 里放 4 个实现，用来看同一笔流量下不同做法的差距：
  torch      : fp32 upcast + 5 个 kernel（框架朴素写法）
  triton     : 一行一个 program（手写 kernel 的常见起点）
  cuda_v1    : CUDA 两遍读（读 x → 归约 → 再读 x → 写）；账面 3x 流量
  cuda       : CUDA 寄存器缓存单遍（读 x 进寄存器 → 归约 → 写）；账面 2x 流量
"""

from __future__ import annotations

import torch

from ..case import Case, Config, compare_many, tok
from ..kernels import cuda_ops
from ..kernels.rmsnorm_triton import rms_norm
from . import register

EPS = 1e-6


def _naive_rms(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """朴素实现：fp32 upcast → 平方 → 均值 → rsqrt → 乘 weight，约 5 个 kernel。"""
    xf = x.float()
    var = xf.mul(xf).mean(-1, keepdim=True)
    out.copy_((xf * torch.rsqrt(var + EPS) * w.float()).to(out.dtype))


def _fp32_rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    var = xf.mul(xf).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + EPS) * w.float()).to(x.dtype)


@register("norm/rmsnorm")
def rmsnorm(cfg: Config) -> Case:
    T, H = tok(cfg), cfg.hidden
    x = torch.randn(T, H, device="cuda", dtype=cfg.dtype)
    w = torch.randn(H, device="cuda", dtype=cfg.dtype)
    out_t = torch.empty_like(x)
    out_k = torch.empty_like(x)
    out_v1 = torch.empty_like(x)
    out_c = torch.empty_like(x)

    variants = {
        "torch": lambda: _naive_rms(x, w, out_t),
        "triton": lambda: rms_norm(x, w, out_k),
        "cuda_v1": lambda: cuda_ops.rms_norm(x, w, out_v1, EPS, impl=0),
    }
    launches = {"torch": 5, "triton": 1, "cuda_v1": 1}
    if cuda_ops.available():
        variants["cuda"] = lambda: cuda_ops.rms_norm(x, w, out_c, EPS, impl=1)
        launches["cuda"] = 1
        outs = {"triton": out_k, "cuda_v1": out_v1, "cuda": out_c}
    else:
        outs = {"triton": out_k, "cuda_v1": out_v1}

    return Case(
        name="norm/rmsnorm",
        variants=variants,
        nbytes=2 * T * H * x.element_size() + H * w.element_size(),
        nflops=4 * T * H,
        check=lambda: compare_many(_fp32_rms(x, w), outs, atol=4e-2, rtol=4e-2),
        note="v1 账面 3x 流量、v2 账面 2x；%BW 按理想 2x 记，v1 的超额读不可见",
        launches=5,
        launches_by_impl=launches,
    )


@register("norm/fused_add_rmsnorm")
def fused_add_rmsnorm(cfg: Config) -> Case:
    """residual add + RMSNorm 融合成一个 kernel：两输入两输出。"""
    T, H = tok(cfg), cfg.hidden
    x = torch.randn(T, H, device="cuda", dtype=cfg.dtype)
    residual = torch.randn(T, H, device="cuda", dtype=cfg.dtype)
    w = torch.randn(H, device="cuda", dtype=cfg.dtype)

    res_t = residual.clone()
    norm_t = torch.empty_like(x)
    res_c = residual.clone()
    norm_c = torch.empty_like(x)
    res_orig = residual.clone()

    def torch_impl():
        # 必须幂等：bench 会反复调用同一个 fn，in-place 累加会让 residual 一直长大
        torch.add(residual, x, out=res_t)   # kernel 1
        _naive_rms(res_t, w, norm_t)        # kernel 2..6

    variants = {"torch": torch_impl}
    launches = {"torch": 6}
    if cuda_ops.available():
        variants["cuda"] = lambda: cuda_ops.fused_add_rms_norm(x, residual, res_c, w, norm_c, EPS)
        launches["cuda"] = 1

    def check():
        ref = _fp32_rms(x + res_orig, w)
        outs = {"torch": norm_t}
        if "cuda" in variants:
            outs["cuda"] = norm_c
        res = compare_many(ref, outs, atol=4e-2, rtol=4e-2)
        res["acc_ok"] = bool(torch.allclose(res_t, x + res_orig, atol=2e-2, rtol=2e-2))
        if "cuda" in variants:
            res["acc_ok"] = res["acc_ok"] and bool(
                torch.allclose(res_c, x + res_orig, atol=2e-2, rtol=2e-2))
        res["ok"] = res["ok"] and res["acc_ok"]
        return res

    return Case(
        name="norm/fused_add_rmsnorm",
        variants=variants,
        # 理想流量：读 x + residual，写回 residual + normed
        nbytes=4 * T * H * x.element_size() + H * w.element_size(),
        nflops=4 * T * H,
        check=check,
        note="每层两侧各一次；融合后 1 个 kernel 干掉 6 个",
        launches=6,
        launches_by_impl=launches,
    )
