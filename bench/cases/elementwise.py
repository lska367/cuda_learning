"""逐元素算子：衡量"带宽利用率"的最小单元，也是所有 elementwise 融合的模板。

默认 T=1024（张量 ≳ 20 MB，才可能跑成带宽受限）；研究 decode 时用 --sweep T=1,8,64。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..case import Case, Config, compare, compare_many, tok
from ..kernels import cuda_ops
from ..kernels.silu_mul_triton import silu_and_mul
from . import register


@register("elt/silu_mul")
def silu_mul(cfg: Config) -> Case:
    T, I = tok(cfg), cfg.inter
    x = torch.randn(T, 2 * I, device="cuda", dtype=cfg.dtype)
    out_t = torch.empty(T, I, device="cuda", dtype=cfg.dtype)
    out_k = torch.empty_like(out_t)
    out_c = torch.empty_like(out_t)

    def torch_impl():
        # 框架里的朴素写法：silu 一个 kernel + mul 一个 kernel，中间多一个 [T,I] 暂存
        tmp = F.silu(x[:, :I])
        torch.mul(tmp, x[:, I:], out=out_t)

    def triton_impl():
        silu_and_mul(x, out_k)

    variants = {"torch": torch_impl, "triton": triton_impl}
    launches = {"torch": 2, "triton": 1}
    outs = {"triton": out_k}
    if cuda_ops.available():
        variants["cuda"] = lambda: cuda_ops.silu_and_mul(x, out_c)
        launches["cuda"] = 1
        outs["cuda"] = out_c

    return Case(
        name="elt/silu_mul",
        variants=variants,
        # 最小必要流量：读 gate+up (2I)，写 out (I)
        nbytes=T * 3 * I * x.element_size(),
        nflops=T * I * 6,
        check=lambda: compare_many(out_t, outs),
        note="SwiGLU；torch 版 2 kernel，手写版一次读一次写",
        launches=2,
        launches_by_impl=launches,
    )


@register("elt/residual_add")
def residual_add(cfg: Config) -> Case:
    T, H = tok(cfg), cfg.hidden
    a = torch.randn(T, H, device="cuda", dtype=cfg.dtype)
    b = torch.randn(T, H, device="cuda", dtype=cfg.dtype)
    out = torch.empty_like(a)

    def torch_impl():
        torch.add(a, b, out=out)

    return Case(
        name="elt/residual_add",
        variants={"torch": torch_impl},
        nbytes=3 * T * H * a.element_size(),
        nflops=T * H,
        note="2 读 1 写；融合进 RMSNorm 可消掉整个 kernel",
    )


@register("elt/cast")
def cast(cfg: Config) -> Case:
    """最小的可能 kernel：用来测 launch 开销上限和"带宽天花板"。"""
    T, H = tok(cfg), cfg.hidden
    a = torch.randn(T, H, device="cuda", dtype=torch.float16)
    out = torch.empty(T, H, device="cuda", dtype=torch.float32)

    def torch_impl():
        out.copy_(a)

    return Case(
        name="elt/cast",
        variants={"torch": torch_impl},
        nbytes=T * H * (2 + 4),
        check=lambda: compare(out.to(torch.float16), a, atol=0, rtol=0),
        note="fp16→fp32 转换；一次多余 cast = 双倍流量",
    )
