"""线性层：decode 是 GEMV，prefill 是 GEMM —— 同一个算子两个世界。

用法示例（看 M 如何改变 roofline 位置）:
    python -m bench.run -c 'gemm/qkv' --sweep T=1,8,64,512,2048
"""

from __future__ import annotations

import torch

from ..case import Case, Config, compare, tok
from . import register


def _gemm(name: str, K: int, N: int, note: str) -> None:
    @register(name)
    def make(cfg: Config) -> Case:
        M = tok(cfg, default=1)   # 默认按 decode（GEMV）测；--sweep T=… 看 prefill
        a = torch.randn(M, K, device="cuda", dtype=cfg.dtype)
        w = torch.randn(N, K, device="cuda", dtype=cfg.dtype)
        out = torch.empty(M, N, device="cuda", dtype=cfg.dtype)
        _ref: list[torch.Tensor] = []

        def torch_impl():
            torch.matmul(a, w.t(), out=out)

        def check():
            # 参考：fp32 累加，只在 check 时算一次，不进计时
            if not _ref:
                _ref.append((a.float() @ w.float().t()).to(cfg.dtype))
            return compare(out, _ref[0], atol=0.5, rtol=3e-2)

        return Case(
            name=name,
            variants={"torch": torch_impl},
            nbytes=(M * K + K * N + M * N) * a.element_size(),
            nflops=2 * M * K * N,
            check=check,
            note=note,
        )


# Llama-3-8B 的四个大 GEMM
_gemm("gemm/qkv", K=4096, N=6144, note="fused QKV proj（3 个投影拼一起）")
_gemm("gemm/o_proj", K=4096, N=4096, note="attention 输出投影")
_gemm("gemm/gate_up", K=4096, N=28672, note="FFN 第一层（每层最大的权重）")
_gemm("gemm/down", K=14336, N=4096, note="FFN 第二层")
_gemm("gemm/lm_head", K=4096, N=128256, note="logits：decode 时第二大的权重流")
