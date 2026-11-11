"""RoPE（NeoX half-split 布局，Llama 系）。最容易写错的一类算子。

注意两种布局：
- NeoX / half-split（Llama）：前一半和后一半配对，rotate_half 用 cat(-x2, x1)；
- GPT-J / interleaved：相邻两维配对 (x[2i], x[2i+1])。
混用不会报错，只会让模型悄悄变蠢 —— 所以必须有对拍。

基线特意做成"4 kernel / 张量"，不搞 cat 和 move：这样它是一个公平的对手，
你自己的 fused RoPE（含 KV 写入）应该能打到 2 倍以上。
"""

from __future__ import annotations

import torch

from ..case import Case, Config, compare, tok
from . import register


@register("rope/neox")
def rope(cfg: Config) -> Case:
    T, H, kvH, D = tok(cfg), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    half = D // 2
    max_pos = max(cfg.S, T, 1)

    q = torch.randn(T, H, D, device="cuda", dtype=cfg.dtype)
    k = torch.randn(T, kvH, D, device="cuda", dtype=cfg.dtype)

    # cos/sin 表预计算成 [max_pos, half] 并转成目标精度（kernel 里只做 gather）
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device="cuda", dtype=torch.float32) / half))
    pos = torch.arange(max_pos, device="cuda", dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().to(cfg.dtype).unsqueeze(1)   # [max_pos, 1, half]
    sin = freqs.sin().to(cfg.dtype).unsqueeze(1)

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    def _rope_into(src: torch.Tensor, dst: torch.Tensor) -> None:
        c, s = cos[:T], sin[:T]
        x1, x2 = src[..., :half], src[..., half:]
        d1, d2 = dst[..., :half], dst[..., half:]
        torch.mul(x1, c, out=d1)          # d1 = x1*cos
        d1.addcmul_(x2, s, value=-1)      # d1 -= x2*sin
        torch.mul(x1, s, out=d2)          # d2 = x1*sin
        d2.addcmul_(x2, c, value=1)       # d2 += x2*cos

    def torch_impl():
        _rope_into(q, out_q)
        _rope_into(k, out_k)

    def check():
        def ref(x):
            c, s = cos[:T].float(), sin[:T].float()
            x1, x2 = x.float()[..., :half], x.float()[..., half:]
            return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(cfg.dtype)
        return compare(out_q, ref(q), atol=3e-2, rtol=3e-2) | {
            "k_ok": bool(torch.allclose(out_k, ref(k), atol=3e-2, rtol=3e-2))
        }

    return Case(
        name="rope/neox",
        variants={"torch": torch_impl},
        # 读 q+k，写 out_q+out_k；K 只对 kv_heads 做，流量比 Q 小 4 倍
        nbytes=(T * H * D + T * kvH * D) * q.element_size() * 2,
        nflops=(T * H * D + T * kvH * D) * 3,
        check=check,
        note="half-split 布局；纯带宽活，目标是融合成 1 个 kernel 并把 KV 直接写进 cache",
        launches=8,
    )
