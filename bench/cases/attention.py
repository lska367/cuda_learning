"""Attention：prefill（算力受限，S²）和 decode（带宽受限，KV 流）分开测。

decode 的基线用 bmm 手写 GQA（4 个 kernel），因为 torch SDPA 在 V100 上对
B=1/32 heads 这种 decode 形状会退到 math 后端，比手写更慢。
下一步把自己写的 flash attention / paged decode 注册成 variant 来对比。
"""

from __future__ import annotations

import torch

from ..case import Case, Config, compare, tok
from . import register


@register("attn/decode")
def decode_attention(cfg: Config) -> Case:
    B, S, H, kvH, D = cfg.B, max(cfg.S, 1), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    G = H // kvH
    q4 = torch.randn(B, kvH, G, D, device="cuda", dtype=cfg.dtype)      # = [B,H,D] 重排
    k = torch.randn(B, kvH, S, D, device="cuda", dtype=cfg.dtype)       # paged cache 的自然布局
    v = torch.randn_like(k)
    out = torch.empty(B, kvH, G, D, device="cuda", dtype=cfg.dtype)

    qf = q4.view(B * kvH, G, D)
    kt = k.view(B * kvH, S, D).transpose(-1, -2)
    vf = v.view(B * kvH, S, D)
    of = out.view(B * kvH, G, D)

    def torch_impl():
        # 4 个 kernel：QK^T → scale → softmax → PV（scale 融进 softmax 前的乘）
        p = torch.softmax(torch.bmm(qf, kt) * (D ** -0.5), dim=-1)
        of.copy_(torch.bmm(p, vf))

    def check():
        p32 = torch.softmax((qf.float() @ kt.float()) * (D ** -0.5), dim=-1)
        ref = (p32 @ vf.float()).to(cfg.dtype)
        return compare(of, ref, atol=5e-2, rtol=5e-2)

    return Case(
        name="attn/decode",
        variants={"torch": torch_impl},
        # 理想流量：读完整 KV cache（GQA 只读 kvH 份）+ q/out
        nbytes=2 * B * S * kvH * D * q4.element_size() + 2 * B * H * D * q4.element_size(),
        # 每个 (b,h): QK^T 2·S·D + PV 2·S·D
        nflops=4 * B * H * S * D,
        check=check,
        note="非 paged 基线；KV 流量随 S 线性增长，S=128k 时超过权重流",
        launches=4,
    )


@register("attn/prefill")
def prefill_attention(cfg: Config) -> Case:
    B, S, H, D = cfg.B, tok(cfg), cfg.n_heads, cfg.head_dim
    q = torch.randn(B, H, S, D, device="cuda", dtype=cfg.dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)

    def torch_impl():
        out.copy_(torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True))

    def check():
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=True).to(cfg.dtype)
        return compare(out, ref, atol=5e-2, rtol=5e-2)

    return Case(
        name="attn/prefill",
        variants={"torch": torch_impl},
        nbytes=4 * B * H * S * D * q.element_size(),
        # causal 只算下三角 ≈ S²/2
        nflops=B * H * S * S * D,
        check=check,
        note="causal FMHA；FLOPs 按 S²/2 记（online softmax 不物化 S×S）",
        launches=2,
    )
