"""case 的数据结构与 roofline 判定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .calib import Peaks

__all__ = ["Case", "Config", "analyze", "compare", "compare_many", "n_launches", "tok"]


def tok(cfg: Config, default: int = 1024) -> int:
    """case 的默认 token 数。用户 --sweep T=… 时以用户为准。"""
    return max(cfg.T if cfg.T is not None else default, 1)


@dataclass
class Config:
    """一个 forward step 的形状。默认值 = Llama-3-8B。"""
    hidden: int = 4096
    inter: int = 14336
    vocab: int = 128256
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    n_layers: int = 32
    T: int | None = None  # 本 step 的 token 数；None = 由 case 决定默认值（decode 用 1，带宽类用 1024）
    S: int = 2048       # 上下文长度（decode 时的 KV 长度）
    B: int = 8          # batch（decode attention 用）
    dtype: torch.dtype = torch.float16

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


@dataclass
class Case:
    """一个待测 workload：多个实现 + 流量/算力记账 + 正确性检查。"""
    name: str
    variants: dict[str, Callable[[], None]] = field(default_factory=dict)
    nbytes: int = 0                      # 每次迭代搬多少字节（HBM）
    nflops: int = 0                      # 每次迭代多少 FLOP
    check: Callable[[], dict] | None = None
    note: str = ""
    launches: int = 1                              # 基线实现的 kernel 数（launch 开销占比用）
    launches_by_impl: dict[str, int] = field(default_factory=dict)


def n_launches(case: "Case", impl: str) -> int:
    return case.launches_by_impl.get(impl, case.launches)


def analyze(ms: float, nbytes: int, nflops: int, peaks: Peaks) -> dict:
    """一个数字不能说明问题：同时给带宽、算力、以及 roofline 预测。"""
    sec = ms * 1e-3
    gbps = nbytes / sec / 1e9 if nbytes else 0.0
    tflops = nflops / sec / 1e12 if nflops else 0.0
    pct_bw = gbps / peaks.bw_gbps if peaks.bw_gbps else 0.0
    pct_flop = tflops / peaks.fp16_tflops if peaks.fp16_tflops else 0.0
    ai = nflops / nbytes if nbytes else float("inf")
    t_mem = nbytes / (peaks.bw_gbps * 1e9) * 1e3
    t_flop = nflops / (peaks.fp16_tflops * 1e12) * 1e3
    bound = "mem" if t_mem >= t_flop else "flop"
    predicted = max(t_mem, t_flop)
    return {
        "ms": ms,
        "gbps": gbps,
        "pct_bw": pct_bw,
        "tflops": tflops,
        "pct_flop": pct_flop,
        "ai": ai,
        "bound": bound,
        "predicted_ms": predicted,
        "roofline_eff": predicted / ms if ms else 0.0,  # 达到 roofline 的比例
    }


def compare(out: torch.Tensor, ref: torch.Tensor, atol: float = 2e-2, rtol: float = 2e-2) -> dict:
    """对拍：max_abs / max_rel / 是否通过。

    fp16 累加与 fp32 累加的差异在这类 kernel 上可到 1e-2 量级，所以默认容差是 2e-2，
    不要用 1e-6 —— 那只会让你怀疑自己的 kernel 而其实是参考实现本身有误差。
    """
    o = out.float()
    r = ref.float()
    diff = (o - r).abs()
    denom = r.abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / denom).max().item(),
        "ok": bool(torch.allclose(o, r, atol=atol, rtol=rtol)),
    }


def compare_many(ref: torch.Tensor, outs: dict[str, torch.Tensor], atol: float = 2e-2,
                 rtol: float = 2e-2) -> dict:
    """多实现同时和同一个参考对拍，返回 {"ok", "<name>_ok", "<name>_max_abs", ...}"""
    res: dict = {}
    oks = []
    worst = 0.0
    for name, out in outs.items():
        c = compare(out, ref, atol=atol, rtol=rtol)
        res[f"{name}_ok"] = c["ok"]
        res[f"{name}_max_abs"] = c["max_abs"]
        worst = max(worst, c["max_abs"])
        oks.append(c["ok"])
    res["ok"] = all(oks)
    res["max_abs"] = worst
    return res
