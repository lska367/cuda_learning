"""CUDA 练习 kernel 的 Python 侧包装：签名和 Triton 版保持一致，方便在 case 里互换。"""

from __future__ import annotations

import torch

from .build import ops

__all__ = ["silu_mul_v0", "silu_mul_v1", "rms_norm_v0", "rms_norm_v1", "fused_add_rms_norm"]


def silu_mul_v0(x: torch.Tensor, out: torch.Tensor) -> None:
    ops().silu_mul_v0(x, out)


def silu_mul_v1(x: torch.Tensor, out: torch.Tensor) -> None:
    ops().silu_mul_v1(x, out)


def rms_norm_v0(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float = 1e-6) -> None:
    ops().rms_norm_v0(x, w, out, eps)


def rms_norm_v1(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float = 1e-6) -> None:
    ops().rms_norm_v1(x, w, out, eps)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor,
                       out: torch.Tensor, eps: float = 1e-6) -> None:
    """residual 会被就地更新（两输入两输出，和 vLLM 语义一致）。"""
    ops().fused_add_rms_norm(x, residual, w, out, eps)
