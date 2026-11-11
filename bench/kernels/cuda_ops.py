"""CUDA kernel 的 Python 入口：首次 import 时用 torch.utils.cpp_extension 编译，之后走缓存。

为什么要 JIT 而不是 setup.py：改一行 .cu 就能重编（增量、只编改动的文件），
学习阶段这比打包安装快得多。产物落在 bench/.build/（已 gitignore）。

编译失败（没有 nvcc / gcc 版本不匹配）不会让 bench 挂掉：cuda variants 会被跳过。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

__all__ = ["available", "load_module", "silu_and_mul", "rms_norm", "fused_add_rms_norm", "build_error"]

_CSRC = Path(__file__).resolve().parent / "csrc"
_BUILD = Path(__file__).resolve().parent.parent / ".build"
_SOURCES = ["bindings.cpp", "elementwise.cu", "norm.cu"]

_state: dict = {"mod": None, "err": None, "tried": False}

# V100 = sm_70。用 env 而不是 extra_cuda_cflags，让 torch 自己拼 -gencode，
# 避免和它默认的 --generate-code 冲突（重复指定会报 "arch redefine"）。
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_BUILD))

# ninja 装在 venv 里，但没 activate 时不在 PATH 上；cpp_extension 是靠 which("ninja") 找它
_venv_bin = str(Path(sys.prefix) / "bin")
if _venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")


def load_module():
    if _state["tried"]:
        return _state["mod"]
    _state["tried"] = True
    try:
        from torch.utils.cpp_extension import load

        _state["mod"] = load(
            name="llm_bench_ops",
            sources=[str(_CSRC / s) for s in _SOURCES],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
            with_cuda=True,
        )
    except Exception as exc:  # noqa: BLE001 - 编译失败就降级
        _state["err"] = f"{type(exc).__name__}: {exc}"
        print(f"[cuda_ops] 编译失败，跳过 cuda variants：{_state['err']}")
    return _state["mod"]


def available() -> bool:
    return load_module() is not None


def build_error() -> str | None:
    load_module()
    return _state["err"]


def _mod():
    m = load_module()
    if m is None:
        raise RuntimeError(f"CUDA extension 不可用：{_state['err']}")
    return m


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    """x: [T, 2I] fp16 → out: [T, I] fp16"""
    _mod().silu_and_mul(x, out)


def rms_norm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float = 1e-6,
             impl: int = 1) -> int:
    """impl=0: v1 两遍读；impl=1: v2 寄存器缓存。返回 dispatch 到的 VPT（0=回退路径）。"""
    return _mod().rms_norm(x, w, out, eps, impl)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, residual_out: torch.Tensor,
                       w: torch.Tensor, out: torch.Tensor, eps: float = 1e-6) -> int:
    return _mod().fused_add_rms_norm(x, residual, residual_out, w, out, eps)
