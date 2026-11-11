"""CUDA 练习 kernel 的 JIT 编译入口。

为什么用 torch.utils.cpp_extension 而不是手写 nvcc + ctypes：
- 张量能直接传（不用自己管 data_ptr / stream）；
- 改完 .cu 下次 import 自动增量重编（改一个文件约 20–40 s）；
- TORCH_CUDA_ARCH_LIST 必须显式设成 7.0 —— 否则 torch 会按当前 GPU 猜，
  在只有一张 V100 的机器上一般没问题，但多卡/跨代时会编出一堆用不到的 arch，编译时间翻倍。

用法：
    from bench.cuda.build import ops
    ops.silu_mul_v1(x, out)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_SOURCES = [_DIR / "bindings.cpp", _DIR / "elementwise.cu", _DIR / "norm.cu"]

_ops = None


def _arch() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def build(verbose: bool = False, use_fast_math: bool = False):
    """编译并返回扩展模块（结果缓存，源文件没变就直接复用）。"""
    from torch.utils.cpp_extension import load

    cuda_flags = [
        "-O3",
        "-lineinfo",              # nsys/ncu 里能对到源码行
        "--expt-relaxed-constexpr",
        "-Xptxas", "-v",          # 打印寄存器/ spill，做优化时很有用
        "-Wno-deprecated-gpu-targets",
    ]
    if use_fast_math:
        cuda_flags.append("--use_fast_math")

    # torch 的 JIT 需要 PATH 上有 `ninja` 可执行文件；venv 里装了但 shell PATH 里通常没有。
    # 注意：sys.executable.resolve() 会跳到 uv 的基础解释器，必须用 sys.prefix。
    venv_bin = str(Path(sys.prefix) / "bin")
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch())
    os.environ.setdefault("MAX_JOBS", str(min(16, os.cpu_count() or 4)))

    return load(
        name="cuda_learning_ops",
        sources=[str(s) for s in _SOURCES],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        verbose=verbose,
    )


def ops():
    """惰性构建：import 时不编译，第一次调用才编译（避免 bench 里没用到也等）。"""
    global _ops
    if _ops is None:
        _ops = build()
    return _ops


def available() -> bool:
    """探测能否编译（用于 case 注册时优雅降级）。"""
    try:
        ops()
        return True
    except Exception as e:  # pragma: no cover
        print(f"[cuda] 扩展不可用：{e}")
        return False
