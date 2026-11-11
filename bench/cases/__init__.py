"""case 注册表。新增 case：写一个 @register("family/name") 的函数，返回 Case。"""

from __future__ import annotations

from typing import Callable

from ..case import Case, Config

CASES: dict[str, Callable[[Config], Case]] = {}


def register(name: str):
    def deco(fn):
        if name in CASES:
            raise ValueError(f"duplicate case {name}")
        CASES[name] = fn
        return fn
    return deco


from . import attention, elementwise, gemm, norm, rope  # noqa: E402,F401
