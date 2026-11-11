"""手写 kernel 的放置点。每个文件导出一个或多个 kernel，case 里按名字注册成 variant。

约定：kernel 只做计算，张量由 case 分配好传进来 —— 这样 bench 才能复用缓冲区、
避免把 allocation 的时间算进 kernel 时间。
"""

from . import rmsnorm_triton, silu_mul_triton  # noqa: F401
