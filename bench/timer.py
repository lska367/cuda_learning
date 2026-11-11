"""CUDA 计时工具：warmup + 多次采样 + L2 flush + CUDA Graph 三种模式。

设计要点（都是"测出来能信"的必要条件）：
1. 用 CUDA event 而不是 time.time() —— 后者包含 launch 开销和 host 抖动。
2. 每次迭代之间清空 L2：V100 只有 6 MiB L2，小张量（hidden=4096 fp16 只有 8 KB）
   会整个驻留 L2，不 flush 会测出"比 HBM 还快"的假象。
3. 报告分位数而不是单个均值：GPU 有时钟抖动/其他租户干扰，看 p20/p50/p80 才知道稳不稳。
4. CUDA Graph 模式用来把"kernel 本身"和"launch 开销"分开。
"""

from __future__ import annotations

import statistics
from typing import Callable, Sequence

import torch

__all__ = ["do_bench", "do_bench_cudagraph", "measure_launch_overhead", "format_stats"]

# 默认 flush 缓冲区大小。要明显大于 L2（V100: 6 MiB），64 MiB 足够且开销可忽略
# （660 GB/s 下写 64 MiB ≈ 0.1 ms，对 0.5 ms 以上的 kernel 影响 < 20%）。
DEFAULT_FLUSH_MB = 64


def _sync():
    torch.cuda.synchronize()


def _quantiles(vals: Sequence[float], qs: Sequence[float]) -> list[float]:
    s = sorted(vals)
    n = len(s)
    out = []
    for q in qs:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        out.append(s[idx])
    return out


def do_bench(
    fn: Callable[[], None],
    warmup: float = 25.0,
    rep: float = 100.0,
    quantiles: Sequence[float] = (0.5, 0.2, 0.8),
    flush_l2: bool = True,
    flush_mb: int = DEFAULT_FLUSH_MB,
) -> dict:
    """返回 {'median','p20','p80','mean','std','n_repeat'}，单位 ms。

    warmup/rep 的单位是毫秒（和 triton.testing.do_bench 一致）：先估一次运行时间，
    再决定重复次数，所以不用手调 n_warmup/n_repeat。
    """
    # 1) 估时（不计入结果）
    for _ in range(3):
        fn()
    _sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    _sync()
    est_ms = max(start.elapsed_time(end) / 5.0, 1e-3)

    n_warmup = max(1, int(warmup / est_ms))
    n_repeat = max(5, int(rep / est_ms))

    cache = None
    if flush_l2:
        # int8 缓冲区：只做写，不读，纯粹是为了把 L2 里的脏行顶出去
        cache = torch.empty(flush_mb * 1024 * 1024, dtype=torch.int8, device="cuda")

    for _ in range(n_warmup):
        fn()
    _sync()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        if cache is not None:
            cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    _sync()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    q = _quantiles(times, quantiles)
    return {
        "median": q[0],
        "p20": q[1],
        "p80": q[2],
        "mean": statistics.fmean(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "n_repeat": n_repeat,
    }


def do_bench_cudagraph(fn: Callable[[], None], rep: int = 100, warmup: int = 10) -> dict:
    """把 fn 捕获成 CUDA Graph，重复 rep 次。

    用途：
    - 测「纯 kernel 时间」：消除 python + launch 开销；
    - 与 do_bench 相减，得到每个 kernel 的 launch 成本。
    注意：fn 里的所有张量地址在捕获期间必须固定，且不能有 host-device 同步
    （比如 .item()、动态 shape、pin_memory 拷贝）——这正是 CUDA Graph 在推理里的约束。
    """
    for _ in range(warmup):
        fn()
    _sync()

    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    _sync()

    with torch.cuda.graph(g):
        for _ in range(rep):
            fn()
    _sync()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        g.replay()
    end.record()
    _sync()
    per_iter = start.elapsed_time(end) / 5.0 / rep
    return {"median": per_iter, "p20": per_iter, "p80": per_iter,
            "mean": per_iter, "std": 0.0, "n_repeat": rep}


def measure_launch_overhead(n: int = 200) -> float:
    """空 kernel 的端到端 launch 成本（µs/次）。

    decode 时每层约 10 个小 kernel、32 层 = 320 次 launch；
    如果 launch 是 5 µs，单独就是 1.6 ms —— 这就是 CUDA Graph 的必要性来源。
    """
    x = torch.zeros(1, device="cuda")
    fn = lambda: x.add_(1.0)
    for _ in range(20):
        fn()
    _sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / n * 1e3


def format_stats(st: dict) -> str:
    return f"{st['median']:8.3f} ms  (p20 {st['p20']:7.3f} / p80 {st['p80']:7.3f})"
