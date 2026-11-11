"""标定 + 记录本机实测峰值。所有 roofline 判断都用这里的数字，不用 spec。"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from . import nvml
from .timer import do_bench, measure_launch_overhead

RESULTS = Path(__file__).resolve().parent / "results"
PEAKS_FILE = RESULTS / "machine.json"


@dataclass(frozen=True)
class Peaks:
    bw_gbps: float          # 实测 HBM 带宽（GB/s）
    fp16_tflops: float      # 实测 fp16 Tensor Core 峰值
    fp32_tflops: float      # 实测 fp32 CUDA core 峰值
    launch_us: float        # 单次 kernel launch 成本

    @property
    def balance_fp16(self) -> float:
        """机器平衡点 AI（FLOP/byte）：低于它访存受限，高于它算力受限。"""
        return self.fp16_tflops * 1e12 / (self.bw_gbps * 1e9)


def _matmul_tflops(n: int = 8192, dtype=torch.float16, iters: int = 6) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    fn = lambda: a @ b
    st = do_bench(fn, warmup=50, rep=80, flush_l2=False)
    for _ in range(iters):
        fn()
    return 2 * n ** 3 / (st["median"] * 1e-3) / 1e12


def _copy_bw_gbps(sizes_mb: tuple[int, ...] = (256, 512, 1024)) -> tuple[float, float]:
    """拷贝带宽取多个尺寸的最大值。

    小张量测出来会偏高/偏低（launch 与 L2 都在干扰）：256 MB 只有 631 GB/s，
    1 GB 才稳定到 663 GB/s。所以"峰值"必须在大尺寸上取 max，否则下游 %BW 会虚高。
    """
    best_copy = best_fill = 0.0
    for mb in sizes_mb:
        n = mb * 1024 * 1024 // 4  # fp32 元素数
        x = torch.empty(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        x.fill_(1.0)
        copy = do_bench(lambda: y.copy_(x), warmup=50, rep=80, flush_l2=False)
        fill = do_bench(lambda: y.fill_(0.0), warmup=50, rep=80, flush_l2=False)
        best_copy = max(best_copy, 2 * n * 4 / (copy["median"] * 1e-3) / 1e9)
        best_fill = max(best_fill, n * 4 / (fill["median"] * 1e-3) / 1e9)
        del x, y
        torch.cuda.empty_cache()
    return best_copy, best_fill


def _python_version() -> str:
    return platform.python_version()


def calibrate(verbose: bool = True) -> dict:
    assert torch.cuda.is_available(), "CUDA 不可用"
    torch.backends.cuda.matmul.allow_tf32 = False  # V100 无 TF32，显式关掉避免误解
    props = torch.cuda.get_device_properties(0)

    if verbose:
        print(f"[calib] {props.name}  sm_{props.major}{props.minor}  "
              f"SMs={props.multi_processor_count}  L2={props.L2_cache_size/2**20:.0f}MiB")

    with nvml.ClockSampler(interval=0.05) as sampler:
        bw_copy, bw_fill = _copy_bw_gbps()
        fp16 = _matmul_tflops(dtype=torch.float16)
        fp32 = _matmul_tflops(dtype=torch.float32, n=6144)
        launch_us = measure_launch_overhead()

    # 读带宽和写带宽不同（HBM 写通常更低），copy 是 1:1 读写，取它作为综合带宽
    peaks = Peaks(
        bw_gbps=round(bw_copy, 1),
        fp16_tflops=round(fp16, 2),
        fp32_tflops=round(fp32, 2),
        launch_us=round(launch_us, 2),
    )
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": {
            "name": props.name,
            "sm": f"sm_{props.major}{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_mem_gib": round(props.total_memory / 2**30, 1),
            "l2_mib": round(props.L2_cache_size / 2**20, 1),
        },
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": _python_version(),
            "arch_list": torch.cuda.get_arch_list(),
            "driver": _driver_version(),
        },
        "measured": asdict(peaks),
        "peaks": asdict(peaks),
        "extra": {
            "fill_bw_gbps": round(bw_fill, 1),
            "balance_fp16_flop_per_byte": round(peaks.balance_fp16, 1),
            "spec_bw_gbps": 900.0,
            "spec_fp16_tflops": 113.0,
            "bw_pct_of_spec": round(100 * bw_copy / 900.0, 1),
            "fp16_pct_of_spec": round(100 * fp16 / 113.0, 1),
        },
        "clocks_during_calib": sampler.summary(),
    }
    RESULTS.mkdir(exist_ok=True)
    PEAKS_FILE.write_text(json.dumps(info, indent=2, ensure_ascii=False))

    if verbose:
        print(f"[calib] HBM  copy {bw_copy:7.1f} GB/s | fill {bw_fill:7.1f} GB/s "
              f"({info['extra']['bw_pct_of_spec']:.0f}% of 900)")
        print(f"[calib] FP16 TC {fp16:7.2f} TFLOPS ({info['extra']['fp16_pct_of_spec']:.0f}% of 113) "
              f"| FP32 {fp32:.2f} TFLOPS")
        print(f"[calib] 平衡点  {peaks.balance_fp16:6.1f} FLOP/byte  | launch {launch_us:.2f} µs")
        print(f"[calib] {sampler.format()}")
        print(f"[calib] -> {PEAKS_FILE}")
    return info


def _driver_version() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def load_peaks(path: Path = PEAKS_FILE) -> Peaks:
    if not path.exists():
        print("[calib] 没有 machine.json，先跑一次标定 …")
        return Peaks(**calibrate(verbose=False)["measured"])
    return Peaks(**json.loads(path.read_text())["measured"])


if __name__ == "__main__":
    calibrate()
