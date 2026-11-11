"""benchmark 期间采样 SM 频率 / 功耗 / 温度。

为什么必须有这个：GPU 的实际算力取决于**采样时刻的频率**，而 nvidia-smi 的
`clocks.sm` 常常不可信（V100 上我们实测到它报 1380 MHz，用 clock64 反推只有 ~340 MHz）。
benchmark 如果不记录频率，跨天的两次结果就没法比较。

用法：
    with ClockSampler() as s:
        do_bench(...)
    print(s.summary())   # sm_mhz: min/mean/max, power_w: mean/max
"""

from __future__ import annotations

import threading
import time

__all__ = ["ClockSampler", "query_clocks_once"]


def _nvml():
    try:
        import pynvml
    except ImportError:  # pragma: no cover
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None
    return pynvml, h


class ClockSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []  # (t, sm_mhz, power_w)
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._nv = _nvml()

    def _loop(self):
        pynvml, h = self._nv
        t0 = time.time()
        while not self._stop.is_set():
            try:
                sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                self.samples.append((time.time() - t0, sm, pw))
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        if self._nv is not None:
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        sm = [s[1] for s in self.samples]
        pw = [s[2] for s in self.samples]
        return {
            "sm_mhz_min": min(sm),
            "sm_mhz_mean": sum(sm) / len(sm),
            "sm_mhz_max": max(sm),
            "power_w_mean": sum(pw) / len(pw),
            "power_w_max": max(pw),
            "n_samples": len(sm),
        }

    def format(self) -> str:
        s = self.summary()
        if not s:
            return "nvml: unavailable"
        return (f"SM {s['sm_mhz_min']:.0f}/{s['sm_mhz_mean']:.0f}/{s['sm_mhz_max']:.0f} MHz "
                f"min/mean/max | power mean {s['power_w_mean']:.0f} W max {s['power_w_max']:.0f} W")


def query_clocks_once() -> dict:
    nv = _nvml()
    if nv is None:
        return {}
    pynvml, h = nv
    try:
        return {
            "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        }
    except Exception:
        return {}
