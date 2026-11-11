"""bench 入口。

常用：
    python -m bench.run --list
    python -m bench.run                              # 全部 case，Llama-3-8B 形状
    python -m bench.run -c norm/* -c elt/*           # 按 glob 过滤（shell 里要加引号）
    python -m bench.run -c gemm/qkv --sweep T=1,8,64,512
    python -m bench.run --csv bench/results/$(date +%F).csv
    python -m bench.run --cudagraph                  # 去掉 launch 开销看纯 kernel 时间
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tabulate import tabulate

from .calib import RESULTS, load_peaks
from .case import Case, Config, analyze, n_launches
from .cases import CASES
from .nvml import ClockSampler
from .timer import do_bench, do_bench_cudagraph


def build_cases(patterns: list[str], sweeps: dict[str, list[int]], base: Config) -> list[tuple[Config, str, object]]:
    names = sorted(CASES)
    if patterns:
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)]
    if not names:
        sys.exit(f"没有匹配的 case：{patterns}\n可用：{', '.join(sorted(CASES))}")

    # sweep：把 key=v1,v2 展开成多个 Config
    grid: list[dict] = [{}]
    for key, vals in sweeps.items():
        grid = [dict(g, **{key: v}) for g in grid for v in vals]

    out = []
    for name in names:
        for override in grid:
            cfg = Config(**{**base.__dict__, **override})
            suffix = ""
            if override:
                suffix = "  " + " ".join(f"{k}={v}" for k, v in override.items())
            out.append((cfg, name, suffix))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LLM 推理算子 bench")
    p.add_argument("-c", "--case", action="append", default=[], help="case 名/glob，可重复")
    p.add_argument("--list", action="store_true", help="列出所有 case")
    p.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2",
                   help="对 Config 字段做扫描，例：--sweep T=1,64,1024")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL", help="覆盖 Config 字段")
    p.add_argument("--warmup", type=float, default=25.0, help="warmup 毫秒")
    p.add_argument("--rep", type=float, default=100.0, help="测量毫秒")
    p.add_argument("--cudagraph", action="store_true", help="用 CUDA Graph 测（去 launch 开销）")
    p.add_argument("--no-check", action="store_true", help="跳过正确性对拍")
    p.add_argument("--csv", type=Path, help="把结果追加写入 csv")
    p.add_argument("--json", type=Path, help="把结果写入 json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        print("可用 case：")
        for n in sorted(CASES):
            print(f"  {n}")
        return

    sweeps: dict[str, list[int]] = {}
    for item in args.sweep:
        k, v = item.split("=", 1)
        sweeps[k] = [int(x) for x in v.split(",") if x]
    base = Config()
    for item in args.set:
        k, v = item.split("=", 1)
        setattr(base, k, int(v))

    peaks = load_peaks()
    print(f"peaks: BW {peaks.bw_gbps:.0f} GB/s | FP16 TC {peaks.fp16_tflops:.1f} TFLOPS | "
          f"平衡点 {peaks.balance_fp16:.0f} FLOP/B | launch {peaks.launch_us:.1f} µs")

    jobs = build_cases(args.case, sweeps, base)
    rows = []
    t_begin = time.time()

    with ClockSampler(interval=0.1) as sampler:
        for cfg, name, suffix in jobs:
            case: Case = CASES[name](cfg)
            case_rows = []
            for impl, fn in case.variants.items():
                if args.cudagraph:
                    st = do_bench_cudagraph(fn, rep=max(10, int(args.rep / 10)))
                else:
                    st = do_bench(fn, warmup=args.warmup, rep=args.rep)
                ev = analyze(st["median"], case.nbytes, case.nflops, peaks)
                case_rows.append({
                    "case": case.name + suffix,
                    "impl": impl,
                    "ms": round(ev["ms"], 4),
                    "p20": round(st["p20"], 4),
                    "p80": round(st["p80"], 4),
                    "gbps": round(ev["gbps"], 1),
                    "pct_bw": round(100 * ev["pct_bw"], 1),
                    "tflops": round(ev["tflops"], 2),
                    "pct_flop": round(100 * ev["pct_flop"], 1),
                    "ai": round(ev["ai"], 2),
                    "bound": ev["bound"],
                    "roofline_pct": round(100 * ev["roofline_eff"], 1),
                    "launch_pct": round(100 * n_launches(case, impl) * peaks.launch_us / 1e3 / ev["ms"], 1),
                    "err": "",
                    "note": case.note,
                })
            # 正确性检查在所有 variant 都跑完之后再做（否则会读到没写过的缓冲区）
            if case.check is not None and not args.no_check:
                c = case.check()
                ok = all(bool(v) for k, v in c.items() if k == "ok" or k.endswith("_ok"))
                err = "ok" if ok else f"FAIL {c}"
                for r in case_rows:
                    r["err"] = err
            rows.extend(case_rows)
            del case  # 及时释放显存，避免多个 case 同时驻留
            torch.cuda.empty_cache()

    # ------- 输出 -------
    keys = ["case", "impl", "ms", "gbps", "pct_bw", "tflops", "pct_flop", "ai", "bound", "roofline_pct", "launch_pct", "err"]
    print(tabulate([[r[k] for k in keys] for r in rows], headers=keys, tablefmt="github",
                   floatfmt=(".0f", ".0f", ".3f", ".0f", ".0f", ".2f", ".0f", ".1f", ".0f", ".0f", ".0f", ".0f")))
    print(f"\n语法：ms=中位延迟 | gbps/%BW=实际带宽及占实测峰值 | tflops/%TC=算力 | "
          f"ai=算术强度 FLOP/B | bound=roofline 判定 | roofline%=达理论上限比例 | launch%=launch 开销占比")
    print(sampler.format())
    print(f"总耗时 {time.time() - t_begin:.1f} s")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS.mkdir(exist_ok=True)
    path = args.json or RESULTS / f"{stamp}.json"
    path.write_text(json.dumps({"peaks": peaks.__dict__, "rows": rows}, indent=2, ensure_ascii=False))
    if not args.quiet:
        print(f"json -> {path}")
    if args.csv:
        new = not args.csv.exists()
        with args.csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"csv  -> {args.csv}")


if __name__ == "__main__":
    main()
