# cuda_learning

LLM 推理算子 / CUDA kernel 学习。**文档 + 可复现 bench**，每个结论都能用 `python -m bench.run` 重新测出来。

## 环境

| 项 | 值 |
|---|---|
| GPU | Tesla V100-PCIE-32GB（sm_70，80 SM，L2 6 MiB） |
| CUDA | 12.9（driver 580.178.04）/ torch 2.8.0+cu128 / triton 3.4.0 |
| Python | 3.11.16（`.venv`，uv 管理） |
| profiling | `nsys` 可用；`ncu` **不可用**（见下方坑） |

```bash
make install     # uv pip install -r requirements.txt
make calib       # 标定本机峰值 → bench/results/machine.json
make bench       # 跑全部 case
```

## 目录

| 路径 | 内容 |
|---|---|
| `01_inference_operators.md` | **算子全景**：13 类算子、Llama-3-8B 逐算子形状、融合算子名录、V100 能力边界、14 个手写练习 |
| `bench/` | 算子 bench 环境（标定 + 计时 + roofline 判定 + 对拍） |
| `bench/results/machine.json` | **本机实测峰值的唯一事实源**（所有 %BW / %TC 都除以它） |
| `bench/kernels/csrc/` | 手写 CUDA kernel（练习 1–3）：silu_mul / rmsnorm v1+v2 / fused_add_rmsnorm |
| `bench/results/baseline.csv` | 基线结果，用 `--csv` 追加 |

## 实测机器常数（不要用 spec 数字）

```
HBM copy      663 GB/s   (74% of 900)
FP16 TC        21.6 TFLOPS (19% of 113)
FP32            3.2 TFLOPS (23% of 14.1)
launch 开销    10.8 µs   (正常卡 2–4 µs)
平衡点 AI       32 FLOP/byte   (spec 是 124)
```

**这台机器的算力只有标称的 ~20%。** 用 `clock64` 反推，kernel 执行时 SM 实际约 340 MHz，
而 `nvidia-smi` 报 1380 MHz、功耗只有 54–98 W（远低于 250 W 上限）。显存侧是正常的 74%。
怀疑是平台侧限频或时间片（多租户）。结论：**所有 roofline 判断只用上表实测值**，
`bench/calib.py` 每次都会重新测并写进 `machine.json`，换机器/换时间跑一遍即可。

由此带来的实际影响（写代码时要按这个来想）：
- 平衡点 32 FLOP/byte → 比 spec 的 124 低得多，**更多算子在实测下是算力受限**；
- decode 的带宽地板：16 GB 权重 / 663 GB/s ≈ **24.1 ms/token**（Llama-3-8B fp16，即 41 tok/s 上限）；
- launch 10.8 µs × 每层 ~10 个 kernel × 32 层 ≈ **3.5 ms/step**，CUDA Graph 的收益被放大。

## 基线（2026-11-11，Llama-3-8B 形状）

`nbytes` 按**理想最小流量**记（融合后只需读一次写一次），所以 `%BW` 直接反映离上限多远。

| case | impl | ms | GB/s | %BW | TFLOP/s | %TC | AI | bound | roofline% |
|---|---|---|---|---|---|---|---|---|---|
| attn/decode | torch | 0.632 | 106 | 16% | 0.42 | 2% | 4.0 | mem | 16% |
| attn/prefill | torch | 9.970 | 27 | 4% | 3.45 | 16% | 128.0 | flop | 16% |
| elt/cast | torch | 0.090 | 279 | 42% | – | – | 0.0 | mem | 42% |
| elt/residual_add | torch | 0.044 | 572 | 86% | 0.10 | 0% | 0.2 | mem | 86% |
| elt/silu_mul | torch | 0.600 | 147 | 22% | 0.15 | 1% | 1.0 | mem | 22% |
| elt/silu_mul | **triton** | 0.150 | 585 | 88% | 0.59 | 3% | 1.0 | mem | 88% |
| elt/silu_mul | **cuda** | 0.131 | 672 | 101% | 0.67 | 3% | 1.0 | mem | 101% |
| gemm/down | torch | 0.413 | 285 | 43% | 0.28 | 1% | 1.0 | mem | 43% |
| gemm/gate_up | torch | 0.776 | 303 | 46% | 0.30 | 1% | 1.0 | mem | 46% |
| gemm/lm_head | torch | 3.548 | 296 | 45% | 0.30 | 1% | 1.0 | mem | 45% |
| gemm/o_proj | torch | 0.121 | 278 | 42% | 0.28 | 1% | 1.0 | mem | 42% |
| gemm/qkv | torch | 0.167 | 302 | 46% | 0.30 | 1% | 1.0 | mem | 46% |
| norm/fused_add_rmsnorm | torch | 0.500 | 67 | 10% | 0.03 | 0% | 0.5 | mem | 10% |
| norm/fused_add_rmsnorm | **cuda** | 0.066 | 512 | 77% | 0.26 | 1% | 0.5 | mem | 77% |
| norm/rmsnorm | torch | 0.451 | 37 | 6% | 0.04 | 0% | 1.0 | mem | 6% |
| norm/rmsnorm | **triton** | 0.042 | 400 | 60% | 0.40 | 2% | 1.0 | mem | 60% |
| norm/rmsnorm | cuda_v1 | 0.046 | 364 | 55% | 0.36 | 2% | 1.0 | mem | 55% |
| norm/rmsnorm | **cuda** | 0.037 | 455 | 69% | 0.46 | 2% | 1.0 | mem | 69% |
| rope/neox | torch | 0.277 | 76 | 11% | 0.06 | 0% | 0.8 | mem | 11% |

读数：
1. `elt/residual_add` 86% → 三操作数 elementwise 已经接近上限，bench 本身可信。
2. 同一个 silu_mul：torch 22% → triton 88% → 手写 CUDA **101%**（贴着峰值）。
   这说明"理想流量口径"是可达的，也说明瓶颈从来不是算法而是访存写法。
3. `gemm/*` 在 M=1 时 42–46% BW —— GEMV 在 V100 上的真实水平（cuBLAS 也不算好），
   练习 4 手写 GEMV 的目标就是超过它。
4. `norm/rmsnorm` 的 torch 版只有 6%：5 个 kernel 全耗在 launch 和小张量往返上；
   CUDA 版 69%（T=1024）/ 82%（T=4096），剩余差距见下一节。
5. `attn/*` 离 roofline 很远（预填充 16%），最值得手写。
6. 关注 `%TC` 一列全是 1–3%：decode 阶段所有大 GEMM 都是"算力几乎闲置、纯等权重"，
   这是量化/融合的全部动机。

M 扫描（`make sweep CASE='gemm/qkv'`）能直接看到 roofline 转折：

| M | ms | GB/s | %BW | TFLOP/s | %TC | AI | bound |
|---|---|---|---|---|---|---|---|
| 1 | 0.166 | 304 | 46% | 0.30 | 1% | 1.0 | mem |
| 8 | 0.183 | 276 | 42% | 2.20 | 10% | 8.0 | mem |
| 64 | 0.531 | 97 | 15% | 6.06 | 28% | 62.4 | flop |
| 512 | 1.819 | 33 | 5% | 14.17 | 66% | 423.7 | flop |
| 1024 | 2.916 | 24 | 4% | 17.67 | **82%** | 722.8 | flop |

## 练习 1–3（CUDA 版，已可跑）

`bench/kernels/csrc/` 里是三个手写 kernel，通过 `torch.utils.cpp_extension` JIT 编译
（`make build`，产物在 `bench/.build/`，增量重编约 5 s）：

| 文件 | kernel | 关键手法 |
|---|---|---|
| `elementwise.cu` | `silu_and_mul` | 16 B 向量化（float4 = 8 half）、2D grid 避免整数除法、2 读 1 写 |
| `norm.cu` | `rms_norm` v1 | 两遍读：读 → 归约 → 再读 → 写（**反面教材**，真实流量 1.5×） |
| `norm.cu` | `rms_norm` v2 | 寄存器缓存单遍 + warp shuffle 归约 + `__ldcs` |
| `norm.cu` | `fused_add_rms_norm` | 两输入两输出，顺手把 residual 写回，省一个 kernel |

同一笔流量下四个实现的对比（T=1024, H=4096，Llama-3-8B 形状）：

| kernel | torch 朴素 | triton | cuda v1 | cuda v2 |
|---|---|---|---|---|
| silu_mul | 22% BW | 88% | – | **101%** |
| rmsnorm | 6% | 60% | 55% | **69%** |
| fused_add_rmsnorm | 10% | – | – | **77%** |

**尺寸决定天花板**（这点比 kernel 写法更重要）：

| T | 流量 | 纯 copy | rmsnorm v2 | 说明 |
|---|---|---|---|---|
| 1024 | 16 MB | 77% | 69% | 16 MB 根本喂不饱 HBM，谁来都到不了 85% |
| 4096 | 64 MB | 93% | 82% | 验收应该在这个量级看 |
| 16384 | 256 MB | 96% | 74% | 大 T 下 `__stcs` 有效；剩余差距是 `__syncthreads` 气泡 |

所以：**练习 1 达标（101%）**；练习 2/3 到 69%/77%（T=1024）、82%（T=4096, rmsnorm），
离 85% 还差的是 barrier 气泡，不是向量化 —— 下一步是块内软件流水（把下一行的 load
提到本行归约之前），注释里写了思路。

```bash
make build                                   # 只编译 CUDA 扩展
python -m bench.run -c 'norm/*' --sweep T=1024,4096,16384
```

> 注：并行会话曾在 `bench/cuda/` 下写过一套同目标 kernel（silu_mul 64% / rmsnorm 47% BW，
> 比这里慢 2–2.5 倍），已删除；如需对照可从 commit `6519926` 取回。

## bench 的口径

`bench/run.py` 一行一个 (case, impl)，每个数字都是这么来的：

- **计时**：CUDA event，warmup 25 ms + 测量 100 ms，报 p50（同时存 p20/p80），
  每次迭代前清 64 MiB 缓冲区把 L2（6 MiB）顶掉 —— 否则 hidden=4096 只有 8 KB、
  会整个驻留 L2，测出"比 HBM 还快"的假象。
- **流量** `nbytes`：按融合后的理想最小流量记账，不是基线实际流量。
- **roofline%** = `max(t_mem, t_flop) / 实测ms`，即达理论上限的比例。
- **launch%** = `kernel数 × 10.8 µs / ms`，用来判断"这个数是不是被 launch 开销污染了"。
- **对拍**：所有 variant 跑完后统一和 fp32 参考比，容差 fp16 用 2e-2～5e-2（不是 1e-6）。
- `--cudagraph` 用 CUDA Graph 重测，`do_bench` 与它相减就是 launch 开销。

## 加自己的 kernel（三步）

照着 `bench/kernels/csrc/elementwise.cu` 抄结构：

```cuda
// 1) bench/kernels/csrc/my_gemv.cu：只写 kernel + launch 函数，入参全是裸指针
void my_gemv_launch(const void* x, const void* w, void* out, int M, int N, int K, cudaStream_t s);
// 2) 在 ops.h 声明，在 bindings.cpp 里加一个 pybind 入口（含 fp16/contiguous 检查）
// 3) 在 bench/kernels/cuda_ops.py 加一层薄 wrapper
// 4) 在 case 里注册，bench 会自动对拍 + 算 %BW
variants["mine"] = lambda: cuda_ops.my_gemv(a, w, out)
```

要点：**bench 反复调用同一个 fn，所以 fn 必须幂等**（不要 `x += y` 这类累积写，
会被累加几十次，正确性对拍会炸）；kernel 内部不要 allocate，缓冲区由 case 传进来。

新增 case 只需要在 `bench/cases/<family>.py` 里加一个 `@register("family/name")` 函数，
约定：**bench 反复调用同一个 fn，所以 fn 必须幂等**（不要 `x += y` 这类累积写，
会被累加几十次，正确性对拍会炸）。

## 坑（都踩过了）

| 现象 | 原因 / 处理 |
|---|---|
| `ncu` 报 `ERR_NVGPUCTRPERM` + `GPU is not supported` | 计数器要 root（`NVreg_RestrictProfilingToAdminUsers=0`）；且 2025.x 的 Nsight Compute 已不支持 Volta。用 `nsys`（`make prof`） |
| `torch.cuda.get_arch_list()` 里没有 sm_70 | 装到了 CUDA 13 wheel。CUDA 13 移除 Volta，必须留在 cu128（torch ≤2.8） |
| sdpa 在 decode 形状下慢得离谱 | V100 + `enable_gqa=True` 会退到 math 后端；`EFFICIENT_ATTENTION` 又要求 q/k/v 头数相同。基线改用 bmm 手写 GQA |
| transposed batched GEMM 比想象慢 5–10 倍 | `[8,4,1,128] @ [8,1,128,1024]` 这种 5 维 broadcast + 非连续 K，cuBLAS 会挑到很差的 kernel（334 µs vs 38 µs）。布局和 batch 维度要自己控制 |
| 小 kernel 全是 ~10 µs | 这台机器 launch 就是 11 µs，不是 kernel 慢。看 `launch%` 列 |
| BF16/TF32/FP8/`cp.async`/FA2/Marlin 不可用 | sm_70 硬限制，见 `01_inference_operators.md` §7.1 |

## 进度

- [x] 算子全景整理（`01_inference_operators.md`）
- [x] bench 环境：标定 + 计时 + roofline + 对拍 + 基线
- [x] 练习 1–3：CUDA 版 silu_mul（101% BW）、rmsnorm v1/v2（69%）、fused_add_rmsnorm（77%）
- [ ] 练习 2/3 收尾：软件流水吃掉 barrier 气泡（82% → 90%），T 扫描 + CUDA Graph 对比
- [ ] 练习 4–7：GEMV / GEMM tiling / Tensor Core / epilogue
- [ ] 练习 8–11：RoPE / online softmax / flash attention / paged decode
- [ ] 练习 12–14：sampling / 量化 / MoE
- [ ] 端到端：vLLM 基线 profiling（CUDA Graph on/off 对比）
