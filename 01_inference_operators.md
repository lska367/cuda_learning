# LLM 推理常用算子全景

> 环境基线：单卡 Tesla V100-PCIE-32GB（sm_70，HBM2 900 GB/s，FP16 TC ≈ 112 TFLOPS），CUDA 12.9
>
> ⚠️ **本机实测与 spec 不一致**（见 `bench/results/machine.json`）：copy 597 GB/s（66%）、
> FP16 TC 21.5 TFLOPS（19%）、launch 11 µs。`clock64` 反推 kernel 执行时 SM 仅 ~340 MHz，
> 怀疑平台限频/时间片。**本文所有推导用 spec 作教学口径，实际验收一律用 `bench` 的实测峰值。**
> 参照实现：本地源码 `~/work/distributed_llm_inference/vllm`（v0.23.1rc0），op 清单来自 `vllm/_custom_ops.py`（145 个自定义算子）
> 数字均为**量级估算**，用来判断 roofline 位置，不是 benchmark 结论。

---

## 0. 怎么用这份文档

1. 先读 §1（三个物理事实）——所有算子设计都是它的推论，不看这节后面全是背名词。
2. §2 是索引表，§3 是逐类详解（有空再深读）。
3. §4 用 Llama-3-8B 走一遍真实形状，把"算子名"变成"张量形状 + HBM 流量"。
4. §7 是动手路线（V100 能跑什么、不能跑什么，14 个练习）。

---

## 1. 先记住三个物理事实

### 1.1 prefill 是算力受限，decode 是**带宽受限**

一次 forward 的 FLOPs ≈ `2 × N_params × T`。但 HBM 访问量取决于你怎么切：

| 阶段 | 每步读权重 | 每步算力 | 算术强度 AI | 瓶颈 |
|---|---|---|---|---|
| prefill (T=1024) | 16 GB（一次） | 2×8e9×1024 ≈ 16 TFLOP | ≈ 1000 FLOP/B | **算力** |
| decode (T=1) | 16 GB（每 token！） | 16 GFLOP | ≈ 1 FLOP/B | **带宽** |

关键推导：decode 的 GEMM 是 `[B,H] @ [H,N]`，只读一次权重就完事，所以

```
AI = 2·B·H·N / (2·H·N) = B   (FLOP / byte，fp16 权重)
```

V100 的机器平衡点 = `112e12 / 900e9 ≈ 124 FLOP/byte`，所以**必须 batch≈124 才可能算力受限**（考虑 MFU 60%，实际 batch≈50–70 开始接近）。这就是为什么：

- decode 阶段的 GEMM 实质是 **GEMV**，优化目标是"把权重流干净"，不是"提高 Tensor Core 利用率"；
- 量化在 decode 上收益接近线性（权重减半 → 时间减半），在 prefill 上收益有限；
- continuous batching / 投机解码（一次验 k 个 token 把 B 抬高）本质都是**提高 AI**。

Llama-3-8B fp16 在 V100 上的天花板：`16e9 B / 900e9 B/s ≈ 17.8 ms/token → 56 tok/s`（100% 带宽，实际 60% 就是 33 tok/s）。

### 1.2 长上下文的瓶颈是 KV cache 带宽，不是 attention FLOPs

每 token decode 需读 `2 · S · kv_heads · head_dim · layers · 2B`：

| 上下文 S | KV 读取/token | 权重读取/token | KV 占比 |
|---|---|---|---|
| 4k | 0.54 GB | 16 GB | 3% |
| 8k | 1.07 GB | 16 GB | 6% |
| 128k | 17.2 GB | 16 GB | **52%** |

而 attention 的 FLOPs 即使 S=128k 也只占总量 ~6%（线性层是 `2·N·T`，attention 是 `2·S²·H`，8B 模型要到 S≈180k 才打平）。所以：**长上下文 = 访存问题 = paged/split-KV/sparse attention/GQA-MLA 的战场**。

### 1.3 三种数据流，对应三种优化手段

| 数据流 | 谁在用 | 优化手段 |
|---|---|---|
| 权重流（每步全读） | GEMM/GEMV | 量化（int4/fp8）、Marlin/Machete 重排、大 batch 摊薄 |
| KV cache 流（随 S 增长） | attention | GQA/MLA 压缩、KV 量化、page 管理、稀疏选择 |
| activation 流（层间） | 逐元素/归一化/激活 | **算子融合**（减少 HBM 往返）、fp8 激活、CUDA Graph |

举个融合的账：decode 时一个 token 的 hidden 只有 4096×2 = 8 KB（小到装进一个 CTA 的 shared memory），但朴素实现要 "RMSNorm 写 8KB → GEMM 读 8KB" 反复上十几次。**decode 的低效大量来自这些小张量的 HBM 往返和 kernel launch，而不是算得快不快**。每层约 10 个 kernel，32 层 = 320 次 launch，5 µs/次 ≈ 1.6 ms，放在 30 ms 的 step 里就是 5%——这就是 CUDA Graph 的来源。

---

## 2. 算子总表（13 类）

| # | 类别 | 代表算子 | 复杂度/性质 | 典型瓶颈 |
|---|---|---|---|---|
| A | 线性层 GEMM 家族 | GEMM、GEMV、Batched/Grouped GEMM、fused QKV/gate_up | 算力+带宽 | 取决于 M |
| B | 量化/反量化 | int4/fp8 per-token/per-group quant、dequant-GEMV、scaled_mm | 带宽 | 权重流 |
| C | 归一化 | RMSNorm、LayerNorm、fused_add_rmsnorm、QK-Norm | 带宽 | 读+写 2 遍 |
| D | 激活/门控 | SiLU、GELU(tanh)、SwiGLU(fused silu_and_mul)、GeGLU | 带宽 | 读写比 |
| E | 位置编码 | RoPE（GPT-J / NeoX 布局）、YaRN/NTK 缩放、ALiBi | 带宽 | 索引计算 |
| F | Attention | prefill FMHA、decode paged/split-KV、MLA、稀疏索引 | 算力(S²)+带宽 | 见 §1.2 |
| G | KV cache 管理 | reshape_and_cache、copy_blocks、swap_blocks、concat_and_cache_mla | 带宽+scatter | 非合并访存 |
| H | 采样/logits | argmax、top-k、top-p、penalty、RNG、logprobs | 延迟敏感 | 排序/多 pass |
| I | MoE | router top-k、align_block_size、permute/unpermute、grouped GEMM、moe_sum | 混合 | 小 M + 不规则 |
| J | 通信 | AllReduce/AllGather/ReduceScatter/All2All、custom AR、fused AR+RMSNorm | 带宽+延迟 | 消息大小 |
| K | 逐元素/搬运 | embedding gather、lm_head、residual、cast、transpose、cat/split | 带宽 | 访存模式 |
| L | 投机解码 | tree attention mask、verify/accept、rejection sampling | 延迟 | 分支/mask |
| M | 线性注意力/SSM | selective_scan、causal_conv1d、chunk gated delta rule | 算力+状态 | 递归依赖 |

---

## 3. 逐类详解

### A. 线性层：GEMM 家族

| 算子 | 何时出现 | 备注 |
|---|---|---|
| 普通 GEMM | prefill 的全部线性层 | cuBLASLt / CUTLASS；epilogue 可融 bias/act |
| GEMV / skinny GEMM | decode，M=1~8 | cuBLAS 的 gemv 对小 M 不佳，常手写或 cublasLt 调优 |
| Batched GEMM | 多 LoRA、多序列独立矩阵 | 每 batch 一个独立 GEMM |
| Grouped GEMM | MoE（每个 expert 一个 GEMM，共享 grid） | CUTLASS Grouped GEMM，`cutlass_moe_mm` |
| Fused QKV proj | 每层 attention 前 | 3 个权重拼成 `[H, H+2·kv_h·d]`，一次 GEMM |
| Fused gate_up | 每层 FFN 前 | `[H, 2I]`，省一次 launch 和一次 activation 读写 |

**为什么"融合"能省**：Q/K/V 分开做是 3 次读同一份 hidden（各 8 KB），拼起来只读一次；但输出要 split，所以融合的收益在**输入侧**。

**decode 的 skinny GEMM 专门技巧**：vLLM 里有 `dsv3_router_gemm`、`fp32_router_gemm`（MoE router 的 M 极小）、`LLMM1`、AMD 的 `wvSplitK`（按 K 切分并行，用更多 CTA 喂满 SM）。V100 只有 80 个 SM，M=1 时先看 grid 能不能填满 80。

### B. 量化 / 反量化算子

| 方向 | 算子 | 说明 |
|---|---|---|
| Weight-only W4A16 | GPTQ (`gptq_gemm`)、AWQ (`awq_gemm`)、Marlin (`marlin_gemm`)、Machete (`machete_mm`)、WNA16 (`moe_wna16_gemm`) | 权重量化+激活 fp16；**decode 收益接近线性** |
| W8A8 int8 | `scaled_int8_quant` + `cutlass_scaled_mm` | SmoothQuant 类 |
| FP8 | `scaled_fp8_quant`（per-tensor / per-token / per-block）+ `cutlass_scaled_mm` | Hopper/Ada 原生；V100 只能模拟 |
| FP4 (MXFP4/NVFP4) | `scaled_fp4_quant`、`cutlass_scaled_fp4_mm`、`qutlass` | Blackwell |
| 权重重排（prepack） | `gptq_marlin_repack`、`awq_marlin_repack`、`machete_prepack_B`、`cutlass_encode_and_reorder_int4b` | 一次性离线：把 int4 重排成 Tensor Core 友好的交错布局 |
| 融合量化 | `rms_norm_dynamic_per_token_quant`、`silu_and_mul_per_block_quant`、`fusedQuantizeMx` | 归一化/激活直接输出量化张量，省一次全精度往返 |

**学习要点**：int4 的 Tensor Core 没有原生 int4 mma（除了 Blackwell 的 fp4），所以 Marlin 的核心 trick 是**用 fp16 的 mma 喂 int4 数据**：把 int4 解到 fp16 寄存器后立刻喂进 mma，避开 HBM 往返；再加"异步权重流 + 交错布局解决 bank conflict"。相关 op 名 `marlin_int4_fp8_preprocess` 说明它先把 int4 的零点/scale 预处理成 fp8。

**V100 注意**：Marlin/Machete 需要 sm_80+/sm_90，V100 **跑不了**。老的 GPTQ/AWQ CUDA kernel（dp4a）和 Triton 反量化在 sm_70 可以跑。

### C. 归一化

| 算子 | 说明 |
|---|---|
| `rms_norm` | Llama/Qwen/Mistral 主线；只减平方均值，无 mean shift |
| `fused_add_rms_norm` | residual add + RMSNorm 一次搞定（两输入两输出），每层两侧各一次 |
| LayerNorm | BERT/GPT-2/部分 VLM |
| QK-Norm | per-head 的 RMSNorm，Qwen3/部分新模型，`fused_qk_norm_rope` 把 QK-norm+RoPE 合在一个 kernel |
| 变体 | Gemma 的 `(1+w)` 权重偏移；minimax 的 reduce-rms（跨 TP allreduce 融合进 norm） |

**kernel 形态**：一行（一个 token）一个 warp/CTA，warp shuffle 归约，`float4` 向量化 load。hidden=4096 fp16 = 8 KB/token，纯带宽活。融合 add 的价值：省掉独立的 `x = x + residual` 读-改-写（16 KB 流量）。

### D. 激活与门控

| 算子 | 用在哪 |
|---|---|
| `silu_and_mul` (SwiGLU) | Llama/Qwen/Mistral FFN，`silu(gate) * up` |
| `gelu_and_mul` (GeGLU) | Gemma、部分模型 |
| GELU (tanh approx) / quick_gelu | BERT 系、CLIP |
| ReLU² | Gemma-2 FFN |
| `silu_and_mul_*_quant` | 激活后直接量化，融进 FFN 的中间张量（否则 2I=28672 的中间激活要写一次读一次，decode 时这是 57 KB/token，比 hidden 大 7 倍） |

**关键**：FFN 的中间激活 `[B, 2I]` 是 transformer 里最宽的临时张量，**fused silu_and_mul 是收益最高的 elementwise 融合**（省 2×B×I×2B 的 HBM 流量）。

### E. 位置编码 RoPE

| 变体 | 说明 |
|---|---|
| GPT-J 布局（interleaved） | `(x[2i], x[2i+1])` 成对旋转 |
| NeoX 布局（half-split） | 前一半和后一半配对，Llama 系用这个 → **写 kernel 最容易搞错的第一名** |
| 缩放 | linear scaling、dynamic NTK、YaRN（`yarn_scaling_rope.py`）、Llama-3 rope scaling、LongRoPE、mrope（Qwen-VL 多模态 3 轴） |
| 融合 | `fused_qk_norm_rope`、`fused_minimax_m3_qknorm_rope_kv_insert`（RoPE + 写进 paged KV cache 一次完成） |

**实现要点**：cos/sin 表**预计算**成 `[max_pos, head_dim/2]`（fp32），kernel 里只做 gather；`position_ids` 是 int64 要 cast；GQA 下 K 只对 `kv_heads` 做。融合 KV 写入的收益：否则 K 要写回 HBM 再被 `reshape_and_cache` 读一遍。

### F. Attention（最值得深挖的一类）

**prefill（FMHA）**：分块 + online softmax，永远不物化 `[S,S]` 注意力矩阵。

```
for kv_block in KV:
    S = Q_tile @ K_block^T * scale          # mma
    m_new = max(m, rowmax(S))
    P = exp(S - m_new)                       # 只在寄存器/SMEM
    l = l * exp(m - m_new) + rowsum(P)
    O = O * exp(m - m_new) + P @ V_block     # mma
O /= l
```

要点：running max/sum 的 rescale、causal 跳过整块 if `kv_block > q_block`、对角块单独处理、GQA 里一组 Q head 共享 K/V tile（提高 SMEM 复用）、fp32 累加。

**decode（paged / split-KV）**：Q 只有一行，并行度只来自 `heads × kv_blocks`。

| 后端 | 切分方式 | 二次归约 |
|---|---|---|
| `paged_attention_v1` | (head, seq_group) | 无 |
| `paged_attention_v2` / flash-decoding | + 按 KV 长度切 partition | 需要：`merge_attn_states`（partial max/sum/output 合并） |
| FlashInfer decode | 类似 split-KV，带 plan | 有 |
| FlashMLA / `flash_mla_with_kvcache` | MLA：Q 侧矩阵吸收后等价 MQA，两阶段（`get_flash_mla_metadata`） | 有 |
| cascade attention | 共享 prefix（同 system prompt）只算一次 | `merge_attn_states` |

**mask / bias 家族**（都体现在 kernel 的 skip 逻辑里，写错就静默错）：
causal、sliding window、document mask（packed 多文档）、ALiBi、softcap（Gemma-2 的 `tanh` 截断）、tree mask（投机解码）、prefix-LM mask。

**稀疏 / 变体**：`top_k_per_row_prefill/decode` + `indexer_k_quant_and_cache`（DeepSeek 稀疏注意力的 indexer）；`chunked_local_attention`；linear attention（`linear_attn.py`）。

**KV 精度**：`reshape_and_cache` 支持 fp8 KV → `convert_fp8`、`gather_and_maybe_dequant_cache`。

### G. KV cache 管理算子

| 算子 | 作用 |
|---|---|
| `reshape_and_cache` / `_flash` | 把新 token 的 K/V `[T, kv_h, d]` 按 `slot_mapping` **scatter** 进 paged cache `[num_blocks, block_size, kv_h, d]`；NHD vs HND 布局两种 |
| `copy_blocks` | prefix caching 共享物理块、beam search 分叉（引用计数在 CPU 侧） |
| `swap_blocks` | GPU↔CPU 换出/换入（显存不够时） |
| `concat_and_cache_mla` | MLA 只存压缩 latent（576 = 512 + 64 rope） |
| `cp_gather_cache` / `cp_gather_and_upconvert_fp8_kv_cache` | context parallel 下按 rank 取 KV |
| `convert_fp8` | KV 量化/反量化批量转换 |

**性质**：这是**纯 scatter + 非合并写**的算子，性能瓶颈在写事务数而不是带宽。`slot_mapping` 必须是 per-token 一个 int64，且 block 内 offset 决定相邻 token 是否连续——所以 block_size 通常取 16（让一个 warp 能连续写 16×d 个元素）。**它是 decode 前最容易被低估的算子**（每个 token 每层都要做）。

### H. 采样与 logits 处理

| 阶段 | 算子 |
|---|---|
| 取 logits | lm_head GEMM（`[B,H]@[H,V]`，decode 时读 V×H×2B = 1.05 GB，占 step 时间 ~6%）；TP 下 vocab 分片 + allgather |
| 惩罚 | repetition / presence / frequency penalty（`apply_repetition_penalties_cuda`，需 gather 历史 token）、logit_bias、min_tokens |
| 温度/缩放 | `logits / T`、softcap |
| 截断 | top-k（radix select / bitonic sort）、top-p（排序+prefix sum+找截断点）、min-p、typical |
| 采样 | `cumsum` + `searchsorted` + multinomial；greedy = `argmax` |
| RNG | Philox/curand，per-request seed + position offset → **可复现性**（`VLLM_BATCH_INVARIANT` 就是为这个） |
| 词表并行 | `all_gather` 或者只在最后 rank 采样（vLLM 的 `from_topk` 优化） |
| 结构化输出 | grammar/xgrammar 生成的 token bitmask 在采样前 mask 掉 |

**性质**：延迟敏感 + 多 pass + host-device 同步点。top-p 在长 vocab（128k）上是显存带宽+排序混合；业界会用"先 top-k 再 top-p"降复杂度。

### I. MoE

| 步骤 | 算子 | 说明 |
|---|---|---|
| 1. router | `topk_softmax` / `topk_sigmoid` / `grouped_topk`，`dsv3_router_gemm`、`fp32_router_gemm` | M 极小（=token 数）、N=E（128~256），需专门 skinny GEMM；DeepSeek 用 sigmoid + group-limited greedy + expert bias |
| 2. 对齐 | `moe_align_block_size`、`batched_moe_align_block_size` | 按 expert 排序 token，**pad 到 block 对齐**（grouped GEMM 的 tile 需要）；用 counting sort，输出 `sorted_ids / expert_ids / num_tokens_past_padded` |
| 3. 重排 | `shuffle_rows`、`moe_permute_unpermute` | 收集 token 到 expert 连续段 + 记录逆索引 |
| 4. grouped GEMM | `cutlass_moe_mm`、`cutlass_fp4_moe_mm`、`gptq_marlin_moe` | w1(gate_up) → act·mul → w2(down) |
| 5. 归约 | `moe_sum`、`topk_weight_and_reduce` | top-k 专家的加权累加（scatter-add，有 atomic / 排序后规约两版） |
| EP 通信 | all2all dispatch/combine（DeepEP，NVSHMEM/IBGDA） | 专家并行时 token 跨卡 |

**融合视野**：`fused_experts`（Triton）把 permute → GEMM1 → silu_and_mul → GEMM2 → unpermute → 加权和 合成一次 launch；`moe_fused_mul_sum` 是更细的融合。**MoE 的痛点是 M 小且不规整，SM 填不满**，所以 3 个 expert 的权重按 8 个 token/块的小 tile 处理。

### J. 通信算子

| 并行 | 集合通信 | 出现位置 |
|---|---|---|
| TP | AllReduce | 每个 decoder layer 的 attention out + FFN out 各一次（列并行后、行并行后） |
| TP | （无） | QKV 列并行（split 输出）、O/FFN 行并行（split 权重） |
| PP | send/recv（P2P） | micro-batch 流水 |
| DP | AllGather / AllReduce | logits、embedding 词表 |
| EP | All2All | MoE dispatch/combine |
| CP | AllGather / Ring P2P | 长序列 KV 分片 |

| 优化算子 | 说明 |
|---|---|
| `custom_all_reduce` | 小消息（<256 KB~1 MB）走 one-shot（每卡写所有对端），大消息 two-shot（reduce-scatter + all-gather）；需 NVLink/P2P |
| `quickreduce` | AMD 版 |
| **fused AllReduce + RMSNorm** | `minimax_reduce_rms_kernel.h`、`fused_allreduce_gemma_rms_norm.py`：用 reduce-scatter 的部分和顺便算 RMSNorm，省一次全量 AllReduce |
| 异步/重叠 | dual-batch overlap、`all2all_utils` 的异步 prepare/finalize |

**V100 单卡学习者**：PCIe 版无 NVLink → custom allreduce 不可用。用 2 进程 + gloo/NCCL 观察语义就够；理解"一个 AllReduce ≈ 每个 rank 读写 `2(n-1)/n · S` 字节"即可。

### K. 逐元素 / 搬运 / 索引

| 算子 | 备注 |
|---|---|
| embedding lookup | 词表按 TP 切分 → token id 不在本 rank 就 mask 成 0，再 AllReduce；本质 gather，随机但 cache-friendly |
| lm_head | 见 §H，decode 时是**单个体量第二大的权重流** |
| residual add | 常被融进 layernorm |
| cast（fp16↔bf16↔fp32） | 别小看，`type_convert.cuh`；一次多余 cast = 双倍流量 |
| transpose / contiguous | 最典型的**性能陷阱**：attention 前的 `[B,S,H] → [B,heads,S,hd]` 若产生实际拷贝，代价等同一次全层读写 |
| cat / split | QKV 融合后的配套操作 |
| 元数据构造（host 侧） | `slot_mapping`、`seq_lens`、`block_tables`、`cu_seqlens`、`query_start_loc` — **CPU-GPU 同步 + 小拷贝**，用 pinned memory + 批量 memcpy + CUDA Graph 消除 |

### L. 投机解码

| 算子 | 说明 |
|---|---|
| draft 采样 | 小模型自回归 k 步，或 EAGLE/MTP 的多头并行 |
| tree/多候选 attention mask | 一次 forward 验证 k 个候选 token，用树形 mask 让候选互不可见 |
| verify / accept | greedy：`argmax` 逐位比较 + `cumsum` 求最长匹配；sampling：rejection sampling（`rejection_sampler.py`）保证分布等价 |
| KV 回收 | 被拒绝 token 的 KV 要 truncate（改 seq_len，不真正删块） |

**本质**：用 `B=k+1` 的 batch 提高 §1.1 里的算术强度，代价是 mask 复杂度和接受率。

### M. 线性注意力 / SSM / Conv（新架构）

| 算子 | 用在哪 |
|---|---|
| `selective_scan_fwd` | Mamba/Mamba-2 |
| `causal_conv1d_fwd` / `_update` / `_weight_pack` | Mamba 的 short conv（decode 时是"滑动窗口更新"） |
| `chunk_gated_delta_rule`、`fused_sigmoid_gating_delta_rule_update` | GDN（Qwen3-Next 等线性注意力） |
| lightning_attn / FLA | 线性注意力核 |

**和标准 attention 的区别**：有**递归状态**（recurrent state），decode 只更新状态不扫历史 → 复杂度 O(1)/token；代价是 kernel 有串行依赖，难以并行，prefill 需要 chunked 扫描技巧（chunk 内并行 + chunk 间串行）。

---

## 4. 走一遍：Llama-3-8B 的算子序列

符号：H=4096，L=32，heads=32，kv_heads=8，hd=128，I=14336，V=128256。每层权重 218M × 2 bytes = **436 MB**。

| # | 算子 | decode（M=1~B） | prefill（M=1024） | 性质 |
|---|---|---|---|---|
| 1 | RMSNorm | 1×4096 | 1024×4096 | 带宽 |
| 2 | Fused QKV GEMM | [B,4096]·[4096,**6144**] | 同 | 25.2M 权重 |
| 3 | RoPE (+cache insert) | [B,32,128] | 同 | 索引/带宽 |
| 4 | reshape_and_cache | scatter B tokens | 1024 tokens | scatter |
| 5 | Attention | QK^T: [B,32,128]×paged KV | S² tile，online softmax | decode 带宽 / prefill 算力 |
| 6 | O GEMM | [B,4096]·[4096,4096] | 同 | 16.8M |
| 7 | residual add | B×4096 | 同 | 带宽 |
| 8 | RMSNorm | | | |
| 9 | Fused gate_up GEMM | [B,4096]·[4096,**28672**] | 同 | **117.4M（最大）** |
| 10 | silu_and_mul | [B,14336] | 同 | 带宽 |
| 11 | down GEMM | [B,14336]·[14336,4096] | 同 | 58.7M |
| 12 | residual add | | | |
| → | **× 32 层** | | | **13.96 GB 权重** |
| 13 | Final RMSNorm | | | |
| 14 | lm_head GEMM | [B,4096]·[4096,**128256**] | 同 | 525M 权重，1.05 GB |
| 15 | Sampler | [B,128256] | 同 | 延迟 |

**读这张表的三个结论**：
1. 权重流的 3 大头：`gate_up`(117M) > `down`(59M) > `qkv`(25M)，FFN 占每层 81%。→ 量化优先动 FFN。
2. token embedding + lm_head 合计 1.05 GB（6.5%），未 tie 权重时它是第三大流量。
3. 每层只有 4 个"大"GEMM，其余 8 个小算子全是带宽/延迟活 → **融合与 graph 优化的空间在这里**。

---

## 5. 业界融合算子名录（对着名字找论文/源码）

| 名字 | 融了什么 | 关联算子类别 |
|---|---|---|
| FlashAttention 1/2/3 | prefill attention + softmax + mask（v2 去掉了中间 O rescale 写回；v3 用 Hopper TMA+wgmma 做 warp-specialization） | F |
| FlashInfer | decode/prefill attention + plan（cascade、split-KV、可组合 mask） | F |
| PagedAttention (vLLM) | attention + paged KV 寻址 | F, G |
| FlashDecoding | decode 按 KV 切分 + 二阶段合并 | F |
| FlashMLA | MLA decode 的矩阵吸收 + 两阶段 | F |
| xFormers `memory_efficient_attention` | FA1 系；**V100 可用的 attention 后端** | F |
| Marlin / Machete | int4 权重流 + mma（Ampere / Hopper） | A, B |
| ExLlamaV2 kernels | GPTQ GEMV，专为小 batch 调优 | A, B |
| DeepGEMM | fp8 block-wise GEMM（含 MoE grouped 版） | A, B |
| CUTLASS / CuTe DSL | 手写 GEMM 的模板底座（epilogue fusion、grouped） | A |
| TRT-LLM | XQA（decode attention）、GPT attention plugin、MoE plugin、weight-only plugin | 全家桶 |
| Liger-Kernel / Unsloth | fused cross-entropy、RMSNorm、RoPE、SwiGLU（偏训练，思路通用） | C, D, E |
| Triton / TileLang | 写这些 kernel 的 DSL | — |
| flash-linear-attention (FLA) | GDN/Mamba 等线性注意力 | M |

---

## 6. 在本地源码里怎么顺着找

根目录：`~/work/distributed_llm_inference/vllm`

| 想看的算子 | 去哪 |
|---|---|
| **全部自定义 op 的签名** | `vllm/_custom_ops.py`（145 个 `def`，最权威的清单）+ `csrc/torch_bindings.cpp` |
| attention 后端注册 | `vllm/v1/attention/backends/registry.py`（flash_attn / flashinfer / triton_attn / rocm / mla / gdn / mamba） |
| paged attention CUDA 实现 | `csrc/attention/` |
| 归一化 / 激活 | `vllm/model_executor/layers/layernorm.py`、`activation.py`（前面调 `_custom_ops`，真正的 kernel 在 `vllm-flash-attn` / `vllm/_C`） |
| 线性层与并行切分 | `vllm/model_executor/layers/linear.py`（ColumnParallel / RowParallel / QKVParallel / MergedColumn） |
| RoPE 全部变体 | `vllm/model_executor/layers/rotary_embedding/`（21 个文件，按缩放类型分） |
| MoE | `layers/fused_moe/`（`fused_moe.py`、`moe_align_block_size.py`、`experts/`、`router/`、`prepare_finalize/`） |
| 量化 | `layers/quantization/`（`fp8.py`、`mxfp4.py`、`awq_triton.py`、`compressed_tensors/`…）+ `csrc/quantization/` |
| 采样 | `vllm/v1/sample/sampler.py`、`ops/`、`logits_processor/`、`rejection_sampler.py` |
| 通信 | `csrc/custom_all_reduce.cuh`、`csrc/minimax_reduce_rms_kernel.h`、`layers/fused_allreduce_gemma_rms_norm.py` |

**方法**：不要从 `csrc` 往下读。从 `model/llama.py` 的 `forward` 出发，逐行问"这行调了哪个 layer → 那个 layer 的 `forward` 调了哪个 `_custom_ops` → 那个 op 在 `csrc` 的哪个 kernel"。走通一层，32 层就是重复。

---

## 7. V100 学习路线

### 7.1 能力边界（先认清，省得照着 Hopper 论文写代码跑不起来）

| 特性 | V100 (sm_70) | 影响 |
|---|---|---|
| FP16 Tensor Core | ✅ 112 TFLOPS（`mma.sync` m8n8k4 / `wmma` 16×16×16） | 主线精度用 fp16 + fp32 累加 |
| BF16 | ❌ 无 TC 支持 | bf16 只能存，计算会退化 → 学习用 fp16 |
| TF32 | ❌（Ampere+） | fp32 GEMM 没有捷径，慢 4 倍 |
| FP8 / FP4 | ❌（Hopper / Blackwell） | 量化只能做 int8/int4 模拟 |
| `cp.async` | ❌（sm_80+） | 双缓冲只能 ldg→寄存器→st.shared |
| `wgmma` / TMA / cluster / DSMEM | ❌（sm_90+） | 用不了 warp-specialization、异步 bulk copy |
| Marlin / Machete / FA2 / FA3 / XQA | ❌ | 只能用 PagedAttention v1/v2、xformers mem-efficient、Triton、GPTQ/AWQ 老 kernel |
| NVLink | ❌（PCIe 版） | custom allreduce 用不了，通信学习靠 2 进程 gloo/NCCL |

### 7.2 14 个练习（按依赖顺序，每个都有验收标准）

先立基线：`vllm` 跑通 Llama-3-8B fp16（32 GB 刚好），用 `torch.profiler` / `nsys` 抓一张 per-kernel 时间表——**亲眼看到 GEMM 占 85%+、小 kernel 占 launch 数的 80%**，后面所有优化才有锚点。

| # | 练习 | 关键点 | 验收 |
|---|---|---|---|
| 1 | elementwise：`silu_and_mul` | grid-stride、`float4`/`half2` 向量化、读写比 2:1 | ≥ 90% 峰值带宽 |
| 2 | reduction：RMSNorm | warp shuffle + block reduce、`float4` load、一个 warp 处理一行 | ≥ 85% 带宽；对齐/非对齐两种 hidden |
| 3 | fused add + RMSNorm | 省一次读-改-写；两输入两输出 | 对比朴素两 kernel，看流量降 1/3 |
| 4 | GEMV：`[1,4096]@[4096,4096]` | 一个 CTA 一个输出 tile；shared memory 暂存 x；理解为什么 cuBLAS 的 gemv 在这里不够好 | ≥ 80% 带宽；对拍 `torch.matmul` |
| 5 | GEMM tiling（CUDA Core） | SMEM tile + register tile + 边界处理；用 nsys 看 occupancy | 达 cuBLAS 的 30% |
| 6 | GEMM + Tensor Core（`wmma` 16×16×16 或 `mma.sync` m8n8k4） | fragment 布局、`ldmatrix`（sm_75+，V100 没有 → 手工 swizzle 避免 bank conflict）、fp32 累加 | 达 cuBLAS 的 60% |
| 7 | GEMM epilogue 融合 | bias + 激活 + per-token 量化输出 | 比"GEMM 后接独立 kernel"快 ≥ 10% |
| 8 | RoPE（两种布局）+ fused KV 写入 | NeoX half-split vs GPT-J interleaved；cos/sin 预计算；按 `slot_mapping` scatter 进 paged layout | 与 HF 对拍（含 YaRN 缩放） |
| 9 | online softmax（独立） | running max/sum + 无中间矩阵 | 与 `torch.softmax` 误差 < 1e-5 |
| 10 | causal flash attention（prefill, Triton 或 CUDA） | §F 的 core loop、对角块 mask、GQA 共享 K/V tile | 显存 O(S) 而非 O(S²)；达 xformers 的 70–80% |
| 11 | paged decode attention + split-KV 合并 | `block_tables` 间接寻址、per-partition max/sum 合并（对照 `merge_attn_states`） | 与 §10 对拍；S=8k 且 batch=1 时延迟下降可见 |
| 12 | sampling：top-k / top-p / multinomial | radix-select 或 bitonic；cumsum + searchsorted；CPU/GPU 边界 | 与 HF `LogitsProcessor` 分布对拍 |
| 13 | int4 反量化 GEMV（GPTQ 风格 group=128）+ fp8 per-token 缩放 | 解包 nibble、scale/zero 处理、per-group 广播 | 正确性对拍 fp16；decode 速度接近 2× |
| 14 | MoE 小闭环：align_block_size → grouped GEMM → moe_sum | counting sort、pad 到 block、逆索引 unpermute | 与"逐 expert for 循环"结果一致 |

**可选进阶**：手写 ring AllReduce（2 进程 + gloo），画 latency vs bandwidth 区制曲线 → 理解为什么小消息要 one-shot。

### 7.3 工具与参考文献

- **profiling**：`ncu`（`--set full` 看 SM/Memory 利用率、bank conflict、stall 原因）、`nsys`（kernel 时间线 + launch 开销）、Triton `proton`
- **教程**：siboehm《How to Optimize a CUDA Matmul Kernel》（从 naive 到 cuBLAS 的 6 步，最值得抄一遍）、Triton 官方 tutorials（01–10 里有 softmax、fused attention、layer norm、matmul）、GPU MODE / CUDA-MODE lectures、`LeetGPU`
- **论文导读顺序**：PagedAttention(vLLM) → FlashAttention 1/2 → FlashDecoding → GQA → MLA(DeepSeek-V2) → Marlin → DeepGEMM → DeepEP
- **系统性材料**：`jax-ml/scaling-book`（通信算子的 roofline 讲得最清楚，TPU 但通用）、CUTLASS 官方 tutorial

---

## 8. 自测清单

做完上面，应该能不看文档回答：

1. 为什么 decode 优化"少读一个字节"就等于"快一倍"，而 prefill 不是？
2. `fused_add_rms_norm` 相比两个独立 kernel 省了哪些 HBM 事务？估算 hidden=4096 fp16 下的字节数。
3. 为什么 MoE 需要 `align_block_size`？不 pad 会发生什么？
4. PyTorch 里 `x.transpose(1,2)` 后接 GEMM 为什么可能比显式 `contiguous()` 慢（或者反过来）？什么时候会触发真实拷贝？
5. 为什么 attention kernel 内部要用 fp32 累加，但 Q/K/V 用 fp16 存？
6. RoPE 的 NeoX 布局和 GPT-J 布局差在哪？混用会是什么现象？
7. split-KV decode attention 为什么需要二次归约，合并时要用什么（不是简单相加）？
8. top-p 采样为什么复杂度比 top-k 高？长 vocab 下怎么降？
9. 为什么 MLA 要把 KV 压成 576 维 latent，这对 decode 的带宽省了多少？
10. V100 上没有 `cp.async`，GEMM 双缓冲怎么写？和 Ampere 的 prologue 差在哪？
