# KDA linear attention（34 / 45 層）

`Glm5NextLinearAttention`（`models/glm5_next.py`）。KDA = Kimi Delta Attention
風格的 **gated delta rule** linear attention。

```
linear_attn_config = {
    num_heads: 64, head_dim: 128, short_conv_kernel_size: 4,
    gate_lower_bound: -5.0,
    kda_layers:      [i for i in range(45) if i % 4 != 3],   # 34 層
    full_attn_layers: [3, 7, 11, ..., 43],                   # 11 層
}
```

`projection_size = 64 × 128 = 8192`，`local_num_heads = 64 / attn_tp_size`。

最重要的性質：**state 大小與 sequence length 無關**。34 層完全不佔 KV cache，
長 context 的記憶體壓力只來自那 11 層 MLA。

## Projection 清單

| 權重 | 全域 shape | TP4 每 rank | 作用 |
|---|---|---|---|
| `qkv_proj` | `[4096, 3×8192]` | `[4096, 3×2048]` | q, k, v，各 64 head × 128 |
| `b_proj` | `[4096, 64]` | `[4096, 16]` | `beta` logits，每 head 一個 |
| `f_a_proj` | `[4096, 128]` | **replicated** | forget gate 的 low-rank A |
| `f_b_proj` | `[128, 8192]` | `[128, 2048]` | forget gate 的 low-rank B |
| `g_a_proj` | `[4096, 128]` | **replicated** | output gate 的 low-rank A |
| `g_b_proj` | `[128, 8192]` | `[128, 2048]` | output gate 的 low-rank B |
| `dt_bias` | `[8192]` fp32 | `[2048]` | 每 (head, K-channel) 的 gate bias |
| `A_log` | `[1,1,64,1]` fp32 | `[1,1,16,1]` | 每 head 的 log decay scale |
| `qkv_conv1d` | depthwise conv，kernel 4，3×8192 channel | 3×2048 | q/k/v 的 short causal conv |
| `o_norm` | `FusedRMSNormGated(128, activation="sigmoid")` | per-head | 用 g 當 gate 的 RMSNorm |
| `o_proj` | `[8192, 4096]` RowParallel | `[2048, 4096]` | 輸出投影 |

`f_a_proj` / `g_a_proj` 是 `ReplicatedLinear` 而不是 column-parallel：low-rank
bottleneck 只有 **128 維、所有 head 共用**，head-specific 的展開在
`f_b_proj` / `g_b_proj` 才發生。複製這 128 維可以省掉一次 all-gather，
也讓 fused GEMM 的 layout 更簡單。

### Fused 路徑

`Glm5NextLinearAttention._can_fuse_proj` 決定 layout，由寬到窄三種：

| 條件 | in-proj 的 module | in-proj GEMM | 加 `o_proj` |
|---|---|---|---|
| `do_fuse_qkvbfg`：q/k/v/b/f/g 全部可 fuse | `fused_qkvbfg_a_proj`（q, k, v, b, f_a, g_a）+ `fused_fg_b_proj`（f_b, g_b） | 2 | 3 |
| `fuse_bfg`：qkv 不行，但 b/f/g 可以 | `qkv_proj` + `fused_bfg_a_proj`（b, f_a, g_a）+ `fused_fg_b_proj` | 3 | 4 |
| 都不行 | `qkv_proj`、`b_proj`、`f_a_proj`、`f_b_proj`、`g_a_proj`、`g_b_proj` 各自一個 | 6 | 7 |

```python
self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
    hidden_size,
    [8192, 8192, 8192, 64],   # column-parallel: q, k, v, beta
    [128, 128],               # replicated:      f_a, g_a
    quant_config=None,
)
self.fused_fg_b_proj = ColumnParallelBatchedLinear(2, 128, 8192)  # f_b 與 g_b 一起 batched
```

`MergedColumnParallelRepeatedLinear` 的特點是同一個權重矩陣裡混了「要 TP 切」和
「要 replicate」兩種 partition。fused module 一律用 `quant_config=None` 建，也就是
BF16 GEMM，所以「可 fuse」的意思是 **checkpoint 把這些 projection 留在 BF16**：

- 開了 LoRA → 不 fuse。
- `quant_config is None`（BF16 ckpt）→ fuse。
- 否則 quant method 必須在 `{fp8, mxfp8, modelopt_fp8, modelopt_fp4, modelopt_mixed, quark}`
  裡，而且每個來源 projection 都沒被量化（被 ignore / exclude，保留原精度），判斷方式是
  `quant_config.is_linear_unquantized(prefix)`。只要有一個被量化就退回下一種 layout。
- `quark` 是 #41870（merge commit `af1bef3eaf`，已在 1012 image 裡）加進名單的，更舊的 tree 不會 fuse；
  `amd/GLM-5.3-Flash-Quark-MXFP4` 把這些 projection 全部 exclude 成 BF16，所以合併後走
  第一種 layout（7 → 3 個 GEMM）。Quark 的 `get_quant_method` 會把沒被 exclude 的 prefix
  登記成 online-quantized layer，因此 `QuarkConfig.is_linear_unquantized` 直接查
  exclude list，不去呼叫它。

fused in-proj 在 MI355X 上是 aiter 的 BF16 hgemm（trace 裡的 `hgemm_bf16_t16x16x128...`），
TP4 的 tuning 在 aiter#5982（`glm53_flash_bf16_tuned_gemm.csv`）。這個 PR 在 aiter 已 merge，
但 1012 image 的 aiter（`e7d2453f2`）還沒有，要手動把 CSV 複製到
`aiter/configs/model_configs/`（`~/SGLang-benchmarks/tmp/agentic_1014/run_triton.sh` 有做）。

3 個 GEMM 是資料相依的下限：`fused_fg_b_proj` 要等 f_a / g_a 的 128 維輸出，
`o_proj` 要等 recurrence 和 `o_norm`。`f_b(f_a(x))` 中間沒有非線性，理論上可以把
low-rank 預先乘開併進第一個 GEMM，但每 rank 每層權重會從約 3 MB 變約 34 MB，
decode 多讀的頻寬抵掉省下的那次 launch。

### PTPC FP8 projection（PR #38764，opt-in，1012 image 沒有）

`SGLANG_OPT_GLM53_KDA_PTPC_MODULES`（逗號分隔，可選 `qkv_proj`、`f_a_proj`、`g_a_proj`、
`o_proj`）讓 BF16 的 KDA projection 在 **大 M（prefill）** 時改走 per-token-per-channel FP8 GEMM，
小 M 仍是 BF16，所以只影響 TTFT：

| 條件 | 行為 |
|---|---|
| 不是 AITER gfx950 | 設了就 raise |
| `qkv_proj` / `f_a_proj` / `g_a_proj` | 必須三個一起選（共用同一份 activation quant） |
| fused layout（Quark ckpt） | `Glm53KdaPackedPtpcLinearMethod`；TP4（每 rank qkv 6144）M ≤ 4095 走 BF16，4096 ≤ M ≤ 16384 走 FP8，更大回 BF16 |
| `o_proj` | 只在每 rank K ∈ {1024, 2048}（TP8 / TP4）；M ≤ 5631 走 BF16 |
| `o_norm` + FP8 quant 融合 | `rms_norm_gated_per_token_fp8`，K=2048 時 M ≥ 256、K=1024 時 M ≥ 1 |

mHC 的 `hc_attn_pre` 也可以直接吐出 FP8 activation 給 PTPC 用（`can_consume_mhc_prequant`）。
目前 agentic recipe 設的是 `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`。

## Recurrence 數學

State：每個 head 一個 \( S \in \mathbb{R}^{128 \times 128} \)（`[head_v_dim, head_k_dim]`），
pool 裡存成 `[local_heads, 128, 128]`。

**Gate activation** — GLM-5.3-Flash 設了 `gate_lower_bound = -5.0`，所以走 **safe-gate**
分支（不是 softplus）：

\[
g_k = \text{lower\_bound} \cdot \sigma\!\big(e^{A_{\log}} \cdot (a_k + \text{dt\_bias}_k)\big)
\]

程式碼（`kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py`，`USE_LOWER_BOUND` 分支）：

```python
x = b_a + b_dt_bias
if USE_LOWER_BOUND:
    b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)   # GLM-5.3 走這條
else:
    b_g = -tl.exp(b_A_log) * softplus_x
```

**每個 token 的更新**（`IS_KDA=True`，q/k 先做 L2 norm，scale `1/√128`）：

```
S       ← exp(g) ⊙ S                      # per-K-channel 遺忘（gate 是 vector 不是 scalar）
v'      ← σ(beta) · (v − S @ k)            # delta rule：先減掉已經記住的部分
S       ← S + k ⊗ v'                       # 再寫入殘差
o       ← S @ q
```

對應 kernel：

```python
if IS_KDA:
    b_h *= tl.exp(b_g[:, None])
b_v -= tl.sum(b_h * b_k[:, None], 0)
b_v *= b_beta
b_h += b_k[:, None] * b_v[None, :]
b_o  = tl.sum(b_h * b_q[:, None], 0)
```

「gate 是 per-K-channel 的 vector」是 KDA 跟 Mamba2 / GDN 最大的差別——
`f_b_proj` 輸出 8192 維（= 64 head × 128 channel），每個 channel 有自己的遺忘率。

輸出再過 `o_norm = FusedRMSNormGated(sigmoid)`，用 `g_b(g_a(x))` reshape 成
`[s, 64, 128]` 當 gate，最後 `o_proj`。

## State cache 大小

`KimiLinearStateShape.create`（`configs/mamba_utils.py`）：

```
conv_state     = (conv_kernel_size - 1,  3 × 8192 / tp)   # 存 q,k,v 各自的 3 個過去 token
temporal_state = (64 / tp, 128, 128)
```

TP4（`local_heads = 16`）、dtype 用預設（conv bf16、state fp32）：

| 部分 | shape | bytes / seq / layer / rank |
|---|---|---|
| conv state | `(3, 6144)` bf16 | 36,864 B |
| recurrent state S | `(16, 128, 128)` fp32 | 1,048,576 B |
| **每層合計** | | **1,085,440 B ≈ 1.04 MiB** |
| **× 34 KDA 層** | | **≈ 35.2 MiB（36.9 MB）/ sequence / rank** |

TP1 的話是 `(3, 24576)` + `(64,128,128)` → 每層 ≈ 4.14 MiB、34 層 ≈ 141 MiB。

**開 MTP / speculative 時還有兩塊額外的 state**，啟動 log 的
`Mamba Cache is allocated.` 那行會列出來：

- `intermediate_ssm_state_cache`：`[34 層, spec_state_size + 1, num_draft_tokens, 16, 128, 128]`
  fp32，給 verify 時每個 draft token 的 state 用，跟 running request 數和 draft token 數成正比。
- `intermediate_conv_window_cache`：conv 版的同一件事，小很多。

實測（TP4、EAGLE 5 steps / 6 draft tokens、`--mamba-full-memory-ratio 0.5`、
`--mamba-radix-cache-strategy extra_buffer`）：`max_mamba_cache_size: 841, conv_state 1.01GB,
ssm_state 28.78GB, intermediate_ssm_state_cache 16.61GB, intermediate_conv_window_cache 0.58GB`。
也就是 spec 用的 intermediate cache 大約是主 state 的 6 成，估算 KDA pool 時不能漏掉。

**這是固定值，跟 sequence length 完全無關。** 對比 11 層 MLA 的 paged KV：
`kv_lora_rank + qk_rope_head_dim = 512 + 0 = 512` 個元素 / token / layer，
BF16 下 1024 B，11 層 = 11 KiB/token；加上 indexer 的 pooled FP8 K cache
（每 4 個 token 一個 132 B 的 entry → 33 B/token/layer，11 層 ≈ 363 B/token）
總共約 **11.3 KiB / token**。64K context 一條 sequence ≈ 727 MiB。
如果 45 層全都是 MLA，同樣 context 會是約 4.1 倍。

dtype 可由 `SGLANG_MAMBA_CONV_DTYPE` / `SGLANG_MAMBA_SSM_DTYPE` 調整。

## Kernel dispatch

`Glm5NextLinearAttention` → `RadixLinearAttention` → `HybridLinearAttnBackend`
（按 `layer_id` 分流）→ `KDAAttnBackend`。註冊在
`layers/attention/attention_registry.py`，ROCm 上預設 `--linear-attn-backend triton`
（kernel 在 `layers/attention/linear/kernels/kda_triton.py`）。

### Prefill / extend（chunked，chunk size 64）

（#39121 "Tune GLM-5.3-Flash KDA prefill kernels on gfx950" 已 **closed、沒有 merge**，
1012 image 和 PR stack 都沒有，下表是 upstream 的 kernel。）

| 步驟 | kernel |
|---|---|
| 1 | `causal_conv1d_fn`（SiLU，同時更新 conv state） |
| 2 | `chunk_kda` → `chunk_kda_fwd` |
| 2a | `kda_gate_chunk_cumsum_vector_kernel`（gate 在 log2 空間做 cumsum） |
| 2b | `chunk_kda_fwd_intra`（WY / triangular solve） |
| 2c | `chunk_gated_delta_rule_fwd_h`（跨 chunk 的 state 傳遞） |
| 2d | `chunk_gla_fwd_o_gk`（算輸出） |

### Decode（單 token）

| 步驟 | kernel |
|---|---|
| 1 | `causal_conv1d_update` |
| 2 | `fused_sigmoid_gating_delta_rule_update` |

**注意**：`KDAAttnBackend.forward_decode` 的條件是
`supports_packed_decode and (layer.lower_bound is None or supports_packed_decode_safe_gate)`，
成立才走 `kernel_dispatcher.packed_decode`（`fused_recurrent_kda_packed_decode`）。
`supports_packed_decode_safe_gate` 只有 CUDA 的 `kda_helion.py` 設成 True，ROCm 的
`TritonKDAKernel` 沒有；GLM-5.3 的 `gate_lower_bound = -5.0`，所以在 MI355X 上一律走上表的
`fused_sigmoid_gating_delta_rule_update`（1012 trace 每 decode step 34 次）。

另外兩條「fully fused decode」（conv + recurrence + o_norm 一個 kernel）——CUDA 的
`kda_fused_decode` 和 gfx95 的 `kda_fused_decode_aiter_hip`——只有在 model 設了
`layer._k3_fused_decode_args` / `_k3_hip_fused_decode_args` 時才會走，而這些只有
`kimi_k3.py` 會設，**GLM-5.3 不管 bound 是多少都用不到**。

`Glm5NextLinearAttention` 讀的是 `linear_attn_config["gate_lower_bound"]`；
config 頂層的 `linear_lower_bound` 只在沒有 `linear_attn_config` 時當 fallback，兩份
checkpoint 都有 `linear_attn_config`，所以 **override `linear_lower_bound` 沒有任何作用**。
`gate_lower_bound` 是 checkpoint 的數值要求，不要為了效能去改。

MTP / verify 用同一個 `fused_sigmoid_gating_delta_rule_update`，但帶
`disable_state_update=True`。

## ROCm / AITER

**GLM-5.3 的 KDA conv / recurrence / `o_norm` 沒有用到 AITER 或 ROCm 原生 kernel**，全部走
`python/sglang/kernels/ops/attention/fla/` 下的 Triton kernel（Triton 在 ROCm 上可用）。
AITER 只出現在 projection GEMM（BF16 hgemm、#38764 的 PTPC FP8 GEMM）。
repo 裡雖然有 `kda_fused_decode_aiter_hip`，但只接在 `kimi_k3.py` 上（見上一節）。
`SGLANG_USE_AITER=1` 影響的是 DSA MLA 的 KV layout 和 MoE / mHC，不是 KDA recurrence。

CUDA-only 的加速選項（ROCm 用不到）：`flashkda`、`cutedsl`、`nvidia_kda`、`ptx_kda`、
`helion` prefill backend，以及 FlashInfer 的 `recurrent_kda`。

## Hybrid memory pool

`HybridLinearKVPool`（`mem_cache/memory_pool.py`），由
`KVCacheConfigurator._build_hybrid_linear_kv_pool` 建起來：

| 子 pool | 負責的層 | 儲存模型 | backend |
|---|---|---|---|
| `full_kv_pool` = `DSATokenToKVPool` | 11 層 DSA（`i % 4 == 3`） | **paged，隨 seq_len 成長**：MLA latent KV（512/token）+ paged indexer FP8 K cache（`index_head_dim=128`，k-pool 後每 4 token 一個 entry） | DSA backend |
| `mamba_pool` = `MambaPool` | 34 層 KDA | **每 request 固定一個 slot**：conv + temporal state | `KDAAttnBackend` |

分流靠 `HybridLinearAttnBackend._is_full_attn(layer_id)` 查 `full_attention_layer_ids`。
`HybridLinearKVPool.get_kv_buffer(layer_id)` 只服務 full-attention 層；
KDA 的 state 要透過 `req_to_token_pool.mamba2_layer_cache(layer_id)` 取。

實務影響：**兩個 pool 各自有 capacity，concurrency 取兩者較小值。**
啟動時如果 log 說 `max_running_requests` 被調降，通常是 KDA state pool 不夠，
用 `--mamba-full-memory-ratio` 或 `--max-mamba-cache-size` 加大。
反過來，長 context 的 agentic workload（ISL ~100K、ratio 0.5）是 KV pool 先滿：
conc 64 時 `full token usage` 到 1.00，`mamba usage` 最高只有 ~0.38。

## 資料流總結

```
hidden [N, 4096]
  ├─ qkv_proj  ─(causal_conv1d k=4, SiLU)─► mixed_qkv [N, 3×2048]
  ├─ b_proj    ────────────────────────────► beta  [N, 16]
  ├─ f_a → f_b ────────────────────────────► forget gate a [N, 2048]
  └─ g_a → g_b ────────────────────────────► norm gate     [N, 16, 128]
                     │
                     ▼
        prefill: chunk_kda (chunk 64)
        decode : fused_sigmoid_gating_delta_rule_update
        state  : conv [3, 3×2048] + S [16,128,128]   ← 固定大小，就地更新
                     │
                     ▼
        FusedRMSNormGated(sigmoid) → o_proj → [N, 4096]
```
