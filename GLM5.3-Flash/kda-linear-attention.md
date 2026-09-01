# KDA linear attention（34 / 45 層）

`Glm5NextLinearAttention`（`models/glm5_next.py:326-606`）。KDA = Kimi Delta Attention
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

當 `quant_config is None and head_shard_size == tp_size` 時（`do_fuse_qkvbfg=True`），
6 個 projection 合成 2 個 kernel：

```python
self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
    hidden_size,
    [8192, 8192, 8192, 64],   # column-parallel: q, k, v, beta
    [128, 128],               # replicated:      f_a, g_a
)
self.fused_fg_b_proj = ColumnParallelBatchedLinear(2, 128, 8192)  # f_b 與 g_b 一起 batched
```

`MergedColumnParallelRepeatedLinear` 的特點是同一個權重矩陣裡混了「要 TP 切」和
「要 replicate」兩種 partition。GLM-5.3-Flash 官方 ckpt 是 FP8 量化的，所以
`quant_config` 不是 None → **實際跑的是 unfused 路徑**（除了 BF16 ckpt 的情況）。

## Recurrence 數學

State：每個 head 一個 \( S \in \mathbb{R}^{128 \times 128} \)（`[head_v_dim, head_k_dim]`），
pool 裡存成 `[local_heads, 128, 128]`。

**Gate activation** — GLM-5.3-Flash 設了 `gate_lower_bound = -5.0`，所以走 **safe-gate**
分支（不是 softplus）：

\[
g_k = \text{lower\_bound} \cdot \sigma\!\big(e^{A_{\log}} \cdot (a_k + \text{dt\_bias}_k)\big)
\]

程式碼（`kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:191-207`）：

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
| **每層合計** | | **≈ 1.08 MiB** |
| **× 34 KDA 層** | | **≈ 36.8 MiB / sequence / rank** |

TP1 的話是 `(3, 24576)` + `(64,128,128)` → 每層 ≈ 4.14 MiB、34 層 ≈ 141 MiB。

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
`layers/attention/attention_registry.py:503`，ROCm 上預設 `--linear-attn-backend triton`。

### Prefill / extend（chunked，chunk size 64）

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

**注意**：因為 `gate_lower_bound = -5.0` 不是 None，`supports_packed_decode` 的
快路徑被關掉（`layers/attention/linear/kda_backend.py:557`），所以跑不到
`fused_recurrent_kda_packed_decode_kernel`，CUDA-only 的 `kda_fused_decode`
（conv + recurrence + o_norm 一個 kernel）也用不到。這是 checkpoint 的數值要求，
**不要為了效能去 override `linear_lower_bound`**。

MTP / verify 用同一個 `fused_sigmoid_gating_delta_rule_update`，但帶
`disable_state_update=True`。

## ROCm / AITER

**KDA 路徑沒有 AITER 或 ROCm 原生 kernel**，全部走
`python/sglang/kernels/ops/attention/fla/` 下的 Triton kernel（Triton 在 ROCm 上可用）。
`SGLANG_USE_AITER=1` 影響的是 DSA MLA 的 KV layout 和 MoE / mHC，不是 KDA recurrence。

CUDA-only 的加速選項（ROCm 用不到）：`flashkda`、`cutedsl`、`nvidia_kda`、`ptx_kda`、
`helion` prefill backend，以及 FlashInfer 的 `recurrent_kda`。

## Hybrid memory pool

`HybridLinearKVPool`（`mem_cache/memory_pool.py:3710`），由
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
