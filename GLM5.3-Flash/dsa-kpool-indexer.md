# DSA k-pool lightning indexer（11 / 45 層）

只有 `layer_types[i] == "deepseek_sparse_attention"` 的 11 層（3, 7, 11, …, 43）
加上 MTP layer 45 有 indexer。實作類別是 `IndexerKPool`
（`layers/attention/dsa/dsa_indexer_kpool.py`），選用邏輯在
`models/deepseek_v2.py:1853`：

```python
indexer_cls = IndexerKPool if get_dsa_index_kpool(config) > 1 else Indexer
```

相關 config：

```
index_topk 2048          index_kpool 4              index_n_heads 32
index_head_dim 128       index_kpool_compress true  index_kpool_always_select_tail true
index_share_for_mtp_iteration true                  indexer_rope_interleave true
```

啟動時三個 flag 必須同時成立，否則 assert 失敗：

```python
assert self.index_kpool > 1 and self.index_kpool_compress and self.index_kpool_always_select_tail
```

## 核心想法：對 pooled K 算分數

GLM-5.2 / DeepSeek-V3.2 的 indexer 是 **per-token**：logits 矩陣寬度 = `kv_len`。
GLM-5.3-Flash 把連續 4 個 token 壓成 **1 個 K 向量**存進 index cache，
logits 矩陣寬度變成 `floor(kv_len / 4)`，**短 4 倍**。
然後在 pooled 空間選 top-512，再展開回 2048 個 token index。

### Pooling 不是 mean / max，是 dim-wise gated softmax

當一個 4-token 的 pool 湊滿時，Triton kernel
`_kpool_softmax_rotate_write_cache_kernel` 做：

1. 每個 slot 取 `score[slot]`（來自 `index_kpool_compress_gate`，`[128, 4096]` bf16），
   加上 learned APE bias `index_kpool_compress_ape[slot]`（`[4, 128]` fp32）
2. **對 4 個 slot 做 softmax，但是 per head dimension**（每個 dim 有自己的權重分佈，
   不是一個 scalar gate）
3. 4 個 bf16 K 向量的加權和 → 一個 bf16 向量
4. **Hadamard-128** 旋轉（`_hadamard128`）
5. per-vector FP8 e4m3 量化（`scale_fmt="ue8m0"` 時 scale 取到 2 的次方）
6. 寫進 `IndexKeyCache` 的一個 pooled slot（128 B fp8 + 4 B scale = 132 B）

還沒湊滿 4 個的 token 放在 per-request 的 **tail ring buffer**
（`_compress_tail_k`、`_compress_tail_score`，bf16，`[max_reqs, 4 + draft_tokens, 128]`）。
Decode 時用 `_kpool_decode_update_and_maybe_write_cache_kernel`，
只有第 4 個 token 到達才觸發壓縮寫入。

## Top-k：512 個 pool → 2048 個 token → 補 tail

```python
group_topk = history_group_budget_for_topk(topk=2048, pool_size=4)   # = 512
token_ids  = group_ids.unsqueeze(-1) * 4 + arange(4)                 # 512 → 2048
```

然後 `append_kpool_tail_to_topk` **無條件**把 unpooled tail 的
`seq_len % 4` 個最新 token 接在後面（不看分數）：

```python
tail_start = pool_len * POOL_SIZE
tail_count = seq_len % POOL_SIZE       # 0..3
```

所以輸出 buffer 寬度是 **`index_topk + (index_kpool - 1) = 2051`**：

```python
topk_result = torch.empty((token_nums, self.index_topk + tail_pool), dtype=torch.int32)
```

`get_dsa_mtp_topk_width()`（`configs/model_config.py:284`）也回傳 2051，
MTP 的 IndexShare seed buffer 用這個寬度。

### 數字追蹤

| 步驟 | 值 |
|---|---|
| logits 矩陣寬度 | `floor(kv_len / 4)` |
| top-k 預算 | **512 個 pool** |
| 展開後 | 512 × 4 = **2048 個 history token** |
| tail 補上 | `seq_len % 4` = 0~3 個最新 token |
| **實際 attend 的 KV** | **最多 2051 個 token**（不是 8192） |

`compute_dsa_seqlens`（`dsa/utils.py:87`）算 attention metadata 時用同樣的契約：
history 部分向下對齊到 pool 邊界、clamp 到 `index_topk`、再加回 tail。

## Q / K 的投影鏈

| 階段 | 細節 |
|---|---|
| Q | `wq_b(q_lora)` → bf16，reshape `[L, 32, 128]` |
| K | `wk(x)` → `LayerNorm(128, dtype=fp32)` → bf16 |
| **RoPE** | **完全跳過**。`skip_rope=True`（`glm5_next.py:668`）且 `qk_rope_head_dim=0` |
| Hadamard | 只在 **Q** 上做一次（`rotate_activation`，在 `act_quant` 前）。K 的 Hadamard 在 pooling kernel 內部 |
| Q 量化 | `act_quant(query, block_size=128, scale_fmt="ue8m0")` → FP8 e4m3 |
| Head gate | `weights_proj(x).float() * n_heads**-0.5 * q_scale * head_dim**-0.5` |
| Compress gate | `F.linear(x, index_kpool_compress_gate)` → `[L, 128]` bf16 |
| K cache 寫入 | `_kpool_softmax_rotate_write_cache_kernel`（**不是** GLM-5.2 的 `indexer_k_quant_and_cache` / `fused_store_index_k_cache`） |

**Hadamard 數量的判讀跟 GLM-5.2 不一樣**：GLM-5.2 是「i1k = 1（只有 K）、i8k = 2（Q+K）」。
GLM-5.3-Flash 的 K Hadamard 融進了 pooling kernel，所以獨立的 Hadamard kernel 只有 Q 那一個。

### RoPE 在哪裡都沒有

`skip_rope` 的傳遞鏈：

```
glm5_next.py:668          skip_rope=True   →  DeepseekV2AttentionMLA
deepseek_v2.py:1876       indexer_kwargs["skip_rope"] = skip_rope
deepseek_v2.py:1901       if not skip_rope and qk_rope_head_dim > 0:  →  rotary_emb = None
dsa_indexer_kpool.py:684  if not self.skip_rope:                      →  整段 RoPE 跳過
```

而且 `is_mrope_enabled` 需要 `rope_scaling` 裡有 `mrope_section`，
GLM-5.3-Flash 的 config 沒有 `rope_scaling`，所以連 mrope 都不啟用。
**text tower 完全是 NoPE**，位置資訊來自 KDA 層的 causal conv + recurrence；
唯一的 RoPE 在 vision encoder（`get_rope(base=10000, rotary_dim=head_dim//2)`）。

## Kernel 清單

| 階段 | logits kernel | top-k kernel |
|---|---|---|
| Decode | CUDA: `deep_gemm.fp8_paged_mqa_logits`；ROCm: `aiter_paged_mqa_logits(preshuffle=True)`；SM90 特例: `tilelang_fp8_paged_mqa_logits` | `fast_kpool_topk_transform_fused`（group_topk 128–512）／`fast_topk_v2` + expand |
| Extend（ragged） | `fp8_mqa_logits`（DeepGEMM 或 `aiter.ops.triton.fp8_mqa_logits`） | `topk_from_pooled_history_logits` |
| Cache write | `_kpool_softmax_rotate_write_cache_kernel`、`_kpool_decode_update_and_maybe_write_cache_kernel`、`_kpool_assemble_softmax_rotate_write_cache_kernel`、`_kpool_write_tail_and_maybe_compress_kernel` | — |
| Metadata | `_update_kpool_write_plan_kernel`、`_kpool_build_ragged_layout_kernel`、`_gather_index_k_scale_prefix_into_kernel` | — |

Pooled cache 的 page table 由 `build_pooled_page_table_64` 從 64-token page table
每隔 `pool_size` 取一列做出來。

### K-only 快路徑

**只有 prefill / extend（CUDA）有**：`max(seq_lens) <= index_topk` 時
`_forward_cuda_skip_logits` 只算 K + 壓縮寫入，top-k 用
`metadata.topk_transform(dummy_logits)` 造一個全選的結果再補 tail padding：

```python
if forward_batch.forward_mode.is_extend_without_speculative():
    max_kv_len = forward_batch.seq_lens_cpu.max().item()
    skip_logits_computation = max_kv_len <= self.index_topk
```

**Decode 在 CUDA 上永遠算完整的 Q + logits**，沒有 k-only skip
（這點跟 plain `Indexer` 不同，後者有 HIP-only 的 decode skip）。

## 每層各自算，沒有跨層共用

`index_topk_freq` 在 config.json **沒有設定**，parser 預設 1。
`configs/model_config.py:260` 的判斷：

```python
freq = getattr(config, "index_topk_freq", 1)
return max(layer_id - 1, 0) % freq != 0      # freq=1 → 永遠 False → skip_topk=False
```

所以 11 層 DSA **每層都跑完整 indexer**，沒有 GLM-5.2 那種
「full layer 算 top-k、shared layer 重用」的機制（GLM-5.2 是 `index_topk_freq=4`，
78 層裡只有 21 層算）。跨層共用只在 `index_topk_freq > 1` 或
`index_topk_pattern` 含 `"S"` 時啟用。

**估算 indexer 成本時要除以 11，不是 45。**

## `index_share_for_mtp_iteration`

共用的是 **同一個 speculative iteration 內、跨 MTP / EAGLE draft step** 的 top-k，
**不是跨 transformer layer**：

- 機制在 `IndexTopKShareState.mtp_iteration`（`layers/attention/index_topk_share.py`）
- 共用的是 `spec_info.dsa_topk_indices`，step 0 算出來後面的 draft step 重用
- Seed buffer 寬度 = `index_topk + index_kpool - 1` = **2051**
- `seed_dsa_topk_from_draft_extend` 為真時，seed 來自 draft-extend（最後一個已驗證的
  token），而不是 draft-decode step 0
- **要求 `speculative_topk == 1`**，否則自動關掉
- PD disaggregation 下會連帶傳 seed metadata（`get_dsa_seed_metadata_dim`）

## 環境變數 / flag

### k-pool 專屬

| 變數 | 預設 | 作用 |
|---|---|---|
| `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION` | `0` | 融合 kpool metadata refresh；需要 `page_size=64` 且 topk 對齊 pool |
| `SGLANG_DSA_KPOOL_DEBUG_BOUNDS` | unset | 設 `"1"` 做 write plan 的 bounds check |
| `SGLANG_DSA_FUSE_TOPK` | `1` | 融合 page-table top-k transform |
| `SGLANG_DSA_HIP_DISABLE_PRESHUFFLE` | `0` | 關掉 AITER preshuffle paged-MQA（ROCm k-pool 需要 preshuffle） |
| `SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA` | `0` | 把 verify write-plan 收進 CUDA graph |

### 一般 DSA（同樣影響 k-pool）

| 變數 / flag | 作用 |
|---|---|
| `--dsa-prefill-backend` / `--dsa-decode-backend` | `tilelang` / `aiter` / `trtllm`；k-pool 的 tail indices **只有 FA3 / TileLang / TRTLLM sparse decode backend 支援** |
| `--dsa-topk-backend` | `topk_transform` 的實作（`sgl-kernel` / `torch` / `flashinfer`） |
| `SGLANG_DSA_TRITON_PREFILL` | prefill 用 Triton kernel |
| `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION` | logits chunking 的 OOM 門檻 |
| `SGLANG_DSA_TOPK_BROADCAST` | 由 rank 0 broadcast 最終 top-k |
| `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` | 預設就是 `index_topk` = 2048；短於此走 dense attention |

啟動 log 會印：`Set dense attention kv len threshold to model index_topk=2048 for
DeepSeek with DSA.`

## 與 GLM-5.2 indexer 對照

| 項目 | GLM-5.2 / DS-V3.2（`Indexer`） | GLM-5.3-Flash（`IndexerKPool`） |
|---|---|---|
| Index cache | 每 token 一個 FP8 K | 每 **4 個 token** 一個壓縮 FP8 K |
| Cache write kernel | `fused_store_index_k_cache` / `indexer_k_quant_and_cache` | `_kpool_*_softmax_rotate_write_cache` |
| Logits 寬度 | `kv_len` | `floor(kv_len / 4)` |
| Top-k 單位 | token | pool（再 ×4 展開） |
| 輸出寬度 | 2048 | **2051** |
| Tail 處理 | 無 | 無條件保留 `seq_len % 4` 個最新 token |
| 額外參數 | 無 | `index_kpool_compress_gate`、`index_kpool_compress_ape` |
| Pooling | 無 | dim-wise gated softmax + APE + Hadamard |
| RoPE | indexer 內做（Wq/Wk 之後） | **完全沒有** |
| Hadamard 獨立 kernel 數 | 1（k-only）或 2（Q+K） | 1（只有 Q；K 的融進 pooling kernel） |
| 有 indexer 的層 | 21 / 78 | 11 / 45，且每層各自算 |
