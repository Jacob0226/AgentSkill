# mHC — 4 條 parallel residual stream

`mhc=true, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6`。
全名是 **Manifold-Constrained Hyper-Connections**。

程式碼位置：

| 內容 | 位置 |
|---|---|
| Residual stream 狀態（`MHCState`） | `srt/layers/layer_boundary/residual/mhc.py` |
| `hc_attn_pre` / `hc_ffn_pre` / `hc_ffn_post_pre` / `hc_post` | `Glm5NextDecoderLayer`（`srt/models/glm5_next.py`） |
| kernel dispatch、`hc_expand` / `hc_contract` | `kernels/ops/layernorm/mhc.py` |
| attn→FFN boundary 融合 | `srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` |
| FFN all-reduce + `hc_post` 融合 | `srt/layers/layer_boundary/exit.py` |

**MTP draft layer（layer 45）沒有 mHC**：它是 `DeepseekV2DecoderLayer`，單一 residual，
checkpoint 也沒有 `model.layers.45.hc_*`。以下說的「每層」都是指 45 層 text layer。

一般 transformer 的 residual 是一條：`x = x + f(norm(x))`。mHC 改成 **4 條 parallel
residual stream**，每個 sublayer 用 learned、input-dependent 的 gate 決定
「要從 4 條裡怎麼混出一個 input」以及「輸出要怎麼寫回這 4 條」，中間的 stream-to-stream
mixing matrix 還被 Sinkhorn 正規化成近似 doubly stochastic（這就是 manifold-constrained
的意思）。

## Hidden state 的形狀

| 位置 | shape |
|---|---|
| Layer 之間的 hidden state | `[tokens, hc_mult × hidden] = [s, 16384]` |
| Kernel 內部的 3D view | `[s, 4, 4096]` |
| Attention / FFN 實際計算的 input | `[s, 4096]`（由 `hc_pre` 混出來） |

`ModelConfig.hc_hidden_size = hidden_size × hc_mult = 16384`，PP proxy tensor 和
DP buffer 都用這個寬度。

**重要**：mHC 模式下傳統的 residual tensor 被折進加寬的 hidden state。layer 之間的
residual 由 `layer_boundary` 的 `residual_batch`（`ResidualStream`）管理，
`Glm5NextModel.forward` 最後是 `residual_batch.complete_output` → `residual_batch.final_norm`；
最後一層的 FFN write-back（`_FfnUpdate._finish`）已經做完 `hc_contract`，
所以 final norm 拿到的是 `[s, 4096]`。

### 展開與收合

```python
# python/sglang/kernels/ops/layernorm/mhc.py
def hc_expand(x, n):   # layer 0 進入前：[s, 4096] -> [s, 16384]
    return x.repeat(1, n)                      # 4 條 stream 初始完全相同

def hc_contract(x, n): # 最後一層之後：[s, 16384] -> [s, 4096]
    return x.unflatten(-1, (n, -1)).mean(dim=-2)   # 對 4 條取平均
```

`hc_expand` 由第一層 readout 的 `init_residual` 觸發，`hc_contract` 在
`MHCState.is_last_layer` 時由 `_FfnUpdate._finish` 觸發。
GLM 用的是純平均，checkpoint 裡也沒有 `hc_head` 這類 learned mixer。

## 每層的參數（6 個 tensor）

`mix_hc = (2 + hc_mult) × hc_mult = 24`，`hc_dim = hc_mult × hidden = 16384`：

| 參數名（checkpoint 名稱完全一致） | shape | dtype |
|---|---|---|
| `layers.{i}.hc_attn_fn` | `[24, 16384]` | fp32 |
| `layers.{i}.hc_attn_scale` | `[3]` | fp32 |
| `layers.{i}.hc_attn_base` | `[24]` | fp32 |
| `layers.{i}.hc_ffn_fn` | `[24, 16384]` | fp32 |
| `layers.{i}.hc_ffn_scale` | `[3]` | fp32 |
| `layers.{i}.hc_ffn_base` | `[24]` | fp32 |

參數直接掛在 `Glm5NextDecoderLayer` 上（不是在子 module 裡），這樣
`default_weight_loader` 不用做任何 rename 就能對上 checkpoint。
`hc_post` 是 **parameter-free** 的，它重用對應 `hc_pre` 算出來的係數。

## `hc_pre`：4 條 stream → 一個 layer input

輸入 residual stream `x`，view 成 `[s, 4, H]`：

1. **RMS 縮放**（對整個 4H 一起算，不是 per-stream）
   \[ r = \mathrm{rsqrt}\!\left(\tfrac{1}{4H}\textstyle\sum x^2 + \varepsilon_{rms}\right) \]
2. **Mixing GEMM**：`mixes = (x_flat @ hc_fn.T) * r` → `[s, 24]`
3. **切成三組**（順序已驗證）：

| 組 | row index | reshape | 公式 |
|---|---|---|---|
| `pre` | `[0:4)` | `[s,4]` | `sigmoid(mixes[0:4] * scale[0] + base[0:4]) + hc_eps` |
| `post`（`h_post`） | `[4:8)` | `[s,4]` | `2.0 * sigmoid(mixes[4:8] * scale[1] + base[4:8])` |
| `comb`（`h_res`） | `[8:24)` | `[s,4,4]` | `mixes[8:24] * scale[2] + base[8:24]` |

`post_mult_value = 2.0` 是 GLM 寫死的常數，讓 post gate 上限到 ~2.0 而不是 ~1.0。

4. **Sinkhorn 只作用在 `comb`**（`pre` 和 `post` 不做）：

```python
comb = comb.softmax(-1) + hc_sinkhorn_eps
comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)   # 第 1 輪
for _ in range(sinkhorn_iters - 1):                            # 再 19 輪
    comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
```

20 輪之後 `comb` 近似 doubly stochastic（非負、每列和 ≈ 1、每行和 ≈ 1），
保證混合操作不會放大或衰減 residual 的整體 scale。

5. **混出 layer input**：\( \text{layer\_input} = \sum_{j=0}^{3} \text{pre}_j \cdot x_j \)，`[s, H]`
6. **順便融合 output layernorm**：把 `input_layernorm.weight` 或
   `post_attention_layernorm.weight` 傳進來，在同一個 kernel 內做完 RMSNorm。

回傳 `(layer_input [s,H], h_res [s,16], h_post [s,4], norm_fused)`。

## `hc_post`：module 輸出 → 寫回 4 條 stream

Parameter-free，`x` 是 attention 或 FFN 的輸出 `[s,H]`，`r` 是當前的 4 條 stream：

\[
\text{out}_{j,h} = \underbrace{\text{h\_post}_j \cdot x_h}_{\text{把輸出散進第 }j\text{ 條}}
+ \underbrace{\sum_{k=0}^{3} \text{comb}_{k,j} \cdot r_{k,h}}_{\text{4 條之間互相混合}}
\]

index 慣例：`comb[k, j]` 是「從舊的第 k 條流到新的第 j 條」。

## 一層的完整順序

```
residual stream [s,4H]
   │ hc_attn_pre   (+ input_layernorm)
   ▼
[s,H] ── Attention ──> [s,H] ── TP all-reduce
   │ hc_post(attn)  ← 用 h_res/h_post(attn)
   ▼
residual stream [s,4H]
   │ hc_ffn_pre    (+ post_attention_layernorm)     ← gfx95 上與上一步融合
   ▼
[s,H] ── MoE / MLP ──> [s,H] ── TP all-reduce
   │ hc_post(ffn)   ← 用 h_res/h_post(ffn)
   ▼
residual stream [s,4H]  → 下一層
```

所以 **一層 4 次 mHC 操作**（2 pre + 2 post），45 層共 180 次。

在 AITER gfx950 上每個 `hc_pre` 本身是 **2 個 kernel**（`mhc_pre_gemm_sqrsum_kernel` +
`mhc_pre_big_fuse_rmsnorm_kernel`），`hc_post` 是 1 個（`mhc_post`）。decode（≤ 16 token）
實測的每層 kernel 數：

| 設定 | attn pre | attn post + FFN pre | FFN all-reduce + post | 每層 mHC kernel |
|---|---|---|---|---|
| 沒融合 | 2 | 1 + 2 | AR + 1 | 6（+ AR） |
| boundary 融合（image 預設） | 2 | **2**（`mhc_fused_post_pre_gemm_sqrsum_kernel` + `mhc_pre_big_fuse_rmsnorm_kernel`） | AR + 1 | 5（+ AR） |
| 再加 #42020（`SGLANG_ROCM_FUSED_AR_MHC_POST=1`） | 2 | 2 | **1**（`allreduce_mhc_post_large_m_kernel`） | 5（AR 也併進去） |

`MHCState`（`layer_boundary/residual/mhc.py`）負責串起來：

| method | 做什麼 |
|---|---|
| `_AttnReadout.init_residual` | 第一層 `hc_expand` |
| `read_attn_input` | `hc_attn_pre`（+ input_layernorm） |
| `update_and_read_ffn_input` | `hc_post(attn)` + `hc_ffn_pre`；先試 `hc_ffn_post_pre` 融合，回 `None` 才拆開跑 |
| `apply_post` | `hc_post(ffn)`；最後一層由 `_FfnUpdate._finish` 再 `hc_contract` |
| `reduce_and_apply_post` | FFN TP all-reduce + `hc_post` 同一個 kernel（見下方「通訊」） |

## Kernel dispatch

`_mhc_pre_dispatch` / `_mhc_post_dispatch`（`mhc.py`），依序：

| 優先序 | 條件 | kernel |
|---|---|---|
| 1 | HIP + gfx95 + `SGLANG_USE_AITER` | **AITER** `aiter.ops.mhc.mhc_pre` / `mhc_post` |
| 2 | CUDA + `SGLANG_OPT_USE_TILELANG_MHC_{PRE,POST}` | **TileLang** `mhc_pre_*` / `mhc_post_tilelang` |
| 3 | fallback | 純 torch `_mhc_pre_torch` / `_mhc_post_torch` |

Dispatch 函式上有 `@torch._dynamo.disable`，**不走 torch.compile**。
啟用 AITER 時 log 會印 `Using AITER gfx950 mHC pre/post kernels`。

attn→FFN 邊界融合：只有 `is_cross_layer_mhc_fusion_enabled()` 為真時，`MHCState` 才會拿到
`hc_ffn_post_pre`；它在 HIP + gfx95 + `SGLANG_USE_AITER=1`（或 CUDA 上
`SGLANG_OPT_FUSE_MHC_POST_PRE` + TileLang pre/post）時成立，所以 MI355X 上預設就開。
`Glm5NextDecoderLayer.hc_ffn_post_pre` 在 token 數
> `_MHC_FUSED_BOUNDARY_MAX_TOKENS = 16` 時直接回 `None`（16 token 以內融合較快，24 持平），
否則呼叫 `apply_mhc_post_pre_boundary`，順序是：
`try_aiter_fused_mhc_post_pre`（aiter HIP）→ `try_fused_hc_post_pre`（Triton，
`_FUSED_HC_POST_PRE_M_THRESHOLD = 64`）→ TileLang `mhc_fused_post_pre` →
都不行就回 `None`，由 `update_and_read_ffn_input` 跑未融合的 `hc_post` + `hc_ffn_pre`。
這條路徑沒有專屬 log，要從 trace 看 kernel 名稱判斷。

融合的內容：`hc_post(attn)` 的 scatter/mix 併進 `hc_ffn_pre` 的 RMS + mixing GEMM
（`mhc_fused_post_pre_gemm_sqrsum_kernel`）；Sinkhorn + gate + `post_attention_layernorm`
仍是另一個 `mhc_pre_big_fuse_rmsnorm_kernel`。**是 3 個 kernel 變 2 個，不是 1 個**
（`hc_ffn_post_pre` 的註解："two launches instead of three"）。

## 成本

### 記憶體與計算

| 項目 | 相對於單條 residual |
|---|---|
| Layer 之間的 hidden state | **4×**（`[s,16384]` vs `[s,4096]`） |
| Attention / MoE 的 activation | **1×**（`[s,4096]`，完全不變） |
| `h_post` / `h_res` scratch | 很小（每 token fp32 4 + 16 個數） |
| `hc_pre` 的 mixing GEMM | K 維是 4H = 16384，每層兩次 `[s,16384] × [16384,24]` |
| mHC 參數量 | 24 × 16384 × 2 × 45 ≈ 35.4 M（fp32，約 141 MB） |

### 通訊 —— **不是全部 4 倍**

| Collective | tensor shape | 4 倍？ |
|---|---|---|
| Attention output all-reduce | `[s, 4096]` | 否 |
| MoE / MLP output all-reduce | `[s, 4096]` | 否 |
| Layer 之間的 DP gather/scatter | `[s, 16384]` | **是** |
| PP hidden state 傳遞 | `[s, 16384]` | **是** |

原因是 all-reduce 發生在 module 輸出（`[s,H]`）上，`hc_post` 才把它散進 4 條 stream。
只有跨 layer / 跨 PP stage 搬 residual stream 時才是 4 倍。

### FFN all-reduce + `hc_post` 融合（ROCm，PR #42020，預設關）

**這是 #42020 加的，1012 image 沒有**（image 裡搜不到 `SGLANG_ROCM_FUSED_AR_MHC_POST`），
要疊 PR stack 才有。`SGLANG_ROCM_FUSED_AR_MHC_POST=1`（且 `SGLANG_USE_AITER=1`）時，FFN 的 TP all-reduce
和 `hc_post(ffn)` 合成一個 kernel（`group.fused_allreduce_mhc_post` → aiter
`fused_allreduce_mhc_post_only`，trace 裡是 `allreduce_mhc_post_large_m_kernel`，經
`MHCState.reduce_and_apply_post`）。條件由 `ExitPolicy._fuses_all_reduce_into_mhc_post` 判斷：
TP > 1、沒有 DP attention、沒有 MoE a2a、post-experts sum 是單一 all-reduce，且 batch
在 `_FUSED_AR_MHC_POST_MAX_TOKENS = 16` token 以內（decode）。kernel 拒絕時退回
`all_reduce` + `hc_post`。

## Debug 提示

- `hc_*` 參數是 **fp32**，即使模型是 FP8 / MXFP4 量化的。官方 FP8 ckpt 把它們列在
  `modules_to_not_convert` 裡。
- 想確認 AITER mHC pre/post 有沒有生效，grep server log 的
  `Using AITER gfx950 mHC pre/post kernels`（每個 TP rank 印一次）。boundary 融合和
  AR + `hc_post` 融合沒有 log，要看 trace：有 `mhc_fused_post_pre_gemm_sqrsum_kernel` 就是
  boundary 融合生效，有 `allreduce_mhc_post_large_m_kernel` 就是 #42020 生效。
- DFLASH / EAGLE3 的 aux hidden state capture 需要先 `hc_contract` 才能用
  （`_prepare_aux_hidden_state`），否則維度會是 4 倍。
