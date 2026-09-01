# mHC — 4 條 parallel residual stream

`mhc=true, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6`。
程式碼裡的全名是 **Manifold-Constrained Hyper-Connections**
（`python/sglang/srt/layers/communicator_mhc.py` 開頭的 docstring）。

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

**重要**：mHC 模式下 layer 之間 `residual = None`，傳統的 residual tensor 被折進
加寬的 hidden state。所以在 `Glm5NextModel.forward` 裡看到 `if residual is None:
hidden_states = self.norm(hidden_states)` 走的就是 mHC 路徑。

### 展開與收合

```python
# python/sglang/kernels/ops/layernorm/mhc.py
def hc_expand(x, n):   # layer 0 進入前：[s, 4096] -> [s, 16384]
    return x.repeat(1, n)                      # 4 條 stream 初始完全相同

def hc_contract(x, n): # 最後一層之後：[s, 16384] -> [s, 4096]
    return x.unflatten(-1, (n, -1)).mean(dim=-2)   # 對 4 條取平均
```

`hc_expand` 只在 `is_first_layer` 觸發，`hc_contract` 只在 `is_last_layer` 觸發。
GLM 用的是純平均，**沒有** 載入 checkpoint 裡的 `hc_head` learned mixer。

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

所以 **一層 4 次 mHC 操作**（2 pre + 2 post），45 層共 180 次；
`hc_attn_to_mlp` 融合之後在 gfx95 上是 3 次 kernel launch。

`MHCState` 負責串起來（`communicator_mhc.py`）：

| method | 做什麼 |
|---|---|
| `attn_split` | `hc_expand`（layer 0）+ `hc_attn_pre` |
| `attn_to_mlp` | `hc_post(attn)` + `hc_ffn_pre`（gfx95 融合成 `hc_attn_to_mlp`） |
| `mlp_combine` | `hc_post(ffn)`，最後一層再 `hc_contract` |

## Kernel dispatch

`_mhc_pre_dispatch` / `_mhc_post_dispatch`（`mhc.py`），依序：

| 優先序 | 條件 | kernel |
|---|---|---|
| 1 | HIP + gfx95 + `SGLANG_USE_AITER` | **AITER** `aiter.ops.mhc.mhc_pre` / `mhc_post` |
| 2 | CUDA + `SGLANG_OPT_USE_TILELANG_MHC_{PRE,POST}` | **TileLang** `mhc_pre_*` / `mhc_post_tilelang` |
| 3 | fallback | 純 torch `_mhc_pre_torch` / `_mhc_post_torch` |

Dispatch 函式上有 `@torch._dynamo.disable`，**不走 torch.compile**。
啟用 AITER 時 log 會印 `Using AITER gfx950 mHC pre/post kernels`。

AMD gfx95 的 attn→FFN 邊界融合（`apply_mhc_post_pre_boundary`）：
`aiter.ops.mhc.mhc_fused_post_pre` → 退到 `aiter.ops.triton.fusions.mhc.mhc_post_pre`
（只在 ≤ 64 token 時觸發）→ 再退到未融合的 `hc_post` + `hc_pre`。
第一次成功時會 log `Using fused AITER mHC attention-to-FFN boundary`。

融合的內容：`hc_post(attn)` 的 scatter/mix、`hc_ffn_pre` 的 RMS + GEMM + Sinkhorn +
gate、以及 `post_attention_layernorm`，全部在一個 kernel 裡。

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
`get_local_dp_buffer_mhc(group, n)` 的 `n` 在非最後一層是 `hc_mult=4`、
最後一層（已 contract）是 1。

## Debug 提示

- 看到 `residual is None` 一路傳到底是正常的，不是 bug。
- `hc_*` 參數是 **fp32**，即使模型是 FP8 量化的。它們列在 `modules_to_not_convert` 裡。
- 想確認 AITER 融合有沒有生效，grep server log 的
  `Using fused AITER mHC attention-to-FFN boundary` 和
  `Using AITER gfx950 mHC pre/post kernels`。
- DFLASH / EAGLE3 的 aux hidden state capture 需要先 `hc_contract` 才能用
  （`_prepare_aux_hidden_state`），否則維度會是 4 倍。
