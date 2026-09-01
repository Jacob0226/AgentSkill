---
name: GLM5.3-Flash
description: >-
  GLM-5.3-Flash (zai-org/GLM-5.3-Flash, SGLang `Glm5NextForConditionalGeneration`)
  的 model 架構參考：hybrid attention（KDA linear attention + MLA/DSA k-pool sparse
  attention）、mHC 4-stream residual、288-expert MoE、NoPE text tower、24-layer vision
  encoder、MTP draft layer。Use when analysing, profiling, debugging, or tuning
  GLM-5.3-Flash in SGLang — 例如判斷某個 kernel 屬於哪一種 layer、估算 KV cache /
  state cache 大小、解讀 trace 裡的 layer breakdown，或比較它與 GLM-5.2 / DeepSeek-V3.2
  的差異。
---

# GLM-5.3-Flash 架構總覽

官方定位：natively multimodal MoE，**320B total / 18B active**，1M context，FP8 權重
（e4m3、weight_block_size 128×128）。SGLang 裡的 `model_type` 是 `glm5_next`，
entry class `Glm5NextForConditionalGeneration`。

參考檔案（在 SGLang 原始碼裡）：

| 內容 | 路徑 |
|---|---|
| Model 主體 | `python/sglang/srt/models/glm5_next.py` |
| Config parser | `python/sglang/srt/configs/glm5_next.py` |
| MTP draft layer | `python/sglang/srt/models/glm5_next_nextn.py` |
| KDA linear attention backend | `python/sglang/srt/layers/attention/linear/kda_backend.py` |
| DSA k-pool indexer | `python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py` |
| mHC kernels | `python/sglang/kernels/ops/layernorm/mhc.py` |

## 一句話描述

45 層 text decoder，每 4 層一組：**3 層 KDA linear attention + 1 層 MLA/DSA sparse
attention**；FFN 前 3 層是 dense MLP、其餘 42 層是 288-expert MoE；每一層的 residual
不是單一條而是 **4 條 parallel residual streams（mHC）**；整個 text tower **完全沒有
RoPE**（NoPE），位置資訊只由 KDA 的 short conv + recurrence 提供。

## 四個必須知道的設計

| 設計 | 是什麼 | 為什麼重要 |
|---|---|---|
| **Hybrid attention** | 34 層 KDA（linear attention，state 大小固定）+ 11 層 MLA/DSA（paged KV cache） | KV cache 只由 11 層貢獻，長 context 的 memory 是同尺寸 full-attention 模型的約 1/4 |
| **k-pool indexer** | DSA lightning indexer 對「每 4 個 token 壓成 1 個 K」的 pooled cache 算 logits，選 top-512 pool → 展開成 2048 token | logits 矩陣短 4 倍，indexer 成本大幅下降；輸出寬度是 **2051** 不是 2048 |
| **mHC** | hidden state 是 `[tokens, 4 × hidden]`，每個 sublayer 前後用 learned gate + Sinkhorn 正規化的 mixing matrix 在 4 條 stream 間讀寫 | residual tensor 大 4 倍、多出 `hc_pre`/`hc_post` kernel；trace 裡看到的 `mhc_*` kernel 就是它 |
| **NoPE** | `qk_rope_head_dim=0` + `mla_use_nope=true` + `skip_rope=True` | text 側 **完全沒有 RoPE kernel**（跟 GLM-5.2 最大的差異之一），只有 vision encoder 有 RoPE |

## Config 速查（`text_config`）

```
num_hidden_layers   45        hidden_size          4096       vocab_size    154880
first_k_dense_replace 3       intermediate_size    12288      swiglu_limit  10.0
n_routed_experts    288       num_experts_per_tok  8          n_shared_experts 1
moe_intermediate_size 2048    scoring_func         sigmoid    topk_method   noaux_tc
routed_scaling_factor 2.5     n_group / topk_group 1 / 1      norm_topk_prob true

num_attention_heads 64        q_lora_rank   1536   kv_lora_rank      512
qk_nope_head_dim    256       v_head_dim    256    qk_rope_head_dim  0     mla_use_nope true

index_n_heads 32   index_head_dim 128   index_topk 2048   index_kpool 4
index_kpool_compress true     index_kpool_always_select_tail true
index_share_for_mtp_iteration true      indexer_rope_interleave true（因 skip_rope 而無效）

linear_attn_config: num_heads 64, head_dim 128, short_conv_kernel_size 4,
                    gate_lower_bound -5.0

mhc true   hc_mult 4   hc_sinkhorn_iters 20   hc_eps 1e-6
num_nextn_predict_layers 1    max_position_embeddings 1048576
vision_config: depth 24, hidden 1024, heads 16, patch 14, image 448,
               spatial_merge_size 2, out_hidden_size 4096
```

## Layer 分佈（45 層，只有三種組合）

`layer_types[i] == "linear_attention"` 當且僅當 `i % 4 != 3`，
`mlp_layer_types[i] == "sparse"` 當且僅當 `i >= first_k_dense_replace = 3`。
因為 layer 3 同時是第一個 DSA layer 和第一個 MoE layer，所以不存在「DSA + dense MLP」：

| # | layer type | layers | 數量 |
|---|---|---|---|
| 1 | **KDA + dense MLP** | 0, 1, 2 | 3 |
| 2 | **MLA/DSA + MoE** | 3, 7, 11, …, 43 | 11 |
| 3 | **KDA + MoE** | 4-6, 8-10, …, 44 | 31 |
| — | MTP draft layer（layer 45，`nextn`） | 45 | 1 |

MTP layer 不在 `kda_layers` 清單裡，所以它是 **MLA/DSA + MoE**。

## 詳細文件

| 檔案 | 內容 |
|---|---|
| `layer-structure.md` | 三種 layer type 的完整 mermaid dataflow、per-layer kernel 清單、和 GLM-5.2 的逐項對照 |
| `mhc.md` | mHC 的數學（`hc_pre` / `hc_post` / Sinkhorn）、參數命名、kernel dispatch、記憶體與通訊成本 |
| `kda-linear-attention.md` | KDA 的 projection 清單、gated delta rule recurrence、state cache 大小、prefill/decode kernel |
| `dsa-kpool-indexer.md` | k-pool indexer 的 pooling、tail 選取、top-k 展開、kernel 清單、和 GLM-5.2 indexer 對照 |
| `serving-and-perf.md` | 啟動 flag、兩個 memory pool 的調參、MI355X TP4 實測數字、已知踩雷 |

## 常見誤判（讀 trace / debug 前先確認）

- **看不到 RoPE kernel 不是 bug。** text tower 是 NoPE。看到 RoPE 只可能來自 vision encoder。
- **`index_topk=2048` 不等於 attend 2048 個 token。** k-pool 選 512 個 pool 展開成 2048 個 history token，再無條件補上 `seq_len % 4` 個 tail token，輸出 buffer 寬度是 **2051**。
- **11 層才有 indexer。** indexer 成本要除以 11（不是 45）。`index_topk_freq` 沒設（預設 1），所以 11 層各自跑完整 indexer，沒有 GLM-5.2 那種跨層共用。
- **KDA 層沒有 KV cache。** 它用固定大小的 conv state + recurrent state，存在 `MambaPool` 而不是 KV pool。concurrency 常常是被 KDA state pool 卡住而不是 KV pool。
- **residual 是 4 倍寬。** layer 之間 `residual=None`，一切折進 `[tokens, 4*hidden]` 的 hidden state；只有最後一層才 `hc_contract` 平均回 `[tokens, hidden]`。
- **`gate_lower_bound=-5.0` 會關掉 KDA 的 fused decode 快路徑。** 不要為了效能去 override `linear_lower_bound`，那會改變數值。
