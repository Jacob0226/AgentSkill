# GLM-5.3-Flash：三種 decoder layer

兩個獨立的 config axis 決定一層跑什麼：

| axis | config | 效果 |
|---|---|---|
| Attention | `layer_types[i]` | `i % 4 != 3` → **KDA** linear attention（34 層）；`i % 4 == 3` → **MLA + DSA** sparse attention（11 層） |
| FFN | `first_k_dense_replace = 3` | layer 0-2 用 dense `Glm5NextMLP`（= `DeepseekV2MLP`, intermediate 12288, clamped SwiGLU）；layer 3-44 用 `Glm5NextMoE`（= `DeepseekV2MoE`, 288 experts） |

因為 layer 3 同時是第一個 DSA layer 也是第一個 MoE layer，「DSA + dense MLP」這個組合
不存在，所以只有三種：

| # | layer type | layers | 數量 |
|---|---|---|---|
| 1 | **KDA + dense MLP** | 0, 1, 2 | 3 |
| 2 | **MLA/DSA + MoE** | 3, 7, 11, …, 43 | 11 |
| 3 | **KDA + MoE** | 4-6, 8-10, …, 44 | 31 |
| — | MTP draft（`nextn`, layer id 45） | 45 | 1，屬於 type 2 的結構 |

**每一層都套上 mHC**：進 attention 前一次 `hc_pre`、出 attention 後一次 `hc_post`、
進 FFN 前一次 `hc_pre`、出 FFN 後一次 `hc_post`。詳見 `mhc.md`。

## 全模型骨架

```mermaid
flowchart TD
    IN["input_ids"] --> EMB["embed_tokens<br/>154880 x 4096"]
    IMG["image / video"] --> VIS["Vision encoder<br/>24 blocks, hidden 1024, patch 14<br/>RoPE + VisionAttention + clamped SwiGLU<br/>PatchMerger -> 4096"]
    VIS -.->|"取代 image/video token 的 embedding"| EMB
    EMB --> EXP["hc_expand: [s,4096] -> [s,16384]<br/>4 條 residual stream（複製）"]
    EXP --> L0["layer 0-2<br/>type 1: KDA + dense MLP"]
    L0 --> L3["layer 3<br/>type 2: MLA/DSA + MoE"]
    L3 --> L4["layer 4-6<br/>type 3: KDA + MoE"]
    L4 --> DOT["… 重複 (DSA, KDA, KDA, KDA) x 10 …"]
    DOT --> L43["layer 43 type 2 -> layer 44 type 3"]
    L43 --> CON["hc_contract: [s,16384] -> [s,4096]<br/>對 4 條 stream 取平均"]
    CON --> NRM["final RMSNorm"] --> HEAD["lm_head 4096 x 154880"]
    L43 -.->|"hidden state"| MTP["layer 45: MTP draft<br/>MLA/DSA + MoE<br/>index_share_for_mtp_iteration"]

    classDef kda fill:#e3f0ff,stroke:#3366cc,color:#000
    classDef dsa fill:#ffe8cc,stroke:#cc7000,color:#000
    classDef hc fill:#f0e6ff,stroke:#7733cc,color:#000
    class L0,L4 kda
    class L3,L43,MTP dsa
    class EXP,CON hc
```

## Type 1 — KDA + dense MLP（layer 0, 1, 2）

```mermaid
flowchart TD
    IN["residual stream [s, 4x4096]"] --> HP1["hc_attn_pre<br/>RMS + GEMM(24x16384) + sigmoid gate<br/>+ Sinkhorn(20) + fused input_layernorm<br/>-> layer_input [s,4096], h_res, h_post"]
    HP1 --> QKV["fused_qkvbfg_a_proj<br/>column-parallel: q,k,v (3x8192) + beta(64)<br/>replicated: f_a, g_a (2x128)"]
    QKV --> FGB["fused_fg_b_proj (batched)<br/>128 -> 8192 x2: forget gate a, output gate g"]
    QKV --> CONV["causal_conv1d (kernel 4, SiLU)<br/>更新 conv state"]
    CONV --> KDA["KDA gated delta rule<br/>prefill: chunk_kda (chunk 64)<br/>decode: fused_sigmoid_gating_delta_rule_update<br/>state S [16 heads, 128, 128] fp32 就地更新"]
    FGB --> KDA
    KDA --> ON["o_norm: FusedRMSNormGated(sigmoid)<br/>用 g 當 gate"]
    ON --> OP["o_proj RowParallel 8192 -> 4096"]
    OP --> AR1["attn TP all-reduce（[s,4096]，不是 4 倍寬）"]
    AR1 --> HQ1["hc_post(attn) + hc_ffn_pre<br/>gfx95 上融合成一個 kernel"]
    HQ1 --> MLP["dense MLP<br/>MergedColumnParallel gate+up (12288)<br/>clamped SwiGLU (limit 10.0)<br/>RowParallel down"]
    MLP --> AR2["FFN TP all-reduce"]
    AR2 --> HQ2["hc_post(ffn)"] --> OUT["residual stream [s, 4x4096]"]

    classDef hc fill:#f0e6ff,stroke:#7733cc,color:#000
    classDef kda fill:#e3f0ff,stroke:#3366cc,color:#000
    classDef ffn fill:#fff2cc,stroke:#bf9000,color:#000
    class HP1,HQ1,HQ2 hc
    class QKV,FGB,CONV,KDA,ON,OP kda
    class MLP ffn
```

Layer 0 的特別之處：`hc_expand` 在這裡把 embedding 從 `[s,4096]` 複製成 `[s,16384]`
（4 條 stream 一開始完全相同），而且它前面沒有上一層的 all-reduce。

## Type 2 — MLA/DSA + MoE（layer 3, 7, 11, …, 43）

這是唯一有 KV cache 和 indexer 的層。注意 **沒有任何 RoPE**：`qk_rope_head_dim=0`
且 `skip_rope=True`，所以 `rotary_emb is None`，indexer 也跳過 RoPE。

```mermaid
flowchart TD
    IN["residual stream [s, 4x4096]"] --> HP1["hc_attn_pre (+ fused input_layernorm)"]
    HP1 --> QA["fused_qkv_a_proj_with_mqa<br/>q_a (1536) + kv_a_with_mqa (512)<br/>+ fused q/kv RMSNorm"]

    QA --> QB["q_b_proj 1536 -> 64x256<br/>（NoPE，全部是 nope 維度）"]
    QA --> KVB["kv_b_proj 512 -> 64x(256+256)<br/>latent KV 512/token 寫入 paged KV cache"]

    QA --> IDX["INDEXER (k-pool)<br/>wq_b -> Hadamard -> act_quant fp8 (32 heads x 128)<br/>wk -> LayerNorm(fp32)<br/>gate = x @ index_kpool_compress_gate<br/>每 4 個 token: dim-wise softmax pool + APE<br/>-> Hadamard -> fp8 -> IndexKeyCache (1 vec / 4 token)<br/>fp8_paged_mqa_logits over floor(kv/4) pooled cols<br/>top-512 pool -> 展開 2048 token -> 補 tail<br/>輸出 topk_indices [s, 2051]"]

    IDX --> MLA["sparse MLA attention<br/>只 gather 被選中的 <=2051 個 KV"]
    QB --> MLA
    KVB --> MLA
    MLA --> OP["o_proj 64x256 -> 4096"]
    OP --> AR1["attn TP all-reduce"]
    AR1 --> HQ1["hc_post(attn) + hc_ffn_pre"]
    HQ1 --> GATE["MoEGate router (fp32 logits)<br/>sigmoid score + noaux_tc<br/>topk 8 of 288, n_group=1<br/>norm_topk_prob, routed_scaling 2.5"]
    GATE --> EXP["routed experts (fp8)<br/>gate+up 2048, clamped SwiGLU, down<br/>+ 1 shared expert"]
    EXP --> RED["moe reduction: 8 routed + 1 shared"]
    RED --> AR2["FFN TP all-reduce"]
    AR2 --> HQ2["hc_post(ffn)"] --> OUT["residual stream [s, 4x4096]"]

    classDef hc fill:#f0e6ff,stroke:#7733cc,color:#000
    classDef dsa fill:#ffe8cc,stroke:#cc7000,color:#000
    classDef ffn fill:#fff2cc,stroke:#bf9000,color:#000
    class HP1,HQ1,HQ2 hc
    class IDX,MLA dsa
    class GATE,EXP,RED ffn
```

## Type 3 — KDA + MoE（31 層，最常見）

Attention 路徑跟 type 1 完全一樣、FFN 路徑跟 type 2 完全一樣：

```mermaid
flowchart TD
    IN["residual stream [s, 4x4096]"] --> HP1["hc_attn_pre"]
    HP1 --> KDA["KDA linear attention<br/>與 type 1 相同的 conv + gated delta rule<br/>沒有 KV cache、沒有 indexer、沒有 RoPE"]
    KDA --> AR1["o_proj + attn TP all-reduce"]
    AR1 --> HQ1["hc_post(attn) + hc_ffn_pre"]
    HQ1 --> MOE["MoE block<br/>與 type 2 相同: gate, 8/288 experts, shared expert, reduction"]
    MOE --> AR2["FFN TP all-reduce"] --> HQ2["hc_post(ffn)"] --> OUT["residual stream [s, 4x4096]"]

    classDef hc fill:#f0e6ff,stroke:#7733cc,color:#000
    classDef kda fill:#e3f0ff,stroke:#3366cc,color:#000
    classDef ffn fill:#fff2cc,stroke:#bf9000,color:#000
    class HP1,HQ1,HQ2 hc
    class KDA kda
    class MOE ffn
```

## 用 kernel 反推 layer type

看 trace 時最快的判斷方式：

| 看到的 kernel | 一定是 |
|---|---|
| `causal_conv1d_*`, `chunk_kda`, `fused_sigmoid_gating_delta_rule_update`, `FusedRMSNormGated` | KDA layer（type 1 或 3） |
| `fp8_mqa_logits` / `fp8_paged_mqa_logits`, `kpool_*_write_cache`, `topk_transform`, sparse MLA kernel | DSA layer（type 2） |
| `moe_sorting` / `moe1` / `moe2` / `moe_reduction`, router hgemm | MoE layer（layer >= 3） |
| `mhc_pre` / `mhc_post` / `mhc_fused_post_pre` | 每一層都有，一層 4 次（2 pre + 2 post） |
| 任何 RoPE kernel | **vision encoder**，不會是 text layer |

前 3 層沒有 MoE kernel、且是 KDA → type 1。有 MoE 又有 indexer → type 2。

## 參數量拆解（驗證 320B / 18B）

hidden = 4096，MoE expert intermediate = 2048，dense intermediate = 12288。

| 部位 | 計算 | 參數量 |
|---|---|---|
| Routed experts | 288 × 3 × 4096 × 2048 × 42 layer | ≈ 304.4 B |
| Shared expert | 1 × 3 × 4096 × 2048 × 42 layer | ≈ 1.06 B |
| Dense MLP | 3 × 4096 × 12288 × 3 layer | ≈ 0.45 B |
| KDA attention | ≈ 136 M × 34 layer | ≈ 4.6 B |
| MLA + indexer | ≈ 125 M × 11 layer | ≈ 1.4 B |
| mHC (`hc_*_fn` = 24 × 16384，一層兩份) | 0.39 M × 2 × 45 | ≈ 35 M |
| embed + lm_head | 2 × 154880 × 4096 | ≈ 1.27 B |
| Vision encoder | ≈ 16.8 M × 24 + merger | ≈ 0.4 B |
| **合計** | | **≈ 313 B**（官方標 320B，含 MTP layer 與雜項） |

Active per token：8 routed + 1 shared = 9 experts × 25.2 M × 42 = 9.5 B，
加上 dense MLP 0.45 B、所有 attention 6.0 B、embed/head 1.27 B → **≈ 17.3 B ≈ 18B**。

## 與 GLM-5.2 的逐項對照

| 項目 | GLM-5.2 | GLM-5.3-Flash |
|---|---|---|
| 層數 | 78 | 45（+1 MTP） |
| hidden | 7168 | 4096 |
| Attention | 全部 MLA + DSA | **hybrid**：34 KDA + 11 MLA/DSA |
| 有 indexer 的層 | 21 / 78（`index_topk_freq=4`，其餘層共用上一層的 top-k） | 11 / 45（`index_topk_freq=1`，每層各自算，無跨層共用） |
| Indexer 粒度 | per-token K，logits 寬 `kv_len` | **k-pool 4**：pooled K，logits 寬 `kv_len/4` |
| top-k 輸出寬度 | 2048 | **2051**（2048 + 最多 3 個 tail token） |
| RoPE | indexer 內有 RoPE（Wq/Wk 後） | **完全沒有**（NoPE，`qk_rope_head_dim=0`） |
| MLA head dim | qk_nope 128 + qk_rope 64, v 128 | qk_nope **256** + qk_rope **0**, v **256** |
| kv_lora_rank | 512 | 512（KV cache = 512/token/layer，因 rope dim = 0） |
| Residual | 單一條 + `add_rmsnorm_quant` | **mHC 4 條 stream** + `hc_pre` / `hc_post` |
| Experts | 256 routed, topk 8, 1 shared | **288** routed, topk 8, 1 shared |
| Expert 量化 | MXFP4（OneNexus 版） | FP8 e4m3 block 128×128（官方 ckpt） |
| 模態 | text only | text + image + video（24-layer vision encoder） |
| MTP | 無 | 1 層 nextn draft，`index_share_for_mtp_iteration` |
| Dense 層 | 0-2（`first_k_dense_replace=3`） | 0-2（同樣是 3） |
