# GLM-5.2 / DSA lightning-indexer decode flow

Per decode token, per DSA layer. The indexer picks the top-`index_topk` (2048)
KV positions for the main MLA sparse attention.

- Input: `hidden state x` (bf16) + `positions`.
- Indexer q/k projections are **bf16** GEMMs (`hgemm_bf16_...`); only the logits
  input is quantized to **fp8** (`act_quant`). NOT FP4 (unlike the MoE experts).
- 🟩 green = always runs · 🟥 red = skipped when `kv_len ≤ index_topk (2048)`
  (the eager k-only fast path, `_forward_cuda_k_only`; PR31324).

```mermaid
flowchart TD
    X["INPUT: hidden state x (bf16) + positions"]

    X --> KP["Wk proj (bf16 hgemm)"]
    KP --> KN["LayerNorm"] --> KR["RoPE"] --> KH["Hadamard rotate #1"] --> KC["indexer_k_quant_and_cache<br/>write fp8 K to KV cache"]

    X --> QP["Wq proj (bf16 hgemm)"]
    QP --> QN["LayerNorm"] --> QR["RoPE"] --> QH["Hadamard rotate #2"] --> QQ["act_quant -> fp8"]

    QQ --> LOG["logits = q . K_cache^T<br/>paged_mqa_logits + GEMM<br/>O(kv_len), reads all K history"]
    KC --> LOG
    LOG --> TK["real top-2048 select"]
    TK --> IDX["top-2048 page-slot indices"]
    KC -.->|"kv_len <= 2048"| SEL["select-all<br/>topk_transform(dummy)=[0..len-1]"]
    SEL --> IDX
    IDX --> MLA["MLA sparse attention<br/>_sparse_mla_decode_split_kernel<br/>gather selected KV + attend"]

    classDef skip fill:#ffdddd,stroke:#cc0000,color:#000
    classDef keep fill:#ddffdd,stroke:#00aa00,color:#000
    class QP,QN,QR,QH,QQ,LOG,TK skip
    class KP,KN,KR,KH,KC,SEL,MLA keep
```

## What can / cannot be skipped when kv_len ≤ 2048

| Part | Kernels | kv ≤ 2048 |
|------|---------|-----------|
| Key build + cache | Wk proj, LayerNorm, RoPE, Hadamard #1, `indexer_k_quant_and_cache` | **must run** — K for this token must be cached; later steps (kv>2048) score all history |
| Index list for gather | `topk_transform` | **must run** (cheap select-all `[0..len-1]`; MHA path needs none) |
| Query build | Wq proj, LayerNorm, RoPE, Hadamard #2, `act_quant` | **skipped** |
| Scoring | `paged_mqa_logits` + logits GEMM | **skipped** |
| Selection | real top-k | **skipped** (→ select-all) |

Hadamard count is the tell: **i1k = 1** (K only), **i8k = 2** (Q + K) — assuming
`SGLANG_DSA_FUSE_HADAMARD_QUANT` is off (fused → 0 separate hadamard kernels).
Full-layer indexer ≈ 94 µs; k-only ≈ 31 µs.
