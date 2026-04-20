---
name: mi355x-glm5-fp8-decode-timeline
description: >-
  MI355X GLM-5-FP8 decode layer dual-stream kernel timeline reference. Use when
  analyzing MI355X GLM-5-FP8 decode performance, understanding dual-stream overlap
  (current_stream vs alt_stream), creating kernel timeline visualizations,
  or investigating decode layer latency breakdown.
---

# MI355X GLM-5-FP8 Decode Layer Timeline

Reference for the dual-stream kernel execution pattern in one GLM-5 decode layer on AMD MI355X (TP=8, FP8).

## Architecture Overview

Two HIP streams execute concurrently within each decode layer (~397 μs total):

| Stream | ID | Role |
|--------|----|------|
| `current_stream` | 113 | Main path: QKV projection, attention core, output projection, shared MoE |
| `alt_stream` | 106 | Parallel path: NSA indexer (q_b/wk projections, RoPE, paged MQA), MoE routing + routed experts |

## Phase Breakdown

| Phase | Time Range | Description |
|-------|-----------|-------------|
| Pre-Attention (serial) | 0–66 μs | `elem_add` → `allreduce` → `rms+fp8_quant` → `fused_qkv_a_proj` |
| Attn Dual-Stream | 66–218 μs | S113: `q_b_proj` ∥ S106: NSA indexer (wq_b, wk, RoPE, hadamard, paged_mqa, topk) |
| Attn Core + AllReduce | 218–303 μs | Serial: bmm_wkc → qk_rope → attn_kernel_1/2 → bmm_wvc → o_proj → allreduce |
| MoE Dual-Stream | 303–394 μs | S113: shared gate_up/down ∥ S106: routed gate_up/down + MoE sorting |

## Kernel Timing (Stream 113 — current_stream)

| Kernel | Start (μs) | Duration (μs) | Category |
|--------|-----------|---------------|----------|
| elem_add (prev join) | 1.5 | 3.8 | misc |
| allreduce | 5.4 | 15.0 | comm |
| rms+fp8_quant | 20.4 | 5.0 | quant |
| fused_qkv_a_proj (CK) | 25.4 | 40.7 | gemm_ck |
| q_a / kv_a layernorm | 66.1 | 4.4 | norm |
| quant (q_b_proj) | 79.7 | 2.5 | quant |
| q_b_proj (CK) | 82.3 | 9.1 | gemm_ck |
| *IDLE (waiting for S106 indexer)* | 91.3–218.7 | *127.4* | — |
| bmm w_kc (absorb) | 218.7 | 6.2 | attn |
| qk_rope + cache | 224.9 | 5.1 | attn |
| CatBatchCopy | 230.0 | 7.3 | misc |
| attn_kernel_1 | 237.3 | 10.6 | attn |
| attn_kernel_2 | 247.9 | 9.7 | attn |
| bmm w_vc (absorb) | 257.5 | 7.7 | attn |
| fp8_quant | 265.2 | 4.9 | quant |
| o_proj (CK) | 270.1 | 10.1 | gemm_ck |
| allreduce | 280.3 | 23.4 | comm |
| pre-MoE rmsnorm | 303.6 | 5.3 | norm |
| shared quant | 309.0 | 4.8 | quant |
| shared gate_up (CK) | 313.7 | 22.6 | gemm_ck |
| SiLU * gate | 336.4 | 4.4 | misc |
| shared quant | 340.7 | 4.4 | quant |
| shared down (CK) | 345.2 | 5.2 | gemm_ck |
| *IDLE (waiting for S106 routed)* | 350.3–394.2 | *43.9* | — |
| elem_add (MoE join) | 394.2 | 3.7 | misc |

## Kernel Timing (Stream 106 — alt_stream)

| Kernel | Start (μs) | Duration (μs) | Category |
|--------|-----------|---------------|----------|
| indexer rmsnorm | 72.1 | 2.1 | norm |
| indexer quant | 79.6 | 2.4 | quant |
| wq_b GEMM (CK) | 82.0 | 8.1 | gemm_ck |
| wk GEMM (CK) | 90.1 | 23.4 | gemm_ck |
| k_norm | 113.5 | 4.6 | norm |
| RoPE (aiter) | 118.1 | 9.7 | attn |
| hadamard_q | 127.8 | 4.7 | misc |
| hadamard_k | 132.5 | 4.8 | misc |
| act_quant_q | 137.3 | 6.0 | quant |
| act_quant_k | 143.3 | 4.6 | quant |
| store_k_cache | 147.9 | 4.8 | misc |
| to_copy (triton) | 152.8 | 4.6 | misc |
| weights_proj (rocBLAS) | 157.3 | 7.7 | gemm_rocblas |
| hipBLAS GEMM | 165.0 | 6.4 | gemm_hipblas |
| scale * mul | 171.4 | 4.9 | misc |
| fill(-inf) | 176.3 | 5.9 | misc |
| paged_mqa_logits | 182.2 | 9.9 | attn |
| topk_transform | 192.1 | 21.2 | moe |
| *IDLE (waiting for S113 attn core)* | 213.4–314.3 | *100.9* | — |
| wv_splitk (router) | 314.3 | 8.7 | moe |
| grouped_topk | 323.0 | 5.2 | moe |
| MoeSorting | 328.2 | 8.0 | moe |
| routed quant | 336.2 | 4.7 | quant |
| Memset | 340.8 | 4.9 | misc |
| routed up (CK) | 345.8 | 21.9 | gemm_ck |
| SiLU * gate | 367.7 | 5.0 | misc |
| routed quant | 372.6 | 4.9 | quant |
| routed down (CK) | 377.6 | 11.1 | gemm_ck |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total decode layer | ~397 μs |
| Attn dual-stream overlap (q_b_proj ∥ indexer) | 152 μs (saves ~12 μs) |
| Attn sequential (core + allreduce) | 85 μs |
| MoE dual-stream overlap (shared ∥ routed) | 91 μs (saves ~36 μs) |
| AllReduce total (serial) | 38 μs |
| **Dual-stream savings** | **~48 μs (≈11%)** |

## Kernel Categories

| Category | Color Code | Examples |
|----------|-----------|----------|
| GEMM (CK) | cyan | fused_qkv_a_proj, q_b_proj, o_proj, gate_up, down |
| GEMM (rocBLAS) | orange | weights_proj |
| GEMM (hipBLASLt) | yellow | hipBLAS GEMM |
| Attention / RoPE | indigo | bmm_wkc, attn_kernel, RoPE, paged_mqa |
| AllReduce | red | allreduce (x2) |
| Quantize | green | fp8_quant, act_quant |
| Norm (RMS/Layer) | purple | rmsnorm, layernorm, k_norm |
| MoE Routing | pink | topk_transform, grouped_topk, MoeSorting |
| Misc / Elementwise | gray | elem_add, SiLU, hadamard, CatBatchCopy |

## Interactive Visualization

Open [mi355x_decode_layer_timeline.html](mi355x_decode_layer_timeline.html) in a browser for the interactive dual-stream timeline with hover tooltips and phase annotations.
