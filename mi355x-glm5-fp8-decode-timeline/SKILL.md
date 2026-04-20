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

Source code: `~/PR/sglang/python/sglang/srt/models/deepseek_v2.py`

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

## Kernel → Source Code Mapping

### Phase 1: Pre-Attention (serial, 0–66 μs) — `DeepseekV2DecoderLayer.forward`

| Kernel | Source |
|--------|--------|
| `elem_add` (prev join) | Residual add from previous layer's MoE output (`final_hidden_states += shared_output`) |
| `allreduce` | `tensor_model_parallel_all_reduce()` from previous layer's MoE |
| `rms+fp8_quant` | `layer_communicator.prepare_attn()` → `input_layernorm` (RMSNorm) + FP8 quant |
| `fused_qkv_a_proj` (CK) | `DeepseekV2AttentionMLA.prepare_qkv_latent()` → `self.fused_qkv_a_proj_with_mqa()` — fused q_a_proj + kv_a_proj_with_mqa, output dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim |

### Phase 2: Attn Dual-Stream (66–218 μs) — q_b_proj ∥ NSA Indexer

**Stream 113 (current_stream)** — `forward_absorb_prepare` / `forward_absorb_fused_mla_rope_prepare`:

| Kernel | Source |
|--------|--------|
| `q_a / kv_a layernorm` | `self.q_a_layernorm` (RMSNorm on q_lora_rank) + `self.kv_a_layernorm` (RMSNorm on kv_lora_rank) |
| `quant (q_b_proj)` | FP8 quantization of q_a before q_b_proj GEMM |
| `q_b_proj` (CK) | `self.q_b_proj` — ColumnParallelLinear: q_lora_rank → num_heads × qk_head_dim |
| *IDLE 127.4 μs* | `current_stream.wait_stream(alt_stream)` — waiting for Indexer to finish on S106 |

**Stream 106 (alt_stream)** — `Indexer` (NSA indexer, `layers/attention/nsa/nsa_indexer.py`):

| Kernel | Source |
|--------|--------|
| `indexer rmsnorm` | Indexer internal RMSNorm on latent |
| `indexer quant` | Indexer FP8 quantization |
| `wq_b GEMM` (CK) | Indexer q_b projection for index heads |
| `wk GEMM` (CK) | Indexer key projection |
| `k_norm` | Indexer key normalization |
| `RoPE (aiter)` | Indexer RoPE via aiter |
| `hadamard_q/k` | Hadamard transform on q and k for NSA |
| `act_quant_q/k` | Activation quantization for q and k |
| `store_k_cache` | Write compressed keys to KV cache |
| `to_copy (triton)` | Triton copy kernel |
| `weights_proj` (rocBLAS) | Indexer weight projection |
| `hipBLAS GEMM` | Indexer GEMM via hipBLASLt |
| `scale * mul` / `fill(-inf)` | Score scaling + masking |
| `paged_mqa_logits` | Paged MQA attention for index selection |
| `topk_transform` | Top-k selection for sparse attention indices |

### Phase 3: Attn Core + AllReduce (218–303 μs, serial) — MLA absorbed attention

| Kernel | Source |
|--------|--------|
| `bmm w_kc` (absorb) | Absorbed weight-key multiply: `w_kc @ kv_cache` (MLA weight absorption) |
| `qk_rope + cache` | RoPE on rope portion + KV cache store via `self.rotary_emb` |
| `CatBatchCopy` | Batch copy for paged attention |
| `attn_kernel_1/2` | `self.attn_mqa` (RadixAttention) — actual attention computation |
| `bmm w_vc` (absorb) | Absorbed weight-value multiply: `attn_output @ w_vc` |
| `fp8_quant` | FP8 quantization before o_proj |
| `o_proj` (CK) | `self.o_proj` — RowParallelLinear: num_heads × v_head_dim → hidden_size |
| `allreduce` | `tensor_model_parallel_all_reduce()` after attention (via `layer_communicator.prepare_mlp()`) |

### Phase 4: MoE Dual-Stream (303–394 μs) — `DeepseekV2MoE.forward_normal_dual_stream`

Entry: when `alt_stream is not None` and `num_fused_shared_experts == 0` and `get_is_capture_mode()`.

| Kernel | Source |
|--------|--------|
| `pre-MoE rmsnorm` | `layer_communicator.prepare_mlp()` → `post_attention_layernorm` (RMSNorm) |

**Stream 113 (current_stream)** — `_forward_shared_experts()` → `DeepseekV2MLP`:

| Kernel | Source |
|--------|--------|
| `shared quant` | FP8 quantization before shared expert gate_up_proj |
| `shared gate_up` (CK) | `self.shared_experts.gate_up_proj` — MergedColumnParallelLinear (gate + up fused) |
| `SiLU * gate` | `self.shared_experts.act_fn` (SiluAndMul) |
| `shared quant` | FP8 quantization before shared expert down_proj |
| `shared down` (CK) | `self.shared_experts.down_proj` — RowParallelLinear |
| *IDLE 43.9 μs* | `current_stream.wait_stream(alt_stream)` — waiting for routed experts on S106 |

**Stream 106 (alt_stream)** — routed experts (`with torch.cuda.stream(self.alt_stream)`):

| Kernel | Source |
|--------|--------|
| `wv_splitk` (router) | `self.gate()` → `MoEGate.forward()` → `aiter_dsv3_router_gemm()` on gfx95 |
| `grouped_topk` | `self.topk()` → `TopK.forward()` — biased grouped top-k selection |
| `MoeSorting` | MoE token dispatch/sorting inside `self.experts()` |
| `routed quant` | FP8 quantization before routed expert up proj |
| `Memset` | Zero-init output buffer |
| `routed up` (CK) | Routed expert gate_up GEMM (inside FusedMoE) |
| `SiLU * gate` | SiluAndMul activation |
| `routed quant` | FP8 quantization before routed expert down proj |
| `routed down` (CK) | Routed expert down GEMM (inside FusedMoE) |

**Join** (back on current_stream):

```python
current_stream.wait_stream(self.alt_stream)
final_hidden_states += shared_output  # elem_add (MoE join)
```

## Kernel Timing Tables

### Stream 113 (current_stream)

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

### Stream 106 (alt_stream)

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

| Category | Examples |
|----------|----------|
| GEMM (CK) | fused_qkv_a_proj, q_b_proj, o_proj, gate_up, down |
| GEMM (rocBLAS) | weights_proj |
| GEMM (hipBLASLt) | hipBLAS GEMM |
| Attention / RoPE | bmm_wkc, attn_kernel, RoPE, paged_mqa |
| AllReduce | allreduce (×2) |
| Quantize | fp8_quant, act_quant |
| Norm (RMS/Layer) | rmsnorm, layernorm, k_norm |
| MoE Routing | topk_transform, grouped_topk, MoeSorting |
| Misc / Elementwise | elem_add, SiLU, hadamard, CatBatchCopy |

## Interactive Visualization

Open [mi355x_decode_layer_timeline.html](mi355x_decode_layer_timeline.html) in a browser for the interactive dual-stream timeline with hover tooltips and phase annotations.

To create similar visualizations, read the HTML file as a template — it contains the CSS styling, JS rendering logic, and data format for kernel bars, idle zones, and phase bands.
