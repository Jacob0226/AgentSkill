---
name: sglang-decode-layer-timeline
description: >-
  SGLang decode layer dual-stream kernel timeline reference. Use when analyzing
  decode performance on MI355X or B200, understanding dual-stream overlap
  (current_stream vs alt_stream), creating kernel timeline visualizations,
  or investigating decode layer latency breakdown for MLA+NSA+MoE models.
---

# SGLang Decode Layer Timeline

Reference for the dual-stream kernel execution pattern in SGLang's DeepSeek V2-family decode path.
Applies to models using MLA attention + NSA sparse attention + MoE (DeepSeek-V3, GLM-5, etc.).

Source code: `~/PR/sglang/python/sglang/srt/models/deepseek_v2.py`

---

## General Dual-Stream Execution Pattern

SGLang uses two GPU streams to overlap independent work within a single decode layer:

| Stream | Role |
|--------|------|
| `current_stream` | Main path: QKV projection, attention core, output projection, shared MoE expert |
| `alt_stream` | Parallel path: NSA indexer, MoE routing + routed experts |

### Phase Structure (all MLA+NSA+MoE models)

```
Phase 1: Pre-Attention (serial)
  residual_add → allreduce → RMSNorm → fused_qkv_a_proj → layernorms

Phase 2: Attention Dual-Stream FORK
  current_stream: q_b_proj → IDLE (waiting for indexer)
  alt_stream:     NSA indexer (wq_b, wk, RoPE, hadamard, paged_mqa, topk)

Phase 3: Attention Core (serial, after JOIN)
  bmm_wkc_absorb → RoPE+cache → attn_kernel_full → attn_kernel_sparse → bmm_wvc_absorb → o_proj

Phase 4: AllReduce + Norm (serial)
  allreduce → residual_add → RMSNorm

Phase 5: MoE Dual-Stream FORK
  current_stream: shared expert (gate_up → SiLU → down) → IDLE
  alt_stream:     router → topk → MoeSorting → routed experts (gate_up → SiLU → down)

Phase 6: Post-MoE JOIN
  current_stream.wait_stream(alt_stream)
  elem_add(shared + routed) → allreduce → next layer
```

Entry point for dual-stream MoE: `DeepseekV2MoE.forward_normal_dual_stream` (when `alt_stream is not None` and `get_is_capture_mode()`).

### Kernel Categories

| Category | Color (in viz) | Examples |
|----------|----------------|----------|
| GEMM (CK / DeepGEMM) | cyan | fused_qkv_a_proj, q_b_proj, o_proj, gate_up, down |
| GEMM (rocBLAS / nvjet) | orange | weights_proj, logits |
| GEMM (hipBLAS / DeepGEMM bf16) | yellow | hipBLAS GEMM, wq_b bf16 |
| Attention / RoPE | indigo | bmm_wkc, attn_kernel, RoPE, paged_mqa |
| AllReduce | red | allreduce (×2 per layer) |
| Quantize / FP8 | green | fp8_quant, act_quant |
| Norm (RMS/Layer) | purple | rmsnorm, layernorm, k_norm |
| MoE Routing | magenta | topk_transform, grouped_topk, MoeSorting |
| Misc / Elementwise | gray | elem_add, SiLU, hadamard, CatBatchCopy |

---

## Model Reference: GLM-5-FP8 (TP=8)

Interactive visualization: [glm5_fp8_decode_timeline.html](glm5_fp8_decode_timeline.html)

### Architecture — Key Dimensions

| Component | Shape / Value | Notes |
|-----------|--------------|-------|
| hidden_size | 7168 | Input/output per layer |
| fused_qkv_a_proj | (7168, 2112) | → split into q_a(1536) + kv_a(512) + k_rope(64) |
| q_b_proj | (1536, 24576) | q_a → Q: 128 heads × 192 dim (128 nope + 64 rope) |
| wq_b (indexer) | (1536, 1024) | q_a → indexer Q: 8 heads × 128 dim |
| wk (indexer) | (7168, 128) | hidden → indexer K: 1 head × 128 dim (MQA) |
| W_kc (absorb) | (128, 128, 512) | per-head: Q_nope → Q' for latent KV |
| W_vc (absorb) | (128, 128, 512) | per-head: attn_out → V_out |
| o_proj | (16384, 7168) | 128h×128d → hidden |
| gate_up (shared) | (7168, 4096) | hidden → gate+up (2×2048) |
| down (shared) | (2048, 7168) | intermediate → hidden |
| gate (router) | (7168, 256) | hidden → expert scores |
| MLA heads | 128 | Full attention |
| NSA indexer Q heads | 8 | Coarse block selection |
| NSA indexer K heads | 1 | MQA: 8 Q heads share 1 K |
| KV cache per token | 704 (576+128) | 576 MLA latent + 128 indexer K (46× compression vs MHA) |
| MoE experts | 256 total, top-8 | Each: gate(7168→2048), up(7168→2048), down(2048→7168) |

### MI355X Timeline (TP=8, ~397 μs/layer)

Streams: 113 (current) + 106 (alt), layers execute serially (no cross-layer overlap).

#### Phase Breakdown

| Phase | Time Range | Duration |
|-------|-----------|----------|
| Pre-Attention (serial) | 0–66 μs | 66 μs |
| Attn Dual-Stream (q_b_proj ∥ indexer) | 66–218 μs | 152 μs |
| Attn Core + AllReduce (serial) | 218–303 μs | 85 μs |
| MoE Dual-Stream (shared ∥ routed) | 303–394 μs | 91 μs |

#### Stream 113 (current_stream)

| Kernel | Start (μs) | Duration (μs) | Category |
|--------|-----------|---------------|----------|
| elem_add (prev join) | 1.5 | 3.8 | misc |
| allreduce | 5.4 | 15.0 | comm |
| rms+fp8_quant | 20.4 | 5.0 | quant |
| fused_qkv_a_proj (CK) | 25.4 | 40.7 | gemm_ck |
| q_a / kv_a layernorm | 66.1 | 4.4 | norm |
| quant (q_b_proj) | 79.7 | 2.5 | quant |
| q_b_proj (CK) | 82.3 | 9.1 | gemm_ck |
| *IDLE (waiting for indexer)* | 91.3–218.7 | *127.4* | — |
| bmm w_kc (absorb) | 218.7 | 6.2 | attn |
| qk_rope + cache | 224.9 | 5.1 | attn |
| CatBatchCopy | 230.0 | 7.3 | misc |
| attn_kernel_1 (full MLA) | 237.3 | 10.6 | attn |
| attn_kernel_2 (NSA sparse) | 247.9 | 9.7 | attn |
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
| *IDLE (waiting for routed)* | 350.3–394.2 | *43.9* | — |
| elem_add (MoE join) | 394.2 | 3.7 | misc |

#### Stream 106 (alt_stream)

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
| *IDLE (waiting for attn core)* | 213.4–314.3 | *100.9* | — |
| wv_splitk (router) | 314.3 | 8.7 | moe |
| grouped_topk | 323.0 | 5.2 | moe |
| MoeSorting | 328.2 | 8.0 | moe |
| routed quant | 336.2 | 4.7 | quant |
| Memset | 340.8 | 4.9 | misc |
| routed up (CK) | 345.8 | 21.9 | gemm_ck |
| SiLU * gate | 367.7 | 5.0 | misc |
| routed quant | 372.6 | 4.9 | quant |
| routed down (CK) | 377.6 | 11.1 | gemm_ck |

#### Kernel → Source Code Mapping

**Phase 1: Pre-Attention** — `DeepseekV2DecoderLayer.forward`:

| Kernel | Source |
|--------|--------|
| `elem_add` | Residual add from previous layer (`final_hidden_states += shared_output`) |
| `allreduce` | `tensor_model_parallel_all_reduce()` from previous layer's MoE |
| `rms+fp8_quant` | `layer_communicator.prepare_attn()` → `input_layernorm` (RMSNorm) + FP8 quant |
| `fused_qkv_a_proj` | `DeepseekV2AttentionMLA.prepare_qkv_latent()` → `self.fused_qkv_a_proj_with_mqa()` |

**Phase 2: current_stream** — `forward_absorb_prepare`:

| Kernel | Source |
|--------|--------|
| `q_a/kv_a layernorm` | `self.q_a_layernorm` + `self.kv_a_layernorm` |
| `q_b_proj` | `self.q_b_proj` — ColumnParallelLinear: q_lora_rank → num_heads × qk_head_dim |

**Phase 2: alt_stream** — `Indexer` (`layers/attention/nsa/nsa_indexer.py`):

| Kernel | Source |
|--------|--------|
| `wq_b/wk` | Indexer Q/K projections |
| `RoPE (aiter)` | Indexer RoPE |
| `hadamard_q/k` | Hadamard transform for NSA |
| `paged_mqa_logits` | Paged MQA attention for index selection |
| `topk_transform` | Top-k selection for sparse attention block indices |

**Phase 3: Attn Core** — MLA absorbed attention:

| Kernel | Source |
|--------|--------|
| `bmm w_kc/w_vc` | MLA weight absorption |
| `attn_kernel_1` | Full MLA attention (all tokens) |
| `attn_kernel_2` | NSA sparse attention (top-16 blocks, 2048 tokens) |
| `o_proj` | `self.o_proj` — RowParallelLinear |

**Phase 4: MoE** — `DeepseekV2MoE.forward_normal_dual_stream`:

| Kernel | Source |
|--------|--------|
| `shared gate_up/down` | `self.shared_experts` (DeepseekV2MLP) on current_stream |
| `wv_splitk (router)` | `self.gate()` → `aiter_dsv3_router_gemm()` on alt_stream |
| `MoeSorting` | Token dispatch inside `self.experts()` |
| `routed up/down` | FusedMoE routed expert GEMMs on alt_stream |

#### MI355X Performance Summary

| Metric | Value |
|--------|-------|
| Total decode layer | ~397 μs |
| Attn dual-stream (q_b_proj ∥ indexer) | 152 μs (saves ~12 μs) |
| Attn sequential (core + allreduce) | 85 μs |
| MoE dual-stream (shared ∥ routed) | 91 μs (saves ~36 μs) |
| AllReduce total | 38 μs |
| **Dual-stream savings** | **~48 μs (≈11%)** |
| Cross-layer overlap | 0 μs (none) |

---

### B200 Timeline (TP=8, ~234 μs pipeline period)

Streams: 8362/8363 (Layer N) + 8364/8365 (Layer N+1). Uses 4 streams with cross-layer SBO (Stream-Based Overlap).

Key differences from MI355X:
- **4 streams** instead of 2 (cross-layer pipelining)
- **SBO**: Layer N's shared MoE overlaps with Layer N+1's routed MoE start
- **Fused AR+residual**: AllReduce and residual add are fused
- **DeepGEMM**: Uses DeepGEMM (FP8) + nvjet instead of CK + rocBLAS

#### Phase Breakdown

| Phase | Time Range | Description |
|-------|-----------|-------------|
| Prev Routed MoE | 0–62 μs | Tail of previous layer's routed MoE on s8362 |
| Attn: q_b_proj ∥ Indexer | 102–203 μs | s8362 main + s8363 indexer |
| Shared MoE | 203–248 μs | Shared expert on s8362 |
| Routed MoE (s8364) | 224–290 μs | Next layer routed MoE starts overlapping |
| Join+AR(+res) | 290–311 μs | MoE join + fused AllReduce+residual |
| SBO prep | 311–351 μs | Prepare next layer while N+1 attn starts |
| N+1 Attn ∥ Indexer | 335–435 μs | Layer N+1 attention dual-stream |
| N+1 Shared ∥ Routed | 435–525 μs | Layer N+1 MoE dual-stream |

#### Stream 8362 (Layer N main)

| Kernel | Start (μs) | Duration (μs) | Category |
|--------|-----------|---------------|----------|
| router (splitK) | 0.0 | 4.4 | moe |
| splitK reduce | 4.0 | 2.0 | moe |
| routing prep | 6.5 | 9.6 | misc |
| routingMain | 16.1 | 4.5 | moe |
| routingIndices | 17.8 | 5.2 | moe |
| MoE stage1 GEMM | 23.0 | 19.2 | gemm_ck |
| SiLU | 39.9 | 4.0 | moe |
| MoE stage2 GEMM | 41.1 | 15.2 | gemm_ck |
| finalize | 51.9 | 7.6 | moe |
| memcpy | 59.9 | 1.6 | misc |
| *IDLE (allreduce)* | 61.5–102 | *40.5* | — |
| RMSNorm (input) | 102.0 | 2.2 | norm |
| FP8 quant | 123.8 | 2.3 | quant |
| qkv_a_proj (DeepGEMM) | 126.6 | 12.5 | gemm_ck |
| kv_a_norm | 139.4 | 2.8 | norm |
| hadamard | 155.8 | 1.2 | misc |
| store_indexer | 157.1 | 2.4 | misc |
| logits (nvjet) | 182.5 | 2.8 | gemm_rocblas |
| RoPE+FP8 | 185.6 | 2.3 | attn |
| set_kv | 188.2 | 1.5 | misc |
| fmha_sm100_mla | 189.9 | 9.4 | attn |
| o_proj (nvjet) | 199.0 | 4.0 | gemm_rocblas |
| FP8 quant | 203.3 | 2.4 | quant |
| shared gate_up | 206.2 | 10.4 | gemm_ck |
| attn AR | 216.8 | 6.8 | comm |
| FP8 quant | 223.9 | 2.5 | quant |
| shared GEMM | 226.9 | 10.4 | gemm_ck |
| SiLU×gate | 237.9 | 2.5 | misc |
| FP8 quant | 240.9 | 2.1 | quant |
| shared down | 243.4 | 4.4 | gemm_ck |
| *IDLE (waiting for routed on s8364)* | 247.9–290.4 | *42.5* | — |
| MoE join (s+r) | 290.4 | 2.0 | misc |
| AR+residual (fused) | 292.4 | 18.1 | comm |
| FP8 quant | 310.9 | 2.5 | quant |
| qkv_a_proj (N+1) | 313.7 | 21.0 | gemm_ck |
| RMSNorm | 335.4 | 2.6 | norm |
| FP8 quant | 338.6 | 2.3 | quant |
| GEMM (N+1) | 341.4 | 10.0 | gemm_ck |

#### Stream 8363 (Layer N alt / indexer)

| Kernel | Start (μs) | Duration (μs) | Category |
|--------|-----------|---------------|----------|
| wq_b (DeepGEMM bf16) | 105.3 | 15.2 | gemm_hipblas |
| fused_mul (triton) | 120.8 | 2.3 | misc |
| FP8 quant | 123.6 | 2.2 | quant |
| wk (DeepGEMM fp8) | 126.1 | 9.8 | gemm_ck |
| fused_rope | 142.4 | 2.4 | attn |
| elementwise ×4 | 145.4 | 8.6 | misc |
| hadamard | 155.5 | 1.8 | misc |
| act_quant | 158.0 | 2.2 | quant |
| scale mul | 160.4 | 3.2 | misc |
| paged_mqa | 164.3 | 5.0 | attn |
| topk_transform | 169.9 | 12.3 | moe |

#### Stream 8364 (Layer N+1 main) & Stream 8365 (Layer N+1 alt)

These mirror 8362/8363 with ~224 μs offset (pipeline period). See interactive visualization for full data.

#### B200 Performance Summary

| Metric | Value |
|--------|-------|
| Pipeline period (RMSNorm→RMSNorm) | ~234 μs |
| Attn dual-stream (q_b_proj ∥ indexer) | ~98 μs |
| Shared ∥ Routed MoE overlap | ~24 μs saved |
| SBO tail ∥ N+1 start | ~16 μs saved (B200 only) |
| AllReduce total (2×) | ~25 μs |
| Cross-layer overlap | Yes (4-stream SBO) |

---

## MI355X vs B200 Comparison

| Aspect | MI355X | B200 |
|--------|--------|------|
| Layer latency | ~397 μs | ~234 μs (pipeline period) |
| Streams per layer | 2 | 4 (cross-layer) |
| Cross-layer overlap | None | SBO (~16 μs saved) |
| fused_qkv_a_proj | 40.7 μs (CK) | 12.5 μs (DeepGEMM) |
| q_b_proj | 9.1 μs (CK) | 15.2 μs (DeepGEMM bf16) |
| Attention core | 10.6+9.7 μs | 9.4 μs (fmha_sm100_mla) |
| AllReduce | 15.0+23.4 μs | 6.8+18.1 μs (fused AR+residual) |
| NSA indexer total | ~141 μs | ~77 μs |
| Shared MoE | 47 μs | ~40 μs |
| Routed MoE | 74 μs | ~62 μs |

---

## Adding New Model Data

To add timeline data for a new model:

1. Profile with `~/SGLang-benchmarks/GLM.sh --prof` (or equivalent)
2. Run `analyze_trace.py` to get per-layer kernel breakdown CSVs
3. Create `<model>_decode_timeline.html` from the existing HTML template:
   - Update kernel data arrays (timestamps, durations, labels, categories)
   - Update phase definitions
   - Update stream IDs and idle markers
   - Update stats
4. Add a new `### <Model> Timeline` section to this SKILL.md with the kernel tables
5. Update the architecture dimensions if different from GLM-5
