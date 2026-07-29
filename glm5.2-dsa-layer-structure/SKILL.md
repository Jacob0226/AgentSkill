---
name: glm5.2-dsa-layer-structure
description: >-
  GLM-5.2 / GLM-5 DSA (DeepSeek Sparse Attention) decoder-layer structure: the
  three layer types (full-indexer + MLP, full-indexer + MoE, shared-indexer + MoE),
  which layer indices each covers, and per-type flow diagrams. Use when analyzing
  GLM-5.2 (or GLM-5 DSA) decode/prefill profiler traces — especially with
  analyze_trace.py / compare_step3.py — when asked how many layer types the
  model has, or when a layer appears to be "missing" the DSA indexer /
  sparse-attention kernels. Explains why the indexer top-k runs on only 21 of 78
  layers so you don't misreport it as a bug or attribution error.
---

# GLM-5.2 / GLM-5 DSA layer structure

**GLM5.2有shared/full layer，不用每個layer都算indexer。** Seeing the DSA indexer
top-k kernels on only *some* layers is EXPECTED — do not flag it as an
attribution bug in the trace or in `analyze_trace.py`.

GLM-5.2 has exactly **three** decoder-layer types:

| layer type | layers | count |
|---|---|---|
| **full-indexer + MLP** | 0, 1, 2 | 3 |
| **full-indexer + MoE** | 6, 10, 14, …, 74 | 18 |
| **shared-indexer + MoE** | 3-5, 7-9, …, 75-77 | 57 |

Per-type flow diagrams, measured per-layer costs, and what the two boundary layers
(first / last of the stack) do differently: [layer-flow.md](layer-flow.md).
Decode dataflow of the indexer itself (inputs + what's skipped at kv≤2048):
[indexer-flow.md](indexer-flow.md).

## Config facts (GLM-5.2-MXFP4, `config.json`)

- `num_hidden_layers = 78`
- `first_k_dense_replace = 3` → **3 dense layers** (`DeepseekV2MLP`) + **75 MoE
  layers** (`DeepseekV2MoE`). Inherited from DeepSeek-V3, which likewise keeps its
  first 3 FFNs dense; the V3 report only states the choice, and the usual rationale
  is that the earliest representations are barely differentiated, so routing there
  is close to random and destabilises the router.
- `index_topk = 2048` (sparse-attention top-k budget)
- `index_topk_freq = 4`, `index_skip_topk_offset = 3` → the **full/shared** split.

## Full vs shared layers (indexer top-k is periodic, not per-layer)

Source of truth: `sglang/srt/configs/model_config.py :: dsa_layer_skips_topk()`.
With `freq=4, offset=3`:

```
skips_topk(layer_id) = max(layer_id - offset + 1, 0) % freq != 0
                     = max(layer_id - 2, 0) % 4 != 0
```

- **Full layer** (computes fresh top-k): `layer_id ∈ {0, 1, 2, 6, 10, …, 74}` —
  **21 of 78**. The `max(…, 0)` clamp is easy to miss: layers 0 and 1 are full too,
  which is why all three dense layers are full-indexer layers.
- **Shared layer**: the other **57**. They run **no indexer kernels at all** — not
  even the K-cache write — and gather with the index list the previous full layer
  produced.

Confirmed on trace: an i8k prefill forward contains exactly 21 `topk_transform`,
`fp8_mqa_logits`, `hadamard_quant` and `indexer_k_quant_and_cache` launches, and 21
of each Indexer GEMM.

## Implication for analyze_trace.py / compare_step3.py

Both live under `~/SGLang-benchmarks/trace_analysis/` (`sglang/analyze_trace.py`,
`compare/compare_step3.py`); that folder's README documents the step3 schema.

- `analyze_trace.py` step3 groups layers by **the kernels they actually run** and
  aggregates each call site over all 78 layers, so `LaunchCount` (`LaunchCnt` in the
  side-by-side workbooks) is measured, not extrapolated: the indexer rows show 21,
  not 3. (Before Jul 2026 it sampled ONE representative layer per
  sub-module signature, which merged full and shared MoE layers and undercounted the
  indexer chain by 18 layers — 10.7 ms, 1.5% of a prefill forward.)
- The `LayerType` column lists which types reach a call site as tags — `full+MLP`,
  `full+MoE`, `shared+MoE`, or `all` — so the indexer rows read
  "full+MLP, full+MoE" with `LaunchCount = 21`; the workbook's **LayerTypes** sheet
  spells the tags out with their layer indices.
- When estimating a forward by hand, apply the indexer cost to **21 layers**, not to
  all 78 or all 75 MoE layers.
- Two count columns, do not mix them up: `LaunchCnt` belongs to the row (launches of
  that one call site, so `Σ_ms = Avg_us × LaunchCnt`), while `KernelCnt`/`KernelΣ_ms`
  belong to the kernel **name** across the whole forward and are repeated on every row
  sharing the name — never sum them. `KernelCnt` above the summed `LaunchCnt` of those
  rows means the kernel also runs outside the 78 decoder layers (lm_head, sampling).

## Second, independent skip: runtime logits skip (PR31324)

Do not confuse the structural full/shared split with the runtime skip in
`dsa_indexer.py :: _should_skip_logits_computation()`:

- When `max kv_len <= index_topk (2048)` (e.g. i1k), **eager** decode skips the
  indexer logits path (no `paged_mqa_logits`, no logits GEMM; `topk_transform`
  runs in cheap select-all mode). i8k (kv>2048) runs the full path.
- This per-step skip is the **eager** path (Milestone 1). Under a captured
  cuda-graph the same effect needs the dual-graph capture (branch can't be taken
  at replay).
- Net: at i1k a *full* layer's indexer drops from ~94µs to ~31µs; the remaining
  ~31µs is KV prep (proj + LayerNorm + RoPE + Hadamard + `indexer_k_quant_and_cache`
  + select-all `topk_transform`) which must still run to maintain the KV cache.

## PD disaggregation (1P1D) vs aggregated TP4 — decode latency wins, throughput halves

Experiment on MI355X (GLM-5.2-MXFP4, docker `v0.5.15.post1-rocm720-mi35x-20260714`),
identical workload `in=8192, out=1024, conc=64, num_prompts=320`, random-range-ratio 0.8.

- **Baseline (before)**: aggregated TP4 on GPU 0-3 (prefill+decode share the same GPUs).
- **1P1D (after)**: prefill server TP2 on GPU 0-1 + decode server TP2 on GPU 2-3,
  KV transfer via `mooncake_tcp`, `--dsa-prefill-backend triton --dsa-decode-backend triton`
  (`pd_1p1d.sh`).

Median latency (the metric asked for), before → after:

| Median | Aggregated TP4 (before) | 1P1D (after) | Δ |
|---|---|---|---|
| TTFT | 503.9 ms | 0.0 ms* | *median unreliable — see note |
| TPOT | 53.49 ms | 38.07 ms | **−28.8%** (decode faster) |
| ITL  | 27.54 ms | 17.66 ms | **−35.9%** (decode faster) |

Throughput / tail, before → after:

| | Aggregated TP4 | 1P1D | Δ |
|---|---|---|---|
| Request throughput | 1.24 req/s | 0.67 req/s | **−46%** |
| Output token throughput | 1138 tok/s | 619 tok/s | −46% |
| Mean TTFT | 3060 ms | 34063 ms | ~11× worse |
| P99 TTFT | 23901 ms | 259908 ms | ~11× worse |

**Takeaways:**
- Separating decode onto its own TP2 pair removes prefill↔decode contention on the
  decode GPUs → **median TPOT −29% and median ITL −36%** (decode is no longer
  preempted by the prefill-first scheduler; see the aggregated case where prefill
  runs back-to-back and starves decode).
- But with only **one TP2 prefill pair**, prefill becomes the bottleneck: TTFT
  balloons (Mean 3s → 34s, P99 24s → 260s) and **throughput halves** (1P prefill
  capacity < aggregated TP4). 1P1D trades throughput for decode-latency smoothness.
- \*The 1P1D **Median TTFT = 0.00 ms is a measurement artifact** of the PD/router
  streaming path (first-token timestamp), not a real 0. Use Mean/P90/P99 TTFT for
  the true (much worse) prefill latency.
- To keep decode-latency gains *without* losing throughput you need ≥2 prefill
  servers (e.g. 2P1D or multi-node), so prefill capacity matches aggregated TP4.

## Quick check

To see which GLM-5.2 layers are full vs shared for a given config:

```python
from sglang.srt.configs.model_config import dsa_layer_skips_topk  # freq/offset aware
# full layers = [i for i in range(num_hidden_layers) if not dsa_layer_skips_topk(cfg, i)]
```
Or just read `index_topk_freq` / `index_skip_topk_offset` / `index_topk_pattern`
from `config.json`.
