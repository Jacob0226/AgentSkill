---
name: glm5-dsa-layer-structure
description: >-
  GLM-5.2 / GLM-5 DSA (DeepSeek Sparse Attention) decoder-layer structure and
  the shared/full indexer-topk pattern. Use when analyzing GLM-5.2 (or GLM-5 DSA)
  decode/prefill profiler traces — especially with analyze_trace.py /
  compare_breakdown.py — and a layer or layer-type appears to be "missing" the
  DSA indexer / sparse-attention kernels. Explains why the indexer top-k does NOT
  run on every layer so you don't misreport it as a bug or attribution error.
---

# GLM-5.2 / GLM-5 DSA layer structure

**GLM5.2有shared/full layer，不用每個layer都算indexer。** Seeing the DSA indexer
top-k kernels on only *some* layers is EXPECTED — do not flag it as an
attribution bug in the trace or in `analyze_trace.py`.

## Config facts (GLM-5.2-MXFP4, `config.json`)

- `num_hidden_layers = 78`
- `first_k_dense_replace = 3` → **3 dense layers** (`DeepseekV2MLP`) + **75 MoE
  layers** (`DeepseekV2MoE`). This is what `analyze_trace.py` reports as its two
  layer types (A = dense, B = MoE).
- `index_topk = 2048` (sparse-attention top-k budget)
- `index_topk_freq = 4`, `index_skip_topk_offset = 3` → the **full/shared** split.

## Full vs shared layers (indexer top-k is periodic, not per-layer)

Source of truth: `sglang/srt/configs/model_config.py :: dsa_layer_skips_topk()`.
With `freq=4, offset=3`:

```
skips_topk(layer_id) = max(layer_id - offset + 1, 0) % freq != 0
                     = (layer_id - 2) % 4 != 0
```

- **Full layer** (computes fresh top-k): `layer_id ∈ {2, 6, 10, 14, ...}` (~1 in 4).
- **Shared layer** (reuses previous layer's top-k, skips the top-k selection):
  the other ~3/4 of layers.

So the expensive sparse-selection kernels — `paged_mqa_logits`, the logits GEMM
(`Cijk...`), and the real `topk_transform_decode` — run on **only ~1/4 of
layers**. (Every layer still instantiates an Indexer slot, but shared layers do
not recompute the selection.)

## Implication for analyze_trace.py / compare_breakdown.py

- `analyze_trace.py` groups decoder layers into 2 types by sub-module signature
  (dense vs MoE) and samples **ONE representative instance per type** for the
  kernel breakdown. That sample can be a *full* or a *shared* layer, so a layer
  type showing **0 (or few) indexer kernels is expected**, not a missing-kernel
  bug.
- Therefore the per-type attention Σduration differs by whether the sampled
  instance was full vs shared. When estimating a full 78-layer forward, remember
  the indexer-topk cost applies to only ~1/4 of layers, not all 75 MoE layers.

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

## Quick check

To see which GLM-5.2 layers are full vs shared for a given config:

```python
from sglang.srt.configs.model_config import dsa_layer_skips_topk  # freq/offset aware
# full layers = [i for i in range(num_hidden_layers) if not dsa_layer_skips_topk(cfg, i)]
```
Or just read `index_topk_freq` / `index_skip_topk_offset` / `index_topk_pattern`
from `config.json`.
