---
name: sglang-glm5-fp8-decode-timeline
description: >-
  Build interactive decode-layer kernel timeline HTML for GLM-5-FP8 running
  through SGLang. Use when the user provides per-stream kernel CSVs (MI355X,
  B200, or other GPUs) for GLM-5-FP8 and wants an architecture diagram +
  multi-stream timeline visualization (MLA + NSA sparse attention + dual-stream
  MoE).
---

# Build Decode Layer Timeline Visualization

Given per-stream kernel trace CSVs and the SGLang model source, produce an
interactive HTML visualization with (1) an architecture diagram and
(2) a multi-stream kernel timeline — for **any** model and GPU combination.

Reference output: [glm5_fp8_decode_timeline.html](glm5_fp8_decode_timeline.html)

---

## Inputs You Will Receive

1. **Per-stream kernel CSVs** — one CSV per GPU stream, each with columns:

   ```
   #,ts_us,dur_us,stream,kernel_short,kernel_full
   ```

   - `ts_us`: absolute GPU timestamp in microseconds
   - `dur_us`: kernel duration in microseconds
   - `stream`: GPU stream ID (e.g. 113, 106, 8362)
   - `kernel_short`: abbreviated kernel name
   - `kernel_full`: full mangled kernel signature

   Typical file naming: `<GPU>_stream<ID>.csv` (e.g. `MI355X_stream113.csv`, `B200_stream8362.csv`)

2. **SGLang model source** — the `forward()` method of the decoder layer
   (e.g. `~/PR/sglang/python/sglang/srt/models/deepseek_v2.py` for DeepSeek-family).
   Read this to understand what each kernel does and which stream it runs on.

3. **User context** — model name, GPU, TP size, any special flags
   (e.g. "GLM-5-FP8, MI355X, TP=8, --disable-shared-experts-fusion")

---

## Step-by-Step Process

### Step 1: Normalize Timestamps

Raw `ts_us` values are absolute (e.g. 1575528942833). Convert to layer-relative:

- Find the earliest `ts_us` across ALL streams for this layer
- Subtract it from every kernel's `ts_us` so the layer starts at t=0
- Keep `dur_us` unchanged

### Step 2: Classify Kernels Into Categories

Map each kernel to a category using `kernel_short` and `kernel_full`. Use these patterns:

| Category | Match patterns in kernel_short / kernel_full |
|----------|----------------------------------------------|
| `gemm_ck` | `ABScale_GEMM`, `deep_gemm_fp8`, `bmm_E4m3`, `bmm_Bfloat16_MoE` |
| `gemm_rocblas` | `nvjet_sm100`, `Cijk`, `wv_splitk` |
| `gemm_hipblas` | `deep_gemm_bf16`, `hipblaslt`, `hipBLAS` |
| `attn` | `main_kernel` (TileLang MLA), `fmha_sm100_mla`, `batched_gemm_a8w8`, `fused_qk_rope`, `RoPE`, `RopeQuantize`, `fused_rope`, `paged_mqa` |
| `comm` | `cross_device_reduce`, `allreduce`, `allreduce_fusion` |
| `quant` | `fused_rms_fp8`, `dynamic_per_group_quant`, `per_token_group_quant`, `fused_flatten_fp8`, `act_quant` |
| `norm` | `rmsnorm`, `layernorm`, `RMSNormKernel`, `generalLayerNorm`, `add_rmsnorm_quant` |
| `moe` | `topk_transform`, `routingMain`, `routingIndices`, `activationDeepSeek`, `finalizeKernel`, `MoeSorting`, `grouped_topk` |
| `misc` | `elementwise`, `CatArrayBatchedCopy`, `memcpy`, `hadamard`, `act_and_mul`, `SiLU`, `fill`, `Memset`, `fused_store_indexer`, `set_mla_kv_buffer` |

When ambiguous, check `kernel_full` for more detail. This table covers both ROCm (CK, aiter, rocBLAS) and CUDA (DeepGEMM, nvjet, cuBLAS) kernels.

### Step 3: Assign Human-Readable Labels

Give each kernel bar a short label for display. Read the model source code to understand what each kernel computes. Map kernel_short → label using the execution order and source code context:

- `ABScale_GEMM (CK)` at position 0 in current_stream after norm → `fused_qkv_a_proj`
- `main_kernel` → `attn_kernel` (TileLang sparse MLA decode)
- `cross_device_reduce` → `allreduce`
- `add_rmsnorm_quant` → `residual+norm+quant`

The same kernel name (e.g. `ABScale_GEMM`) appears multiple times — distinguish by position in the stream execution order.

### Step 4: Identify Streams and Their Roles

Read the model's `forward()` to determine:
- Which streams exist (typically `current_stream` + `alt_stream`)
- What runs on each stream
- Where `fork` (`alt_stream.wait_stream(cur)`) and `join` (`cur.wait_stream(alt_stream)`) happen

Common patterns in SGLang DeepSeek-family models:
- **current_stream**: main path (QKV proj, attention core, shared MoE expert)
- **alt_stream**: parallel path (NSA indexer, MoE routing + routed experts)
- B200 may use **4 streams** for cross-layer pipelining (SBO)

### Step 5: Identify Phases and Idle Gaps

From the source code + kernel timestamps, group kernels into phases:
- **Serial sections**: kernels that must run sequentially (e.g. pre-attention, attention core)
- **Parallel sections**: dual-stream forks (e.g. q_b_proj ∥ NSA indexer, shared ∥ routed MoE)
- **Idle gaps**: where a stream has no kernels executing (waiting for the other stream at a sync point)

Compute idle gaps: if stream A has a kernel ending at t=91 and next kernel starts at t=218, that's a 127μs idle gap.

### Step 6: Read Model Source → Build Architecture Diagram

Read the model source code to extract:
- Layer structure (attention type, MoE config, etc.)
- Weight matrix shapes and dimensions
- Head counts and attention mechanism details
- Data flow between components

Build an SVG architecture diagram showing:
- Input/output tensor shapes
- Each computation block (GEMM, norm, attention, etc.) with dimensions
- Dual-stream fork/join points
- Color-coded by kernel category

### Step 7: Build the HTML

Use [glm5_fp8_decode_timeline.html](glm5_fp8_decode_timeline.html) as the template. The HTML has two tabs:

**Tab 1: Architecture Diagram** — SVG showing model structure, data flow, and dimensions.

**Tab 2: Stream Timeline** — built from JavaScript data arrays.

#### Timeline Data Format

Each stream's data is a JS array of kernel objects:

```javascript
const stream_data = [
  {ts: 1.5, dur: 3.8, l: 'elem_add (prev join)', c: 'misc', d: 'Combine shared+routed MoE outputs from prev layer'},
  {ts: 5.4, dur: 15.0, l: 'allreduce', c: 'comm', d: 'AllReduce MoE output across 8 TP ranks'},
  // ...
];
```

Fields:
- `ts`: layer-relative start time (μs) — from Step 1
- `dur`: duration (μs)
- `l`: human-readable label — from Step 3
- `c`: category key — from Step 2
- `d`: description of what this kernel computes (shown in hover tooltip) — derived from source code reading

#### Phase Bands

```javascript
const phases = [
  {s: 0, e: 66, l: 'Pre-Attention (serial)', bg: 'rgba(...)', lc: '#color'},
  {s: 66, e: 218, l: 'Attn Dual-Stream', bg: 'rgba(...)', lc: '#color'},
  // ...
];
```

#### Idle Markers

```javascript
const idle_markers = [
  {s: 91.3, e: 218.7, l: 'IDLE (waiting for indexer)'},
];
```

#### Overlap Boxes (for cross-stream overlaps)

```javascript
const overlaps = [
  {s: 224, e: 248, lbl: 'shared ∥ routed overlap (~24μs saved)', rows: [0,1,2,3]},
];
```

#### Stats Cards

```javascript
buildStats('stats-id', [
  {l: 'Total Layer', v: '~397', u: 'μs', co: 'var(--accent1)'},
  // ...
]);
```

#### Timeline Config

```javascript
buildTimeline({
  wrapId: 'mi-wrap',
  phaseId: 'mi-phases',
  axisId: 'mi-axis',
  ppu: 3.0,         // pixels per microsecond (scale factor)
  max: 400,         // max time axis in μs
  phaseHeight: 120,
  phases: phases,
  tracks: [
    {id: 'track-106', data: stream_data_106, idles: idle_106},
    {id: 'track-113', data: stream_data_113, idles: idle_113},
  ],
  overlaps: overlaps,  // optional
});
```

### Step 8: Compute Summary Stats

Calculate and display as stat cards:
- Total layer latency (max timestamp + duration across all streams)
- Per-phase durations
- Dual-stream savings (time saved by parallel execution vs serial)
- AllReduce total time
- Cross-layer overlap savings (if applicable)

---

## Color Scheme

```javascript
const catColors = {
  gemm_ck:      'var(--gemm-ck)',      // #22d3ee cyan
  gemm_rocblas: 'var(--gemm-rocblas)',  // #fb923c orange
  gemm_hipblas: 'var(--gemm-hipblas)',  // #fbbf24 yellow
  attn:         'var(--attn)',          // #818cf8 indigo
  comm:         'var(--comm)',          // #f87171 red
  quant:        'var(--quant)',         // #4ade80 green
  norm:         'var(--norm)',          // #a78bfa purple
  moe:          'var(--moe)',           // #e879f9 magenta
  misc:         'var(--misc)',          // #475569 gray
};
```

---

## Multiple GPUs in One HTML

When comparing two GPUs (e.g. MI355X vs B200), create separate timeline sections
in the same HTML file. Each section gets its own:
- Header with GPU name and stream info
- `buildTimeline()` call with its own config
- `buildStats()` call

See the reference HTML for how MI355X (2 streams) and B200 (4 streams) are placed
in the same page with a separator between them.

---

## Worked Example: GLM-5-FP8 (MI355X + B200)

The reference HTML [glm5_fp8_decode_timeline.html](glm5_fp8_decode_timeline.html) is a complete
worked example. Open it and study the `<script>` section (line ~442 onward) to see the exact
output format for every step of this process.

### What the example demonstrates

**Kernel data with all fields** — every kernel has `ts`, `dur`, `l`, `c`, and `d`.
Here are the first few entries from MI355X stream 113 (current_stream) showing how
raw CSV kernels become annotated timeline objects:

| CSV kernel_short | → label (`l`) | → description (`d`) |
|-----------------|---------------|---------------------|
| `elementwise` (CUDAFunctor_add) | `elem_add (prev join)` | Combine shared+routed MoE outputs from prev layer (line 578) |
| `cross_device_reduce` | `allreduce` | AllReduce MoE output across 8 TP ranks |
| `fused_rms_fp8_group_quant` | `rms+fp8_quant` | Fused: residual add + RMSNorm + FP8 quant for next GEMM |
| `ABScale_GEMM (CK)` #1 | `fused_qkv_a_proj (CK)` | (B,7168)@(7168,2112)→split(q_a[1536], kv_a[512], k_rope[64]) |
| `add_rmsnorm_quant` | `q_a / kv_a layernorm` | LayerNorm on q_a and kv_a latent vectors |
| `ABScale_GEMM (CK)` #2 | `q_b_proj (CK)` | Expand q_a→full Q: (B,1536)→(B,128h,192d=24576) |
| `batched_gemm_a8w8` #1 | `bmm w_kc (absorb)` | Absorb W_kc into Q_nope: (B,128,128)@W_kc→(B,128,512) |
| `main_kernel` #1 | `attn_kernel_1` | Full MLA decode attention (128 heads, full KV cache seq) |
| `main_kernel` #2 | `attn_kernel_2` | NSA sparse attention using top-16 block indices from indexer |
| `batched_gemm_a8w8` #2 | `bmm w_vc (absorb)` | Absorb W_vc: attn_out(512)×W_vc→V_out(128) per head |
| `ABScale_GEMM (CK)` #3 | `o_proj (CK)` | Output projection: (B,128h×128d=16384)→(B,7168) |
| `ABScale_GEMM (CK)` #4 | `shared gate_up (CK)` | Shared expert FFN gate+up: (B,7168)→(B,4096=2×2048) |
| `ABScale_GEMM (CK)` #5 | `shared down (CK)` | Shared expert FFN down: (B,2048)→(B,7168) |

Key insight: `ABScale_GEMM (CK)` appears 5+ times — only the **execution order** + **source code reading** distinguishes them.

**Architecture SVG** — the `<svg>` in tab-arch shows how to build a visual layer diagram with:
- Sections for each phase (pre-attention, dual-stream fork, attention core, MoE dual-stream)
- Weight matrix shapes and dimensions inside each GEMM block
- Head count annotations
- Fork/join arrows between streams
- Reference tables for head counts, KV cache layout, weight shapes

**Multi-GPU comparison** — shows MI355X (2 streams) and B200 (4 streams) in the same page,
each with different stream counts, phase structures, and overlap patterns.

**B200-specific patterns** — demonstrates:
- 4-stream cross-layer pipelining (SBO)
- Fused allreduce+residual (`allreduce_fusion`)
- SBO prep kernels (pre-computing next layer's qkv_a_proj during idle)
- Overlap boxes highlighting where parallel execution saves time

Use this HTML as both a reference for expected output quality and as a starting template
when building a new timeline for a different model.

---

## Checklist

- [ ] All stream CSVs loaded, timestamps normalized to layer-relative
- [ ] Every kernel classified into a category
- [ ] Every kernel has a human-readable label (derived from source code)
- [ ] Streams identified as current/alt with roles described
- [ ] Phases identified with start/end times
- [ ] Idle gaps computed for each stream
- [ ] Architecture SVG built from model source (shapes, dimensions, data flow)
- [ ] Timeline data arrays populated in the HTML template
- [ ] Stats cards computed (total latency, per-phase, savings)
- [ ] HTML opens in browser and all hover tooltips work
