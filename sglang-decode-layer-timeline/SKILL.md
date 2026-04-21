---
name: sglang-decode-layer-timeline
description: >-
  Build interactive decode-layer kernel timeline HTML from GPU trace CSVs and
  SGLang source code. Use when the user provides per-stream kernel CSVs
  (MI355X, B200, or other GPUs) and wants an architecture diagram + multi-stream
  timeline visualization for any model running through SGLang's decode path.
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
  {ts: 1.5, dur: 3.8, l: 'elem_add', c: 'misc'},
  {ts: 5.4, dur: 15.0, l: 'allreduce', c: 'comm'},
  // ...
];
```

Fields:
- `ts`: layer-relative start time (μs) — from Step 1
- `dur`: duration (μs)
- `l`: human-readable label — from Step 3
- `c`: category key — from Step 2

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
