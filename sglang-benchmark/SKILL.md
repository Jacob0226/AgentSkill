---
name: sglang-benchmark
description: >-
  SGLang development, benchmarking, and profiling workflow on AMD MI355X (ROCm)
  and NVIDIA B200 (CUDA). Use when working with SGLang code, running benchmarks,
  profiling GPU kernels, analyzing traces, comparing MI355X vs B200 performance,
  or editing GLM.sh and related scripts.
---

# SGLang Benchmark & Profile Workflow

## Development Workflow

### Directory Layout

| Location | Path | Purpose |
|----------|------|---------|
| Host (edit here) | `~/PR/sglang` | Fork: `Jacob0226/sglang`. Make all code changes here. |
| Container (run here) | `~/work-space/sglang` | Checkout the branch from the fork, then run server/benchmarks. |
| Benchmarks | `~/SGLang-benchmarks/` | Benchmark scripts, profiler parsers, analysis tools. |
| Reusable helper scripts | `~/SGLang-benchmarks/tools/` | **Micro-benchmarks, stream/graph tests, env setup go here** (e.g. `glm5_decode_layer.py`, `glm5_proposalA_v3_test.py`, `glm5_proposalA_overlap_analysis.py`, `mi355x_decode_layer_timeline.html`). |
| Trace analysis | `~/SGLang-benchmarks/trace_analysis/` | **Anything reading a profiler trace goes here**: `sglang/` + `atom/` analyzers, `compare/` workbooks, `diagnostics/`. See its README. |
| Scratch / ad-hoc outputs | `~/SGLang-benchmarks/tmp/` | **Outputs only** — trace dumps, one-off HTMLs produced by tool runs, temporary CSVs. Do NOT put executable helper scripts here; any new script should live under `tools/`. Same host↔container path (`/home/jacchang/SGLang-benchmarks/tmp/` is mounted into the container). |

### Edit → Run Cycle

1. Edit code on **host** in `~/PR/sglang` (branch: `jacob/*`)
2. Commit & push to `origin` (Jacob0226/sglang)
3. Inside the Docker container, `cd ~/work-space/sglang && git fetch origin && git checkout <branch>`
4. Run benchmarks inside the container with `~/SGLang-benchmarks/GLM.sh`

### SGLang Fork Info

- Remote: `git@github.com:Jacob0226/sglang.git`
- Typical branches: `jacob/hip-dual-stream-glm5`, `jacob/topk-smem-optimize`, etc.

---

## Target GPUs

Two primary platforms. For detailed specs see [gpu-specs.md](gpu-specs.md).

| | AMD MI355X | NVIDIA B200 |
|---|---|---|
| Architecture | CDNA 4 | Blackwell |
| Memory | 288 GB HBM3e | 192 GB HBM3e |
| FP8 Peak | 2.3 PFLOPS | 4.5 PFLOPS (dense), 9 PFLOPS (sparse) |
| Software | ROCm 7.2+ | CUDA 13+ |
| Docker | `rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-*` | `lmsysorg/sglang:v0.5.9-cu130-runtime` |

### Platform-Specific Server Args

**MI355X (ROCm)**:
- `--nsa-prefill-backend tilelang --nsa-decode-backend tilelang`
- `--disable-shared-experts-fusion` (for dual-stream)
- Env: `SGLANG_ROCM_FUSED_DECODE_MLA=0`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`

**B200 (CUDA)**:
- `--quantization fp8 --attention-backend nsa`
- `--nsa-prefill-backend trtllm --nsa-decode-backend trtllm`
- `--moe-runner-backend flashinfer_trtllm`
- `--chunked-prefill-size 32768 --max-prefill-tokens 32768`
- `--enable-flashinfer-allreduce-fusion`
- Env: `SGL_ENABLE_JIT_DEEPGEMM=1` (for GLM-5)

---

## Running Benchmarks (GLM.sh)

Script: `~/SGLang-benchmarks/GLM.sh`

### Common Invocations

```bash
# Benchmark only
./GLM.sh

# With MTP (speculative decoding)
./GLM.sh --mtp

# Profile (prefill/decode separated)
./GLM.sh --prof

# Profile (combined, no stage split)
./GLM.sh --prof-combined

# Dual stream on ROCm (disables shared-experts-fusion)
./GLM.sh --dual-stream-rocm

# Custom model path
./GLM.sh --model /data/huggingface/hub/zai-org/GLM-5-FP8
```

### What GLM.sh Does

1. Starts SGLang server (`sglang.launch_server`) with TP=8, FP8 KV cache, model-specific args
2. Warmup run (2048 in / 256 out, concurrency 4, 8 prompts)
3. Accuracy test (GSM8K, 1200 questions) — skipped in profile mode
4. Benchmark loop: iterates `in_out_tokens × concurrencies`
   - Default: `8192:1024`, `1024:1024` × concurrency `4,8,16,32,64`
   - Profile mode: concurrency `4` only, `PROMPT_MULTIPLIER=2`
5. Profile mode additionally runs with `--disable-cuda-graph` for layer-structure traces

### Output Structure

```
~/SGLang-benchmarks/results/<docker_tag>/<model><flags>/
├── server_*.log
├── warmup.log
├── Accuracy_GSM8K.log
├── bench_in8192_out1024_conc4.log
├── bench_in8192_out1024_conc8.log
├── ...
├── prof_in8192_out1024_conc4_p8/       # profiler traces
│   └── *-TP-*-{EXTEND,DECODE}.trace.json.gz
├── no-cuda-graph/                      # no-graph profiler traces
│   └── prof_in*_out*_conc*_p*/
└── Finish.log                          # tracks completed runs (skip on re-run)
```

---

## Analysis Tools

Paths are relative to `~/SGLang-benchmarks/`. For detailed usage see
[analysis-tools.md](analysis-tools.md), and `trace_analysis/README.md` in the repo for
the step3 workbook schema and the two comparison flows.
For ROCm HIP stream/event/graph programming model, see [rocm-streams.md](rocm-streams.md).

| Script | Purpose |
|--------|---------|
| `trace_analysis/sglang/analyze_trace.py` | **Main analysis**: 3-step trace analysis (kernel stats → layer structure → per-call-site breakdown). Always pass `--forward-match`. |
| `trace_analysis/compare/compare_breakdown.py` | Compare two step3 workbooks side-by-side (e.g. MI355X vs B200) |
| `trace_analysis/atom/analyze_atom_trace.py` | Same step3 schema for an ATOM trace (SGLang vs ATOM flow) |
| `trace_analysis/diagnostics/` | `verify_step3_against_trace.py`, `window_kernel_diff.py`, `comm_skew_split.py`, `trace_kernel_summary.py`, `extract_stream.py`, `analyze_trace_overlap.py` |
| `parse_perf_metrics_to_csv.py` | Parse `sglang.bench_serving` result blocks → CSV |

### Quick Analysis Workflow

```bash
# 1. Parse benchmark latencies
python3 ~/SGLang-benchmarks/parse_perf_metrics_to_csv.py --root-dir <results_dir>

# 2. Full trace analysis (graph-on + graph-off)
python3 ~/SGLang-benchmarks/trace_analysis/sglang/analyze_trace.py \
    --graph-on <graph-on-trace.json.gz> \
    --graph-off <graph-off-trace.json.gz> \
    --forward-match "DECODE bs=64"

# 3. Compare MI355X vs B200
python3 ~/SGLang-benchmarks/trace_analysis/compare/compare_breakdown.py \
    --file-a MI355X/step3_layer_breakdown.xlsx \
    --file-b B200/step3_layer_breakdown.xlsx \
    --labels MI355X B200 --out comparison.xlsx
```

---

## HIP Graph dual-stream gotchas (two rules that bite together)

Two independent rules both matter when a decode layer is captured in a HIP graph; mis-
applying either one costs 15-25 μs per layer.

### Rule 1 — dispatch-order picks the physical stream

The replay scheduler decides which physical stream each branch lands on based on **capture
(dispatch) order during graph capture**, not on which `torch.cuda.Stream` object you passed.

> At each fork point, the branch whose first kernel is captured FIRST keeps the current
> physical stream (e.g. `phys 0`). The later-captured branch is moved to a fresh physical
> stream (`phys 4/5/8/…`).

`alt_stream.wait_stream(cur)` and explicit `record_event`/`wait_event` pairs behave
identically here — the sync primitive does not influence the choice.

Symptom: kernels you wrote inside `with torch.cuda.stream(alt):` show up on `phys 0` in
the trace while kernels dispatched on `cur` land on `phys 4`. Looks "remapped"; isn't.

Fix: dispatch the cur-side chain *before* the `with torch.cuda.stream(alt):` block.

### Rule 2 — `wait_stream` position decides when alt starts running

`alt_stream.wait_stream(cur)` records a CUDA event at `cur`'s **current** state. Anything
dispatched on `alt` afterwards waits only for that snapshot of `cur`, not for future cur
ops.  So if you place the `wait_stream` call AFTER a heavy `cur` kernel that alt does not
actually need (e.g. `q_b_proj`), alt is delayed for that kernel's duration even though
there is no real data dependency.

Symptom in trace: `phys_aux` is idle for 15-20 μs after phase1 ends, even though alt work
only needs phase1's outputs. Net result: alt finishes later, attn starts later, whole
iteration is slower.

Fix: place `wait_stream` BEFORE any cur kernel alt doesn't actually need.

### Combined "clean + fast" pattern (GLM-5 NSA decode)

```python
cur = torch.cuda.current_stream()
h_normed, _, q_lat, kv_lat, h_fp8, h_scale = phase1_pre_attention(...)

alt_stream.wait_stream(cur)        # (Rule 2) BEFORE q_b_proj — alt only waits for phase1

q, q_lora = phase2_current(q_lat)  # (Rule 1) dispatched first on cur → cur stays on phys 0

# Gap-fill on cur, still before the with-stream(alt) block, so cur chain is captured first.
kv_normed = aiter_rms_norm(kv_lat, ...)
q_all = phase3_pre_wait(q, kv_normed, ..., kv_cache)

with torch.cuda.stream(alt_stream):
    topk_ids = indexer(h_normed, h_fp8, h_scale, q_lora, ...)  # dispatched last → phys 4

cur.wait_stream(alt_stream)
attn_out = phase3_post_wait(q_all, ..., topk_ids)
```

Why this works: `q_lora` is actually `q_lat` (phase2_current returns the pre-gemm tensor,
not `q`), so the indexer's first kernel (`wq_b`) only needs phase1's output. Placing
`wait_stream` before `phase2_current` lets alt start its indexer chain the instant phase1
completes, in parallel with cur's q_b_proj + gap-fill.

### Measured on MI355X (GLM-5-FP8 TP=8 M=1, median of 3 × 100-iter runs)

Results from `tools/glm5_proposalA_v3_test.py`:

| Variant | μs/layer | Δ vs base | notes |
|---|---|---|---|
| Baseline | 299.8 | — | 1-branch fork, no gap-fill |
| A_v2 (alt dispatched first) | 289.3 | −10.5 | layout wrong, parallelism OK |
| A_v3 (gap-fill first, wait_stream late) | 294.0 | −5.8 | clean layout, alt delayed |
| A_v3e (events) | 292.8 | −7.1 | same as A_v3 |
| **A_v4 (both rules applied)** | **280.9** | **−18.9 (−6.3%)** | **clean layout + alt on time** |

Sanity check via traces: alt-stream first-kernel start time — Baseline `t=50.7`, A_v3
`t=71.0` (late), A_v4 `t=50.0` (on time).

---

## Model: GLM-5-FP8 (GLM-4.7B MoE)

- Default path: `/data/huggingface/hub/zai-org/GLM-5-FP8`
- TP=8, FP8 quantization, FP8 KV cache
- NSA (Native Sparse Attention) for long-context
- MoE with shared experts (dual-stream overlap on ROCm)
- MTP/EAGLE speculative decoding (optional, `--mtp` flag)
- Tool call parser: `glm47`, reasoning parser: `glm45`

---

## Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `SAFETENSORS_FAST_GPU` | `1` | Fast GPU safetensors loading |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | `0` | Disable fused decode MLA on ROCm |
| `ROCM_QUICK_REDUCE_QUANTIZATION` | `INT4` | AllReduce quantization for ROCm |
| `SGLANG_TORCH_PROFILER_DIR` | `$LOG_DIR` | Where profiler traces are saved |
| `SGL_ENABLE_JIT_DEEPGEMM` | `1` | JIT DeepGEMM for NVIDIA GLM-5 |
