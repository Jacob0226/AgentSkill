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
| Reusable helper scripts | `~/SGLang-benchmarks/tools/` | **All Python / HTML tools and unit tests go here** (e.g. `glm5_decode_layer.py`, `glm5_proposalA_test_v2.py`, `analyze_trace_overlap.py`, `mi355x_decode_layer_timeline.html`). |
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

All scripts live in `~/SGLang-benchmarks/`. For detailed usage see [analysis-tools.md](analysis-tools.md).
For ROCm HIP stream/event/graph programming model, see [rocm-streams.md](rocm-streams.md).

| Script | Purpose |
|--------|---------|
| `analyze_trace.py` | **Main analysis**: 3-step trace analysis (kernel stats → layer structure → per-layer breakdown) |
| `compare_breakdown.py` | Compare two breakdown CSVs side-by-side (e.g. MI355X vs B200) |
| `parse_bench.py` | Parse benchmark logs → summary CSV (E2E, TTFT, ITL latencies) |
| `parse_perf_metrics_to_csv.py` | Parse `sglang.bench_serving` result blocks → CSV |
| `parse_torch_profiler.py` | Streaming parse of large `.trace.json.gz` → kernel CSV |
| `fetch_kernels_from_torch_profiler.py` | Fetch kernels by timestamp range from trace |

### Quick Analysis Workflow

```bash
# 1. Parse benchmark latencies
python3 ~/SGLang-benchmarks/parse_perf_metrics_to_csv.py --root-dir <results_dir>

# 2. Full trace analysis (graph-on + graph-off)
python3 ~/SGLang-benchmarks/analyze_trace.py \
    --graph-on <graph-on-trace.json.gz> \
    --graph-off <graph-off-trace.json.gz>

# 3. Compare MI355X vs B200
python3 ~/SGLang-benchmarks/compare_breakdown.py \
    --file-a MI355X/step3_layer_breakdown.xlsx \
    --file-b B200/step3_layer_breakdown.xlsx \
    --labels MI355X B200 --out comparison.xlsx
```

---

## HIP Graph Dispatch-Order Rule (dual-stream gotcha)

On ROCm / HIP Graph, the replay scheduler picks a physical stream for each branch of a
fork based on **dispatch order during capture**, not on which `torch.cuda.Stream` object
you passed. Rule:

> At each fork point, the branch whose first kernel is captured FIRST stays on the same
> physical stream as the predecessor (e.g. `phys 0`). The later-captured branch gets a
> new physical stream (`phys 4/5/8/…`).

Both `alt_stream.wait_stream(cur)` and explicit `record_event`/`wait_event` pairs behave
identically here — the sync primitive does not influence the choice.

**Symptom:** your `with torch.cuda.stream(alt):` block's kernels show up on `phys 0` in the
trace, and kernels you dispatched on `cur` land on `phys 4/5`. Looks like a remap but isn't
— it's the rule above.

**Fix:** dispatch the cur-side work *before* the `with torch.cuda.stream(alt):` block so cur's
chain is captured first.

```python
# BAD — alt captured first, indexer steals phys 0
alt_stream.wait_stream(cur)
with torch.cuda.stream(alt_stream):
    topk_ids = indexer(...)        # 1st dispatch → phys 0 (not what you want)
kv_normed = rms_norm(...)          # 2nd dispatch → phys 4
q_all = gap_fill(...)
cur.wait_stream(alt_stream)

# GOOD — cur captured first, stays on phys 0
alt_stream.wait_stream(cur)
kv_normed = rms_norm(...)          # 1st dispatch → phys 0 ✓
q_all = gap_fill(...)
with torch.cuda.stream(alt_stream):
    topk_ids = indexer(...)        # 2nd dispatch → phys 4 ✓
cur.wait_stream(alt_stream)
```

Confirmed on MI355X with `tools/glm5_proposalA_v3_test.py`. The "bad" layout can actually be
~5 μs *faster* (HIP packs more serial work on phys 0), but traces are unreadable. Prefer the
"good" layout for analysis; drop to "bad" only if you are chasing every last μs and know
what you are reading.

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
