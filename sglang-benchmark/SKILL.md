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
| Reusable helper scripts | `~/SGLang-benchmarks/tools/` | **Micro-benchmarks, stream/graph tests, env setup go here** (e.g. `kill_sglang.sh`, `glm5_decode_layer.py`, `glm5_proposalA_v3_test.py`, `glm5_proposalA_overlap_analysis.py`, `mi355x_decode_layer_timeline.html`). |
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

## Killing a stuck SGLang server (do NOT improvise this)

**Always use `~/SGLang-benchmarks/tools/kill_sglang.sh <container> [port]`.** Every
ad-hoc `pkill` variant has a failure mode that leaves the box unusable and makes
the *next* run die at startup with `localhost:8234 is already in use`.

```bash
~/SGLang-benchmarks/tools/kill_sglang.sh jacchang_GLM5_0728_
```

Five independent traps, all measured on a live run (8 processes total):

| Command | Processes it kills | What it misses |
|---|---|---|
| `pkill -f sglang.launch_server` | **1 of 8** | the 5 renamed workers + GLM.sh |
| `pkill -9 python3` | 2 (launcher, resource_tracker) | `sglang::*` workers, GLM.sh |
| `pkill -9 sglang` | 5 (`sglang::*` workers) | launcher, GLM.sh |
| `pkill -9 -f "GLM[.]sh"` | 2 | the engine |

1. **`pkill -f <pat>` kills its own shell.** `docker exec ... bash -c 'pkill -f
   sglang.launch_server'` puts the pattern into the shell's own
   `/proc/<pid>/cmdline`, so pkill matches itself and the exec dies (exit 143/137)
   before the rest of the cleanup runs. Verified directly: a `bash -c` running
   `pkill -f "sleep 300"` was SIGTERMed mid-script.
   *Fix:* match on **comm** (`pkill -9 python3`, no `-f`) — the shell's comm is
   `bash`, so it can never self-match. Where `-f` is unavoidable (GLM.sh), break
   the literal with a bracket: `pkill -9 -f "GLM[.]sh"`.

2. **SGLang renames its workers, so `-f sglang.launch_server` misses them.**
   Only the launcher has `sglang.launch_server` in its args. setproctitle renames
   the children to `sglang::scheduler_TP0`, `sglang::tokenizer_worker`,
   `sglang::detokenizer`, which comm truncates to `sglang::schedul` etc. Matching
   `python3` and `sglang` are **both** required — neither covers the other
   (launcher comm is `python3`; worker comms contain no "python").

3. **The port is held by a tokenizer_worker, not the launcher.** `pgrep -f
   launch_server` returning nothing while `ss -ltn` still shows `:8234` is the
   normal symptom. Always verify by polling the **port**, never the process list.

4. **Cancelling `docker exec` does not kill anything inside the container.** The
   in-container tree is reparented and keeps running — observed as an orphan
   `GLM.sh` (ppid 0) still alive 863 s after its exec was cancelled. Worse, GLM.sh
   has a config loop and a server-restart retry, so if you kill the engine while
   GLM.sh survives it launches a **new** server within seconds and you play
   whack-a-mole. **Kill GLM.sh first, engine second.**

5. **Never pass the cleanup script as an argument to `bash -c`.** `docker exec c
   bash -c "$script"` puts the whole script text — *including its comments* — into
   the shell's cmdline, so a `pkill -9 -f "GLM[.]sh"` line self-matches on the word
   `GLM.sh` sitting in a nearby comment. This killed the first version of
   `kill_sglang.sh` at exit 137. Feed the script over **stdin** instead
   (`... | docker exec -i c bash -s -- "$port"`); then the cmdline is only
   `bash -s -- 8234` and there is nothing to match.

If you must do it by hand, this is the whole recipe (note the quoting — typing it
inline is exactly how trap 5 bites):

```bash
docker exec <container> bash -s <<'EOS'
pkill -9 -f "GLM[.]sh"; pkill -9 -f "bench[_]serving"   # orchestrator first
pkill -9 python3; pkill -9 sglang                       # then engine, by comm
sleep 8; ss -ltn | grep :8234 || echo "port free"       # verify the PORT
EOS
```

Caveat: the benchmark containers run with host networking, so the port check is
host-wide. If `kill_sglang.sh` reports the port still held after cleaning the
container you named, a *different* container is holding it — check `docker ps`.

---

## Before reporting any kernel speedup

Read [microbench-pitfalls.md](microbench-pitfalls.md) **before** writing a
micro-benchmark, not after. The short version:

- MI355X has a **256 MB Infinity Cache**. Any sweep whose working set fits in it
  measures cache bandwidth, not the kernel. Rotate ≥ 6 input buffers (> 512 MB
  total) or the ranking will be wrong — this has already produced one bogus
  "1.3-1.8x faster tile" that was worth 0% in the real trace.
- `torch.randn` inputs hide quantization accuracy loss. Any quant change needs
  GSM8K + output throughput, never a kernel timing alone.
- Benchmark at the M the scheduler actually emits (read `tok=` off the
  `prefill[...]` trace annotation), not a round number.

Related: [triton-specialization-cliffs.md](triton-specialization-cliffs.md) for
why that last point matters so much. Triton `@triton.heuristics` flags are
`constexpr`, so each combination is a **separate compiled binary** — on the MLA
absorb bmm, 16 tokens of misalignment flips `EVEN_MN` and costs 2x (3819 vs 1562
instructions around identical MFMA work). Includes how to diff the ISA of two
specializations, and why the sibling `EVEN_K` flag is *not* worth chasing.

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
- `--disable-shared-experts-fusion` (only for the MoE half of dual-stream; it
  costs 0.76 ms TPOT at conc64, where the MXFP4 MoE kernels already saturate)
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

# Dual stream on ROCm. --dual-stream-rocm = alt_stream + unfused shared experts;
# --dual-stream-attn-only = alt_stream only (isolates the overlap from the MoE half).
# Measured a loss on GLM-5.2 i8k/conc64: +6.3% TPOT as shipped, +3.6% after two
# fixes -- see tools/dual_stream_conc64_design.md before spending time here.
./GLM.sh --dual-stream-attn-only

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
| `trace_analysis/analyze_sglang_trace.py` | **Main analysis**: 3-step trace analysis (kernel stats → layer structure → per-call-site breakdown). Always pass `--forward-match`. |
| `trace_analysis/compare/side_by_side.py` | Put step3 workbooks side by side: `--align lcs` row-aligns two SGLang runs (MI355X vs B200), `--align none` keeps each side's own call order (SGLang vs ATOM) |
| `trace_analysis/compare/glm52_buckets.py` | Both sides' kernels in ~9 GLM-5.2 functional buckets, from the same step3 workbooks |
| `trace_analysis/analyze_atom_trace.py` | Same step3 schema for an ATOM trace (SGLang vs ATOM flow) |
| `trace_analysis/diagnostics/` | `verify_step3_against_trace.py`, `window_kernel_diff.py`, `comm_skew_split.py`, `trace_kernel_summary.py`, `extract_stream.py`, `analyze_trace_overlap.py`, `per_forward_kernel_delta.py` (attribute a per-forward regression to individual kernels; normalizes traces that captured different forward counts), `ar_by_layer_type.py` (split decode all-reduce calls by GLM-5.2 layer type — tells a per-fork cost apart from a global one) |
| `parse_perf_metrics_to_csv.py` | Parse `sglang.bench_serving` result blocks → CSV |

### Quick Analysis Workflow

```bash
# 1. Parse benchmark latencies
python3 ~/SGLang-benchmarks/parse_perf_metrics_to_csv.py --root-dir <results_dir>

# 2. Full trace analysis (graph-on + graph-off)
python3 ~/SGLang-benchmarks/trace_analysis/analyze_sglang_trace.py \
    --graph-on <graph-on-trace.json.gz> \
    --graph-off <graph-off-trace.json.gz> \
    --forward-match "DECODE bs=64"

# 3. Compare MI355X vs B200
python3 ~/SGLang-benchmarks/trace_analysis/compare/side_by_side.py --align lcs \
    --src MI355X MI355X/step3_layer_breakdown.xlsx \
    --src B200   B200/step3_layer_breakdown.xlsx \
    --out comparison.xlsx
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

### Rule 3 — never split a pair that a fused kernel already handles

Before overlapping two ops on separate streams, check whether aiter already fuses
them. `forward_mla.py` split the q-norm and k-norm across streams whenever
`alt_stream` existed, which on ROCm/gfx95 threw away `fused_qk_rmsnorm_bf16` /
`fused_rms_mxfp4_quant` (one launch, and the FP8/MXFP4 quant folded in). Cost:
**+901 μs per decode forward** — 31% of the entire dual-stream regression — to
overlap two 4 μs kernels. Anything written for CUDA's unfused path is suspect here.

Symptom in trace: a fused kernel disappears from the dual-stream trace and is
replaced by two unfused ones plus a pile of `elementwise_kernel_manual_unroll`
launches. `per_forward_kernel_delta.py` surfaces this immediately as a launch-count
change, not just a duration change.

### The first fork pays a flat toll (MI355X, ROCm 7.2)

Measured by fork-count sweep, GLM-5.2 TP4 i8k/conc64, median TPOT:

| forks | 0 | 1 | 4 | 8 | 21 |
|---|---|---|---|---|---|
| TPOT | 48.04 | 49.78 | 49.87 | 49.95 | 49.91 |

Enabling multi-stream is free: with `SGLANG_ROCM_USE_MULTI_STREAM=1` and
`alt_stream` allocated but nothing forked, TPOT matches baseline (48.04 vs
48.10-48.24 over three runs). The **first** cross-stream dependency in the captured
graph costs **1.65 ms**; the next twenty cost 0.13 ms. After that toll every kernel
in the graph replays slower, which is why all-reduce slows +1.99 μs/call and MoE
mfma 1.8-3.1% on the 57 layers that never fork, and why gating forks 78 → 21
returned only 7%.

Practical consequences:

- **Do not budget per fork; budget one 1.65 ms toll.** If the total hideable work
  is under ~1.7 ms TPOT, no fork count wins and no stream layout helps.
- **Once you pay it, fork widely.** Marginal cost past the first fork is ~6 μs
  against ~20 μs hidden, so a narrow fork is the worst of both worlds.
- Sweep it yourself with `SGLANG_DS_FORK_MAX_LAYER_ID=<layer-id ceiling>` (in the
  6PR tree's `forward_mla.py`) plus
  `VARIANTS="baseline fork0 fork1 fork4 fork8 fork21" tools/dual_stream_conc64_ablation.sh`.

Details and re-test triggers: `SGLang-benchmarks/tools/dual_stream_conc64_design.md`.

### B200 forks the same indexer — check before assuming NVIDIA does it better

GLM-5.2 decode on B200 shows 46 stream IDs. Most are graph-internal partitioning
that runs back to back, but **21 are functionally specialized**: one per
full-indexer layer, each carrying only the DSA indexer chain (`paged_mqa_logits`,
`_act_quant_kernel`, the hadamard `triton_poi_fused_mul`s, q/k proj). Same fork as
ROCm, same `is_decode_or_idle()` gate, and prefill forks nothing on either
platform. B200 hides 0.32 ms/forward with it; MI355X hides 0.50 ms. Both are near
the structural ceiling (~0.42 ms), because the indexer's `topk` feeds sparse MLA so
only `q_b_proj` plus the pre-attention gemm/rope can overlap, ~20 μs per layer.

So "B200 is faster because it overlaps more streams" is wrong for this model — its
lead is raw kernel speed (111 ms/prefill pass, ~3.2 ms/decode pass) against ≤0.32 ms
of indexer overlap.

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
