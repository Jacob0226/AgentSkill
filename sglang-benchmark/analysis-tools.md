# Analysis Tools Reference

Trace tooling lives in `~/SGLang-benchmarks/trace_analysis/`; run it from the
`~/SGLang-benchmarks/` root. That folder's own README documents the step3 workbook
schema and the two comparison flows — read it before changing an analyzer.

```
trace_analysis/analyze_sglang_trace.py      the core; only this and the next read raw traces
trace_analysis/analyze_atom_trace.py        same step3 schema for an ATOM trace
trace_analysis/compare/      side_by_side.py    --align lcs  = two SGLang runs, row-aligned
                                                --align none = SGLang vs ATOM, call order
                             glm52_buckets.py   ~9 GLM-5.2 functional buckets (+ the rules)
                             glm52_full_layer_sidebyside.py
trace_analysis/diagnostics/  verify_step3_against_trace.py, window_kernel_diff.py,
                             comm_skew_split.py, trace_kernel_summary.py,
                             extract_stream.py, analyze_trace_overlap.py
trace_analysis/regen_glm52_sidebyside.sh    driver for the whole SGLang-vs-ATOM set
```

Everything under `compare/` consumes the analyzers' workbooks, never a raw trace, so
there is one forward-segmentation implementation per stack and no comparison can
disagree with the analyzer about which forward it describes.

## analyze_sglang_trace.py (Main Analysis)

Three-step analysis combining the cuda-graph-**ON** trace (real timing) with the
cuda-graph-**OFF** trace (the only one carrying the `nn.Module` tree).

```bash
# Full analysis, restricted to ONE forward pass (always do this)
python3 trace_analysis/analyze_sglang_trace.py \
    --graph-on on.trace.json.gz --graph-off off.trace.json.gz \
    --forward-match "DECODE bs=64" --out DIR --tag _MI355X

# Step 1 only: kernel statistics from the graph-on trace
python3 trace_analysis/analyze_sglang_trace.py --graph-on on.trace.json.gz

# Step 2 only: layer structure from the graph-off trace
python3 trace_analysis/analyze_sglang_trace.py --graph-off off.trace.json.gz

# Export step 1 to CSV
python3 trace_analysis/analyze_sglang_trace.py --graph-on on.trace.json.gz --csv kernels.csv
```

**Steps**:
1. **Step 1** (graph-on): per kernel NAME — count, Σ, avg, % of the forward
2. **Step 2** (graph-off): layer structure — module classes, sub-modules, call order
3. **Step 3** (combined): one row per **call site**, aggregated over all layers

Without `--forward-match` every kernel is averaged over the whole trace, mixing
different batch/token sizes — that once turned an 861 µs all-reduce into 713 µs.

Key step3 columns: `LayerType` (which layer types reach the call site; the
**LayerTypes** sheet decodes the tags), `LaunchCount` (launches of that call site in
the forward — one per layer on the SGLang side), and `KernelCount_fwd` /
`KernelSum_ms_fwd` (the kernel NAME's totals in the forward, repeated on every row
sharing the name — never sum them). The tail of the run prints a coverage report
comparing the two.

---

## compare/side_by_side.py

`--align lcs`: compare two step3 workbooks row by row — two GPUs (MI355X vs B200) or
the same GPU before and after a change. LCS-matches (ParentModule, LeafModule) pairs,
so both sides must use the same module names.

```bash
python3 trace_analysis/compare/side_by_side.py --align lcs \
    --src MI355X MI355X/step3_layer_breakdown.xlsx \
    --src B200   B200/step3_layer_breakdown.xlsx \
    --out comparison.xlsx
```

`--align none` (default): keep each source in its own call order, any number of
sources. This is the SGLang-vs-ATOM mode — the module names do not correspond there,
so LCS would invent pairings. Every row still carries its caller and call site, and a
bucket summary is appended (`--summary-csv` embeds `glm52_buckets.py`'s CSV instead of
recomputing).

## compare/glm52_buckets.py

Sorts both sides' kernels into ~9 GLM-5.2 functional buckets (sparse-MLA attn, DSA
indexer+topk, dense GEMM, MoE up/gate, MoE down, MoE routing, all-reduce,
rmsnorm/quant, rope/kv-cache) and prints/writes the per-bucket Σ and ratio.

```bash
python3 trace_analysis/compare/glm52_buckets.py --phase prefill --out cmp.csv \
    --src SGLANG step3_layer_breakdown_SGLANG.xlsx \
    --src ATOM   step3_layer_breakdown_ATOM.xlsx
```

Scope follows the workbook kind and is printed: step3 = decoder layers only, step1 =
whole forward including the lm_head / sampling tail. Mixing kinds across sides is
refused. It is also the single source of the bucket regexes (`side_by_side.py` imports
them), so a workbook's summary and the CSV can never disagree.

---

## diagnostics/

```bash
# Does a step3 workbook actually match the trace? (run after touching an analyzer)
python3 trace_analysis/diagnostics/verify_step3_against_trace.py \
    --step3 step3_layer_breakdown_SGLANG.xlsx --trace on.trace.json.gz \
    --forward-match "bs=3"

# Which kernels differ between two windows (two ranks, or two stacks)?
python3 trace_analysis/diagnostics/window_kernel_diff.py --stats --show-args ...

# Is a slow all-reduce real comm or waiting for the slowest rank? (needs all TP ranks)
python3 trace_analysis/diagnostics/comm_skew_split.py ...

# Quick per-kernel totals of one trace, no structure needed
python3 trace_analysis/diagnostics/trace_kernel_summary.py trace.json.gz

# Per-stream inspection / stream concurrency and bubble time
python3 trace_analysis/diagnostics/extract_stream.py trace.json.gz --list-streams
python3 trace_analysis/diagnostics/analyze_trace_overlap.py dual.trace.json.gz single.trace.json.gz

# Attribute a per-forward regression to individual kernels. Normalizes traces that
# captured a different number of forwards (profile-by-stage often gives 1 vs 2), and
# surfaces launch-COUNT changes, which is how a lost kernel fusion shows up.
python3 trace_analysis/diagnostics/per_forward_kernel_delta.py \
    --a baseline-DECODE.trace.json.gz --b variant-DECODE.trace.json.gz --labels base dual

# Split GLM-5.2 decode all-reduce calls by layer type: tells a per-fork cost apart
# from a cost that is global to multi-stream graph capture.
python3 trace_analysis/diagnostics/ar_by_layer_type.py \
    --label baseline base-DECODE.trace.json.gz --label dual dual-DECODE.trace.json.gz
```

### Reading overlap numbers: two traps

1. **`Σ kernel dur / union busy` overstates CUDA overlap ~2x.** CUPTI reports ~1.3 µs
   of *same-stream* overlap at nearly every kernel boundary, which is physically
   impossible. On a B200 GLM-5.2 decode trace, 6.06 ms of an apparent 8.48 ms
   "saving" was same-stream skew; true cross-stream overlap was 3.84 ms. roctracer
   shows 0.000 ms of this, so MI355X looks "cleaner" purely because of
   timestamping. Always recompute overlap using cross-stream pairs only before
   comparing vendors.
2. **Stream count is not concurrency, but do not conclude that from aggregates.**
   A CUDA-graph decode trace can show 46 stream IDs where most are graph-internal
   node partitioning that runs back to back (B200 GLM-5.2: 19 such streams, busy
   19.8 ms, only 10.9% overlapping the main stream). But in the same trace 21
   streams *are* functionally specialized, one per full-indexer layer carrying the
   DSA indexer chain. Grouping kernel names over the whole trace makes them look
   identical to the rest; you only see it by grouping per stream **and** windowing
   to one forward. Look for a marker kernel (`paged_mqa_logits`, a collective) and
   ask which streams carry it.

---

## parse_perf_metrics_to_csv.py (repo root)

Parse `sglang.bench_serving` result blocks from log files into CSV.

```bash
python3 parse_perf_metrics_to_csv.py --root-dir <results_dir>
```

Extracts from log filenames: `bench_in{input}_out{output}_conc{concurrency}.log`
Parsed fields: E2E latency, TTFT, ITL, throughput, etc.
