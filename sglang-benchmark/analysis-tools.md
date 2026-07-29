# Analysis Tools Reference

Trace tooling lives in `~/SGLang-benchmarks/trace_analysis/`; run it from the
`~/SGLang-benchmarks/` root. That folder's own README documents the step3 workbook
schema and the two comparison flows — read it before changing an analyzer.

```
trace_analysis/sglang/       analyze_trace.py                 (the core)
trace_analysis/atom/         analyze_atom_trace.py
trace_analysis/compare/      compare_step3.py         (two SGLang runs)
                             sglang_vs_atom_glm52.py  (SGLang vs ATOM, buckets)
                             callorder_sidebyside.py  (SGLang vs ATOM, call order)
                             glm52_full_layer_sidebyside.py
trace_analysis/diagnostics/  verify_step3_against_trace.py, window_kernel_diff.py,
                             comm_skew_split.py, trace_kernel_summary.py,
                             extract_stream.py, analyze_trace_overlap.py
trace_analysis/regen_glm52_sidebyside.sh    driver for the whole SGLang-vs-ATOM set
```

## sglang/analyze_trace.py (Main Analysis)

Three-step analysis combining the cuda-graph-**ON** trace (real timing) with the
cuda-graph-**OFF** trace (the only one carrying the `nn.Module` tree).

```bash
# Full analysis, restricted to ONE forward pass (always do this)
python3 trace_analysis/sglang/analyze_trace.py \
    --graph-on on.trace.json.gz --graph-off off.trace.json.gz \
    --forward-match "DECODE bs=64" --out DIR --tag _MI355X

# Step 1 only: kernel statistics from the graph-on trace
python3 trace_analysis/sglang/analyze_trace.py --graph-on on.trace.json.gz

# Step 2 only: layer structure from the graph-off trace
python3 trace_analysis/sglang/analyze_trace.py --graph-off off.trace.json.gz

# Export step 1 to CSV
python3 trace_analysis/sglang/analyze_trace.py --graph-on on.trace.json.gz --csv kernels.csv
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

## compare/compare_step3.py

Compare any two step3 workbooks side by side — two GPUs (MI355X vs B200) or the same
GPU before and after a change.

```bash
python3 trace_analysis/compare/compare_step3.py \
    --file-a MI355X/step3_layer_breakdown.xlsx \
    --file-b B200/step3_layer_breakdown.xlsx \
    --labels MI355X B200 \
    --out comparison.xlsx
```

Uses LCS alignment to match (ParentModule, LeafModule) pairs, so both sides must use
the same module names. For SGLang vs ATOM they do not — use `sglang_vs_atom_glm52.py`
(functional buckets) plus `callorder_sidebyside.py` (call order) instead.

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
```

---

## parse_perf_metrics_to_csv.py (repo root)

Parse `sglang.bench_serving` result blocks from log files into CSV.

```bash
python3 parse_perf_metrics_to_csv.py --root-dir <results_dir>
```

Extracts from log filenames: `bench_in{input}_out{output}_conc{concurrency}.log`
Parsed fields: E2E latency, TTFT, ITL, throughput, etc.
