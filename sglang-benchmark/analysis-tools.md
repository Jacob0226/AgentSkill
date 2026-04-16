# Analysis Tools Reference

All scripts are in `~/SGLang-benchmarks/`.

## analyze_trace.py (Main Analysis)

Three-step trace analysis combining cuda-graph-ON and OFF traces.

```bash
# Full analysis (both traces)
python3 analyze_trace.py --graph-on on.trace.json.gz --graph-off off.trace.json.gz

# Step 1 only: kernel statistics from graph-on trace
python3 analyze_trace.py --graph-on on.trace.json.gz

# Step 2 only: layer structure from graph-off trace
python3 analyze_trace.py --graph-off off.trace.json.gz

# Export step 1 to CSV
python3 analyze_trace.py --graph-on on.trace.json.gz --csv kernels.csv

# Add tag to output files
python3 analyze_trace.py --graph-on on.trace.json.gz --graph-off off.trace.json.gz --tag MI355X
```

**Steps**:
1. **Step 1** (graph-on): Parse kernel stats — count, total duration, avg, percentage
2. **Step 2** (graph-off): Parse layer structure — module types, sub-modules, execution order
3. **Step 3** (combined): Per-layer kernel breakdown with sub-module labels

**Output**: CSV/XLSX files with kernel breakdown per layer.

---

## compare_breakdown.py

Compare two `step3_layer_breakdown` files side by side (e.g. MI355X vs B200).

```bash
python3 compare_breakdown.py \
    --file-a MI355X/step3_layer_breakdown.xlsx \
    --file-b B200/step3_layer_breakdown.xlsx \
    --labels MI355X B200 \
    --out comparison.xlsx
```

Uses LCS alignment to match (ParentModule, LeafModule) pairs across platforms.

---

## parse_perf_metrics_to_csv.py

Parse `sglang.bench_serving` result blocks from log files into CSV.

```bash
python3 parse_perf_metrics_to_csv.py --root-dir <results_dir>
```

Extracts from log filenames: `bench_in{input}_out{output}_conc{concurrency}.log`
Parsed fields: E2E latency, TTFT, ITL, throughput, etc.

---

## parse_bench.py

Parse benchmark log directories into summary CSV with averaged latencies.

```bash
python3 parse_bench.py --root-dir ~/prof/0123_bench --output summary.csv
```

Expected directory structure: `i{N}-o{N}-n{N}-concurrency{N}/bench*.log`
Output columns: config, E2E, TTFT, ITL (median values averaged across runs).

---

## parse_torch_profiler.py

Streaming parse of large `.trace.json.gz` files using ijson. Extracts GPU kernel statistics.

```bash
python3 parse_torch_profiler.py \
    --file trace.json.gz \
    --out kernels.csv
```

Output columns: Name, TotalCalls, TotalDuration_us, Ave_us, Percentage.
Uses streaming (ijson) to handle multi-GB trace files without loading into memory.

---

## fetch_kernels_from_torch_profiler.py

Fetch GPU kernels with timestamps from a trace, optionally filtered by time range.

```bash
# All GPU kernels
python3 fetch_kernels_from_torch_profiler.py trace.json

# Filter by timestamp range
python3 fetch_kernels_from_torch_profiler.py trace.json \
    --start 6195887545100 --end 6195887600000

# Top 20 longest kernels
python3 fetch_kernels_from_torch_profiler.py trace.json --top 20 --sort dur

# Export to CSV
python3 fetch_kernels_from_torch_profiler.py trace.json --csv > kernels.csv
```

Options: `--start`, `--end`, `--sort {start,end,dur,name}`, `--top N`, `--csv`, `--list-all`.
