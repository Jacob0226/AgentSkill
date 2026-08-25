# Micro-benchmark pitfalls (read before trusting any kernel sweep)

Every entry here is a real wrong conclusion that survived a clean-looking
micro-benchmark and was only caught by an end-to-end run. Check these **before**
reporting a speedup, not after.

---

## 1. Working set < LLC → the number is fiction

**The rule: if your benchmark's total touched bytes fit in the last-level cache,
you are not measuring the kernel you think you are.**

| GPU | LLC | Safe working-set floor for a memory-bound sweep |
|---|---|---|
| MI355X (CDNA4) | 256 MB Infinity Cache (MALL) | **≥ 512 MB**, ideally ≥ 600 MB |
| MI300X (CDNA3) | 256 MB Infinity Cache | ≥ 512 MB |
| B200 (Blackwell) | 126 MB L2 | ≥ 256 MB |

MI355X's Infinity Cache is a *memory-side* cache (MALL = Memory Attached Last
Level): it sits in front of the memory controllers and caches everything headed
to HBM regardless of which XCD asked. A `for _ in range(100): kernel(A, B)` loop
leaves both operands resident in it, so the kernel runs at Infinity-Cache
bandwidth instead of HBM bandwidth. In production the same kernel is one of ~60
layers all evicting each other, and never sees that hit rate.

### What it cost us

A tile sweep for the MLA absorb bmm (`w_kc @ q_nope`, M=16368, ~100 MB of
activations) ranked `BLOCK_M=64 / BLOCK_N=256` as **1.3-1.8x faster** than
`BLOCK_M=128`, consistently, across many repeats. Shipped it; the real trace
moved 454.8 us → 444.3 us, i.e. nothing.

Reason: the two tiles trade A-matrix reads against B-matrix reads. With the
weights parked in Infinity Cache, B reads were nearly free, so the sweep
rewarded the config that minimised A reads. Re-running with **6 rotating input
buffers (600 MB total)** collapsed the advantage to 6-7%. The 1.3x never
existed.

### The fix: rotate buffers

```python
# Size the pool so N_BUF * bytes_per_buf > 2 * LLC.
N_BUF = 6
bufs = [torch.randn(M, K, device="cuda", dtype=torch.bfloat16) for _ in range(N_BUF)]

for i in range(warmup + iters):
    out = kernel(bufs[i % N_BUF], w)   # each call touches a different buffer
```

Sanity check before you start: `N_BUF * A_bytes + B_bytes` printed next to the
result. If it is under 512 MB on MI355X, the result is not reportable.

If the tensors are too small to build a 600 MB pool, flush the cache between
iterations instead (memset a 512 MB scratch buffer), and say in the write-up
which method you used — hot and cold numbers are both legitimate, they just
answer different questions. Decode-shaped kernels (M ≤ 64) genuinely *are*
cache-resident in production, so hot-cache is the right model there. Prefill
kernels (M in the thousands) are not.

---

## 2. `torch.randn` inputs hide quantization accuracy loss

Per-tensor FP8 on `o_proj` micro-benchmarked as a clean 2.6x GEMM win with
normal-random activations. End to end it cost **GSM8K 0.936 → 0.922** and
**output throughput 2049 → 1930 tok/s**.

Real activations have outlier tokens; a single per-tensor scale over
`[16368, 4096]` is set by the outlier and starves everything else. Gaussian
inputs have no outliers, so the micro-benchmark cannot see the failure mode.

**Rule: no quantization change ships on a kernel timing alone.** It needs GSM8K
(or equivalent) *and* an output-throughput number, because the extra quant
kernel also lands on the critical path.

---

## 3. A kernel-level win is not a system-level win

Two independent ways this bites:

- **Serialisation.** bf16 is one GEMM; FP8 is `quant → GEMM`, with a hard
  dependency and no overlap (SGLang on MI355X is single-stream here, and so is
  ATOM). A −35 ms GEMM with a +3 ms quant kernel did **not** produce −32 ms end
  to end; go find where the rest went before believing it.
- **Shape coverage.** The sweep picks one M. The real forward runs many M values
  across layers and across chunked-prefill steps, and untuned shapes can fall
  into a JIT path that is slower (or, with aiter flydsl, crashes).

Always close the loop with `trace_analysis/analyze_sglang_trace.py` on a real
trace and compare the *same* kernel's total ms, not the micro-benchmark's.

---

## 4. Alignment cliffs are real, but check which side you are on

`M % BLOCK_SIZE_M == 0` flips the aiter Triton `EVEN_MN` heuristic, which
selects a **different compiled binary** — 2.44x the instructions around
identical MFMA work. Measured on the absorb bmm: M=16384 → 155.8 us,
M=16368 → 311.2 us. A **2x cliff from 16 tokens.**

This one is genuine (it reproduces cold-cache and in the real trace), but it
means a sweep at a convenient round M reports numbers the production workload
will never see. SGLang's scheduler delivers 16289-16368, not 16384 — always
benchmark at the M the scheduler actually produces, which you can read off the
`prefill[... tok=N ...]` annotation in the trace.

The mechanism, the ISA evidence, and the two explanations that sound right but
are wrong ("the modulo breaks load coalescing", "`EVEN_K` must matter for the
same reason") are in
[triton-specialization-cliffs.md](triton-specialization-cliffs.md).

---

## 5. A sweep that times each candidate once is ranking noise

**The rule: if an autotuner measures every candidate once and keeps the argmin,
its winner is partly a lucky draw. Re-measure the top few, and check the winner
against the incumbent on the production entry point before believing the order.**

aiter's `gemm_moe_tune.py --mxfp4-flydsl` sweeps 174 (stage-1, stage-2) pairs per
shape. Each one gets a single `run_perftest(warmup=5, iters=101)`, then:

```python
us = self._run_candidate(row, candidate, args)
if best is None or us < float(best["us"]):
    best = candidate
```

No repeat, no outlier rejection, no re-check of the top-k. 101 iterations inside
one measurement does not help: it tightens that one sample, it does not tell you
whether the sample itself landed wrong.

### What it cost us

Three identical sweeps of GLM-5.2 MoE token=128, same box, same build. Watching
one candidate, `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt`, always timed second:

| sweep | that kernel | its rank | sibling `_atomic`, control |
|---|---|---|---|
| first | **336.87** | 168 / 174 | 220.71 |
| rerun 1 | 208.20 | 6 / 174 | 219.28 |
| rerun 2 | 207.83 | 7 / 174 | 216.27 |

One bad sample buried a good kernel at rank 168. It was not the machine, the
position in the sweep, or a cache effect — all of those would reproduce, and the
sibling timed immediately before it stayed flat across all three.

The subtler half is that **even the clean reruns rank it wrong.** The sweep put a
`moe2_layout` kernel first at 203.5 us with this one 6th at 207.8, a 2.2% gap;
measured through `fused_moe`, the "loser" wins 225.0 vs 232.8. At a 2-3%
separation the sweep has no resolving power, so its order is arbitrary.

Selection bias is measurable here. For a kernel the sweep did *not* pick, sweep
208 vs production 225 is 7.5% optimistic — that is just the two harnesses timing
different regions. For the kernel it *did* pick, 203.5 vs 232.8 is 12.6%. The
extra ~5 points is the argmin: take the minimum of 174 noisy draws and you
preferentially select whichever candidate got the fastest bad measurement.

This nearly shipped. The aiter PR it appeared in argued that a tuner had
overwritten a good config without measuring it — and then proposed shipping a
fresh tune, unchecked. A/B against the row being replaced caught it: the
"improved" pick was 11.0% slower at token=32.

### The fix

- Re-measure the top ~10 candidates, several times each, and rank on the median.
- Treat any gap under ~5% in a single-sample sweep as a tie, not an ordering.
- Never ship a sweep winner without an A/B against the incumbent config on the
  function production actually calls (for MoE that is `fused_moe`, not the
  tuner's internal timing path).

---

## Pre-flight checklist

Before reporting any micro-benchmark result:

- [ ] Working set > 2x LLC, or explicitly labelled as a hot-cache number
- [ ] M / shapes taken from a real trace annotation, not rounded
- [ ] Real activation distributions if quantization is involved
- [ ] Confirmed against the same kernel's total ms in an end-to-end trace
- [ ] Accuracy + output throughput checked if anything numeric changed
- [ ] Autotuner winners re-measured against the incumbent, not trusted on the
      sweep's own single-sample ranking
