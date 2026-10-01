# InferenceX AgentX (agentic benchmark) — what the numbers actually mean

AgentX replays recorded Claude Code sessions: multi-turn, 1M+ context, with
subagent fan-out. It is not a fixed-length sweep with a different ISL, and most
of the ways it bites you come from that difference.

Everything here was established by reading InferenceX source and verifying
against runs under
`~/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/.../bench-Agentic-TP4-f75c8k-d3600/`.

Harness: `~/SGLang-benchmarks/Agentic/ix_agentx_glm53flash_b200.sh` (driver) +
`recipe_glm53flash_fp4_b200_sglang_mtp.sh` (recipe). The split mirrors upstream:
the driver stands in for `benchmark-tmpl.yml`, the recipe for
`benchmarks/single_node/agentic/<model>.sh`.

---

## 1. "P90 Interactivity" is 1 / p90(TPOT), and the metric is misnamed twice

This is the x-axis of every InferenceX Pareto chart, so it is worth knowing
exactly what it is.

```
intvty.p90 = 1.0 / percentile(itls, 90)
```

Not `percentile(1/itl, 90)`. Inverting the p90 of a latency gives the **slow
tail** of the speed distribution — the rate 90% of requests beat, which in
speed terms is a p10. That is deliberate; the docstring says "Derive slow-tail
interactivity from the matching latency statistic." Taking `p90(1/ITL)` instead
would report the fastest 10% and flatter you by a wide margin.

Verified on real results — `1/itl.p90` reproduces `intvty.p90` exactly:

| conc | itl.p90 | 1 / itl.p90 | reported intvty.p90 |
|---|---|---|---|
| 1 | 4.160 ms | 240.4 | 240.3 |
| 8 | 5.310 ms | 188.3 | 188.2 |
| 16 | 8.300 ms | 120.5 | 120.4 |

### The naming is inverted relative to common usage

AIPerf's `inter_token_latency` is the **amortised** formula, which most people
call TPOT:

```
inter_token_latency = (request_latency - TTFT) / (output_sequence_length - 1)
```

The true per-token-gap measurement does exist in AIPerf, under a third name:

| Common name | Formula | AIPerf tag | InferenceX field |
|---|---|---|---|
| TPOT | `(E2EL − TTFT) / (OSL − 1)` | `inter_token_latency` | `itl` **and** `tpot` |
| ITL (real inter-token gaps) | `timestamps[i] − timestamps[i-1]` | `inter_chunk_latency` | unused |

AIPerf has no metric named TPOT. InferenceX computes `itl` and `tpot` from the
same list (`request_metrics.py:126-127`), so the two blocks in every result JSON
are byte-for-byte identical — confirmed on conc 1/8/16.

### Why the amortised form is the right choice here anyway

With MTP on, tokens arrive in bursts of 3-6 per forward pass. Real
`inter_chunk_latency` would be bimodal — near-zero inside a burst, large between
bursts — and its p90 would measure burstiness rather than perceived speed. The
amortised figure smooths that into "tokens per second this user sees", which is
what an interactivity axis should show. So when someone asks why not use real
ITL: for speculative decoding, amortised is correct.

### y-axis, for completeness

`per_gpu.total_tput_tps = (total_input_tokens + total_output_tokens) / duration / num_gpus`.
Input counts too. `duration` is derived from the records (`max(end) − min(start)`),
not the configured window.

---

## 2. A 20-minute window is fine for latency and wrong for throughput

A 3600s point costs ~70 minutes wall clock. Tempting to shorten it. Measured
both at conc 8, same config, only the window differs:

| metric | 1200s | 3600s | error | usable at 20 min? |
|---|---|---|---|---|
| P90 interactivity | 189.3 | 188.2 | **+0.6%** | yes |
| ITL p90 | 5.28 ms | 5.31 ms | **−0.6%** | yes |
| ITL p50 | 3.31 ms | 2.98 ms | +11.1% | marginal |
| per-GPU tok/s | 12,653 | 14,801 | **−14.5%** | no |
| TTFT p90 | 2.00 s | 3.11 s | **−35.7%** | no |
| completed requests | 582 | 1371 | | |

The throughput gap is warm-up, not noise: prefix cache hit rate climbs from
94.6% to 97.6% over the first ten minutes. TTFT is worse still and biased
*optimistic*, because the heaviest long prefills have not landed yet.

So: iterate at 1200s but read only ITL and interactivity; quote throughput and
TTFT only from 3600s. Comparing a 20-minute number against a 60-minute number is
always wrong, even for the same config.

`Agentic/window_sensitivity.py` answers "what would N seconds have reported" for
any finished run with no GPU time — it truncates the records and re-runs
InferenceX's own `build_result`. Its one bias: truncation drops requests still
in flight at the cutoff, so it slightly understates the real gap (it predicted
−10.2% where the real run showed −14.5%).

---

## 3. Watch for OOM *crashes*, not OOM *warnings*

Two different things in `server.log`, and conflating them wastes a day:

```
CUDACachingAllocator.cpp:3933  memory allocation failed with OOM ...   ← warning
Scheduler hit an exception ... CUDA out of memory                      ← failure
```

The warning means the caching allocator's first `cudaMalloc` failed, so it
released cached blocks and retried — and succeeded. The retry costs a
driver round trip, nothing else. A healthy GLM-5.3-Flash curve logs warnings at
every concurrency:

| conc | OOM warnings | scheduler crashes | verdict |
|---|---|---|---|
| 1 / 4 / 8 / 12 / 16 | 4 / 8 / 12 / 24 / 28 | 0 / 0 / 0 / 0 / 0 | all usable |

Chasing warnings to zero is not a goal; on a DSA model it is not even possible,
because the indexer sizes a fresh buffer per prefill batch.

`Agentic/check_health.py` prints crashes, `submission_valid` and error rate per
point. Run it before quoting any curve — the performance table will happily
report a good number for a point whose scheduler died and restarted.

---

## 4. The DSA indexer OOM, and which knob to turn

On GLM-5.3-Flash / B200 TP4, conc 12 and 16 died in warmup until the memory
split changed. The allocation is in the DSA indexer:

```
dsa_indexer_kpool.py:925  logits = deep_gemm.fp8_mqa_logits(...)
```

Its size has a clean closed form — the kpool compression (÷4) and the fp32
element size (×4) cancel:

```
logits bytes ≈ chunked_prefill_size × (sum of full context lengths in the prefill batch)
```

Verified twice: conc 12 at chunk 16384 → 16384 × 1.175M = **17.93 GiB** (OOM
against 25.6 GB headroom); conc 16 at chunk 8192 → 8192 × 1.47M = **11.23 GiB**
(fits in 43.3 GB). Note conc 16's context sum was *larger*; halving the chunk is
what saved it.

**Turn `--chunked-prefill-size` first.** It is linear in the buffer and touches
no pool. `--mem-fraction-static` is a blunt instrument that shrinks the KV pool
*and* the KDA state pool together — on this model going 0.85 → 0.75 cost 110
mamba slots (628 → 518), and the mamba pool is the one that is not obviously
over-provisioned.

Also: the `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION` knob and the chunked-logits
loop live in `dsa_indexer.py`, the **non**-kpool path. A model with
`index_kpool` in its config (GLM-5.3-Flash has `index_kpool: 4`) routes to
`dsa_indexer_kpool.py`, which allocates the whole matrix in one go and ignores
that env var entirely.

Settling this cost ~1% throughput and ~2% interactivity at conc 1/4/8, in
exchange for conc 12/16 existing at all.

---

## 5. Hybrid linear-attention models need the mamba radix path

GLM-5.3-Flash is 45 layers: 11 DSA/MLA full-attention, 34 KDA linear attention
carrying recurrent state. Cross-turn prefix reuse — the entire point of an
agentic scenario — therefore needs the hybrid-mamba radix cache.

Pin it rather than leaving `auto`:

```
--mamba-radix-cache-strategy extra_buffer
```

`auto` resolves to `extra_buffer` only when the arch is in
`_MAMBA_EXTRA_BUFFER_ARCHS` **and** `linear_attn_backend == "triton"`. Otherwise
it falls back to `no_buffer`, which requires `page_size=1` — but DSA forces
page_size 64, so the fallback does not degrade gracefully, it fails the launch.
Pinning turns a silent config dependency into a loud one.

And obviously: **do not carry `--disable-radix-cache` over from a fixed-length
recipe.** Measured prefix hit rate with it on is 93-98%.

---

## 6. KV-light models do not need HiCache

Count the layers that hold per-token KV before assuming the GLM-5.2 playbook
applies. GLM-5.3-Flash has 11 of 45, at ~7.2 KB/token, so TP4 fits 10.75M tokens
in HBM at `--mem-fraction-static 0.85`.

| conc | peak kv_usage | unique tokens over the whole run | ≈ capacity | device pool |
|---|---|---|---|---|
| 1 | 5% | 1.91M | 14 GB | 77 GB |
| 4 | 7% | 3.20M | 23 GB | 77 GB |
| 8 | 9% | 5.91M | 43 GB | 77 GB |

`unique_in_srv` is every distinct KV token the server saw — the working set with
no eviction at all. At conc 8 a full hour fits in HBM with room to spare, so a
host tier has nothing to do. By contrast GLM-5.2's upstream recipe needs 169
GB/rank at conc 12/16 because its working set genuinely outgrows HBM and the
radix hit rate collapses below 0.1.

Note the upstream GLM-5.2 B200 recipe enables HiCache at **every** concurrency,
not just the high ones — at conc 1/4/8 it passes `--hicache-ratio 0.75` instead
of an absolute size. Easy to misread as "off at low concurrency".

Cost of enabling it when unused: near-zero at runtime (write-back only fires on
eviction, and a pool at 5-9% never evicts), but real at startup — it pins
hundreds of GB of host memory.

### For cross-vendor comparison this is a confound, not a tuning choice

`agentic_dram_offload_gb()` scales the budget by the host's RAM, and the two
reference hosts differ:

| runner | host DRAM | TP4 budget | TP8 budget |
|---|---|---|---|
| `cluster:b200-nscale` | 2,063,920 MiB | 865 GB | 1731 GB |
| `cluster:mi355x-amds` | 3,095,781 MiB | 1199 GB | 2399 GB |

MI355X gets 39% more host DRAM under the same rule, *and* has 288 GB HBM against
B200's 178 GiB — so it needs less offload while being granted more. Running both
with HiCache off removes the variable and compares the chips. Running both with
it on compares the nodes. Pick one deliberately and say which.

---

## 7. Model prefix selects the corpus — keep the dot

`MODEL_PREFIX` is not cosmetic. `resolve_trace_source` globs it:

```bash
case "${MODEL_PREFIX:-}" in
    dsv4*|glm5.2*|glm5.3*|minimaxm3*|kimik3*)  # unfiltered 1M-context corpus
    *)                                          # 256k truncated corpus
```

`glm5.3flash` matches `glm5.3*` and gets the 1M corpus. **`glm53flash` does
not** — it silently falls through to the 256k variant, which changes what the
benchmark measures with no error and no log line saying so.

Flash variants also get their own prefix upstream (DeepSeek-V4-Pro is `dsv4`,
DeepSeek-V4.1-Flash is `dsv41flash`), so GLM-5.3-Flash is `glm5.3flash`, not
`glm5.3`. Using the parent's prefix also points `golden_length()` at the parent's
acceptance curve, which is exactly the silent mis-attribution worth avoiding.

`infmax_model_prefix` has no derived consumers in the agentic result path, so
fixing it on already-recorded points is a metadata backfill, not a re-run
(`Agentic/backfill_model_prefix.py`).

---

## 8. Acceptance length must be pinned — unless no curve exists

InferenceX requires a spec-decode AgentX point to pin acceptance to the
committed golden AL via `SGLANG_SIMULATE_ACC_LEN` / `_METHOD` /
`_TOKEN_MODE`, so throughput is not a function of how lucky the drafter got on
this corpus.

Check what is actually committed before assuming you can comply.
`golden_al_distribution/glm5.3_mtp.yaml` carries one cell — `glm-5.3-fp8` at
K=3 — and the file's own header marks it PROVISIONAL, copied from GLM-5.2. There
is no measured curve for GLM-5.3-**Flash** at any draft depth.

So those runs use real verification and are tuning numbers, not submittable
ones. Say so in the result rather than borrowing the parent model's curve.

---

## 9. Driver details worth keeping

- **Source `benchmarks/runtime_settings.sh`** rather than restating the
  `AIPERF_*` defaults. `build_replay_cmd` gates ~12 of them with
  `check_env_vars`, which aborts on unset rather than falling back. One
  exception: `AIPERF_EXPERIMENTAL_FAST` comes from the workflow, not that file,
  so export it as `0` explicitly.
- **`KV_OFFLOAD_BACKEND` must be empty when `KV_OFFLOADING=none`** —
  `process_agentic_result` rejects a result naming a backend it did not use.
- **Skip points whose result JSON exists.** Makes an interrupted sweep
  resumable, which is what lets a TP8 point be interleaved into a TP4 sweep at a
  point boundary instead of orphaning a 3600s window.
- **`--max-running-requests 2 × CONC`.** AgentX concurrency counts session
  trees, not requests; a trajectory fans out to subagents and clipping at CONC
  queues them artificially.
- **The cu13 SGLang image has `lsof` but not `fuser`** (ROCm images have
  `fuser`). Port cleanup has to branch on platform.
- **Grep the memory-pool lines out of `server.log` into the sweep log.**
  `max_total_num_tokens` and `max_mamba_cache_size` decide whether the hybrid
  split is sane and appear in **no** result JSON.
- **Set `MAX_MODEL_LEN` together with `--context-length`.** Capping the server
  without capping the client turns over-length traces into 4xx that still
  occupy a lane.
- **`AIPERF_RUNTIME_DIR` and `HF_HUB_CACHE` on node-local NVMe** (`/raid`), not
  the NFS home.

---

## 10. GLM-5.3-Flash B200 TP4 reference curve

Config: `--mem-fraction 0.75 --chunked-prefill 8192`, MTP steps 5 / 6 draft
tokens, HiCache off, real acceptance, 3600s window, SGLang v0.5.20-cu130.

| conc | ITL p50 | P90 intvty | TTFT p90 | per-GPU tok/s | Mtok/$ |
|---|---|---|---|---|---|
| 1 | 2.55 ms | 240.3 | 2.98 s | 5,638 | 11.73 |
| 4 | 2.87 ms | 210.3 | 3.13 s | 8,308 | 17.29 |
| 8 | 2.98 ms | 188.2 | 3.11 s | 14,801 | 30.80 |
| 12 | 3.24 ms | 162.5 | 3.04 s | 17,397 | 36.20 |
| 16 | 3.65 ms | 120.4 | 5.03 s | 26,977 | 56.14 |

Zero scheduler crashes, 3.4-4.4% error rate throughout. Two reads: the curve has
**not** saturated at conc 16 (12 → 16 adds 55% throughput), and TTFT is flat near
3 s until conc 16, where it jumps to 5 s — the first sign that chunked-prefill
8192 is becoming the constraint.

Per-chip cost basis in `ix_agentx_summarize.py`: B200 $1.73/chip/hr, MI355X
$1.50, B300 $2.26.

---

## Open questions on this model

- `--mamba-ratio`: prefix hit rate is 93.8% at conc 16 against 97% at low
  concurrency. Could be KDA slot starvation (518 slots at f0.75) or just
  concurrency; unresolved. Raising the ratio shifts memory from the
  over-provisioned KV pool back to mamba.
- conc 24/32 — the curve is still climbing at 16.
- TP8 at conc 1 bought 5% interactivity (240 → 262) for 2x the chips. At low
  concurrency this workload is bound by the trace's own idle gaps, not compute,
  so extra chips have nowhere to go. Whether that holds at high concurrency is
  untested.
