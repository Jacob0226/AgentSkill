# Monokernel pitfalls

Each entry: symptom, cause, how to detect it, fix. All were hit while adding
MTP verify to the GLM-5.3-Flash monokernel on MI355X.

## Correctness

### Long-context memory fault (iscore, > 262,144 tokens)

- **Symptom:** GPU memory fault minutes into AgentX; short benchmarks pass.
- **Cause:** above 65,536 pools, the iscore tile loop runs more than 256 tiles. LLVM's structurizer turned a divergent conditional store inside the loop into an exit-and-reenter pattern, and the store's index register (`v18`) held a stale float for some lanes. The store went out of bounds.
- **Detect:** test at a representative long context. Agentic traces reach hundreds of thousands of tokens, so use something like 400K; `check_layer.py`'s default `MAX_LEN` of 8192 is far too short. Pass `--max-len` and `--lens`. Bisect the length: 262,147 passed, 262,148 faulted.
- **Fix:** make the store unconditional through a buffer resource whose `num_records` bounds the valid range. Lanes that shouldn't store get an offset past it, and the hardware drops the store (the same pattern as `deepseek_v4`'s `ffn.py`).
- **Don't:** send the masked lanes to one discard slot. See "Hotspot from a discard address".

### Bitwise check differs at 1e-7

- **Cause:** decode at width 2 and verify at width 8 tile the GEMMs differently.
- **Fix:** compare at the same width. At equal width, verify = L decode launches was bitwise.

### Graph capture fails in `_step_table`

- **Symptom:** capture requires that the graph's step was first used by an eager step.
- **Cause:** step tables are keyed by tensor pointers, and `repeat_interleave` / `clamp` produced new tensors each call.
- **Fix:** fixed per-width buffers (`verify_in`), filled with `copy_`.

### Intra-chain ordering (DSA verify)

Rows of one request in the same launch read keys and pools that earlier rows
in that chain write. Verify rows must wait on `kv_ready` of every earlier row
in the chain, not only their own. With MTP the k-pool tail ring has extra
slots, so the ring index is `pos % TAIL`, with `TAIL` derived from the tail
tensor, not hard-coded 4.

### Test artifacts that look like failures

- `check_keys` lengths divisible by 4 have no tail token to perturb; the "changes the row" check fails on an empty set. Use lengths like 131001.
- `check_layer --rows 6` fails an assert: 6 isn't a compiled width. Six verify rows run at width 8 with 2 pad rows (`--rows 8 --verify 6`).
- New optional (`None`) fields in a state dataclass break harness code that clones every field (`astuple(...)`). Clone only the fields the launch uses.

## Performance

### Hotspot from a discard address

- **Symptom:** after the fault fix, AgentX conc 1 was 39% worse than baseline; 8K/1K was still faster.
- **Cause:** 48 of 64 lanes on every CTA stored, at device scope, to the same discard pair. At 393k tokens the DSA layer took 1960 µs instead of 290 µs.
- **Detect:** microbenchmark across the context range before and after any fix (`time_layers.py --kind dsa --pos <p>`), and an in-kernel stage timeline.

### In-kernel stage timeline

`common/ops.py` has `clock()` (100 MHz realtime counter). Add a scratch region
sized stages × CTAs, have thread 0 of each CTA store `clock()` after each
stage, read it back from a small script built on `check_layer`'s helpers, and
compare per-stage median and max. Finer marks inside a stage (after a poll,
after a barrier) separate waiting from work.

### Interpreting AgentX

- "P90 interactivity" on the InferenceX board is 1 / ITL p90 (`ix_agentx_summarize.py`, `latency.intvty.p90`), not `e2e_norm_intvty`.
- Rows per step = running requests × tokens per request (MTP5 = 6). In the conc 4 baseline, 84.5% of decode batches had 1 request, 12.4% had 2. Read `#running-req` in the server log before predicting coverage.
- Check that a run covered the full window, its error rate and its request count.

## Tooling

### FlyDSL compile cache

- The cache key covers source, not environment variables read at trace time. Toggling an env-gated experiment reuses the previous kernel.
- `FLYDSL_RUNTIME_CACHE_DIR` is used only if the directory exists and is writable; otherwise it silently falls back to the shared cache.
- **Detect:** a "fresh" compile that takes 8 s instead of about 30 s.
- **Fix:** `mkdir -p` a new cache directory per variant, or make variants source changes.

### GPU core dumps

A faulting kernel writes `gpucore.*` files in the working directory, owned by
root when run in a container. Twelve of them took 118 GB and filled the home
quota. Set `ulimit -c 0` in every script that may fault. Delete dumps from
inside the container.

### rocgdb on a multi-rank launch

Wrap only rank 0 in rocgdb (a wrapper script chosen by `LOCAL_RANK`), enable
precise memory reporting, and on the fault print the PC, about 400
instructions before it, the exec mask and the registers of the active lanes.
Look for an address register that holds a float bit pattern.

### GSM8K harness

`sglang.test.few_shot_gsm8k` (5-shot, no chat template, 512 tokens) scored
about 0.91. `sgl-eval` with thinking on scored about 0.97 on the same server.
Use the user's harness. For on vs off, keep parallelism low enough that steps
fit the row limit (MTP3: parallel 2; MTP5: parallel 1), or most steps fall
back to SGLang's path.

## Environment and cluster

- **Image SGLang too old:** "no monokernel backend 'amd' is registered" means `register()` lacks `model_type`. Use an SGLang worktree on `PYTHONPATH` for both arms.
- **AITER JIT locks:** deleting `lock_*` files while another server is building makes waiting ranks import a half-built module. Never clear locks while a server is starting.
- **pkill matches itself:** `pkill -f 'port 30100'` also kills the `bash -c` that runs it. Use a bracketed pattern.
- **srun:** a `docker pull` dies when its srun step ends; keep the step alive. Only one interactive srun per job at a time; `spur exec` runs alongside it.
- **Shared nodes:** another chat of the same user started AgentX on the node mid-test. Check `docker ps` and `rocm-smi` before and after a test batch, and discard overlapping results.
- **Hold expiry:** node holds last 16 hours. Chain follow-up runs into the driver script; the agent cannot start them after its reply ends.
