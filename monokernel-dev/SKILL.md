---
name: monokernel-dev
description: "Workflow for building or extending an sgl-amd-mono monokernel (FlyDSL persistent decode kernels for SGLang on AMD MI355X / gfx950): adding a step type such as MTP / EAGLE / DSpark verify, raising the row limit, or porting a new model. Covers scoping, studying precedents, environment setup, the verification gate ladder (bitwise equivalence, long-context limits, microbenchmarks, GSM8K, fixed-length and AgentX benchmarks), profiling, run hygiene and reporting. Use when the user asks to build, port, extend, debug or benchmark a monokernel, mentions sgl-amd-mono, --enable-amd-monokernel or --monokernel-backend amd, or wants a model's decode or verify steps on the monokernel."
---


# Monokernel development (sgl-amd-mono)

A monokernel is a persistent kernel that runs most of a decoder layer (attention, MoE, both TP all-reduces) in one or two launches. Each model package in `sgl_amd_mono/<model>/` has `spec.py` (what SGLang may route to it, and the forward), optionally `check.py`, and `gfx950/` (`kernel.py`, `runner.py`, `layout.py`, `weights.py`, `config.py`). Kernel tests live in `test/kernels/`, CPU tests and the SGLang contract in `test/unit/`.

Worked example: [case-study-glm53-flash-mtp.md](case-study-glm53-flash-mtp.md).
Detailed failure cases: [pitfalls.md](pitfalls.md).
New model: [new-model.md](new-model.md).

Tested with: **Claude Opus 5.5 at High effort** in Cursor agent mode. 

## Step 0: Scope and inputs

Classify the job; say which it is in your first reply:

- **Add a step type** to an existing kernel. A step type is a kind of forward step SGLang runs: decode (1 token a request), MTP / EAGLE target verify (steps + 1 tokens a request, 4 for MTP3), draft extend, prefill. Most kernels start with decode only; e.g. add MTP verify.
- **Raise the row limit** (check the LDS budget first: `WORK_WORDS <= 32768` per CTA, and each row's activations live in LDS).
- **New model** (far larger). Follow [new-model.md](new-model.md) first: an architecture reference from SGLang, the choice of base package, a design document and the bring-up order, each reviewed by the user.

Collect these before writing code. Ask for any that are missing and have no default:

- [ ] Precedent PRs adding the same feature to another model
- [ ] Dev environment: image tag, SGLang commit or branch, model path
- [ ] Baseline: file, numbers, metric, how measured
- [ ] Accuracy harness. Default: `python3 -m sglang.test.few_shot_gsm8k --num-questions 1319`, on vs off. An expected score is optional; without one, compare on to off.
- [ ] Workload shape: long context, default 400K tokens; decode batch-size distribution



## Step 1: Study before coding

1. Find precedents: PRs in sgl-amd-mono that already added the same step type or feature to another model, or, for a new model, the PR that added the closest package (`gh pr list -R sgl-project/sgl-amd-mono --state all --search "<feature>"`). For GLM-5.3-Flash MTP verify they were #15 (glm53) and #29 (qwen38_flash_next). If there are none, tell the user. Read their full diffs (`gh pr diff`). Copy their spec patterns: `features`, accepted `speculative_algorithm`, `tokens_per_request`, fallbacks, scratch-row indexing.
2. Read the target package end to end: `spec.py` forward, `runner.py` argument passing, the stage functions in `kernel.py`.
3. Read SGLang's semantics for the step type, in the SGLang version you will serve with. For MTP on hybrid models, this means the `intermediate_ssm` and conv-window scratch layouts, the post-verify commit, the k-pool tail ring and its extra slots, and the pad-slot convention (-1).
4. Write down the design: tasks per stage, what each row reads that another row in the same launch writes, and the ordering and synchronization that implies.



## Step 2: Environment

- The serving SGLang must accept this repo's plugin. At startup SGLang runs `sgl_amd_mono.register_specs()`, which calls SGLang's `register(..., model_type=...)` once per model to say which spec serves it. `model_type` came with sgl-project/sglang#46036; an older `register` rejects it, no spec is registered, and the monokernel never runs. The server flag is `--enable-amd-monokernel` (`--monokernel-backend amd` before sgl-project/sglang#45389). If the image's SGLang is older, cherry-pick or put an SGLang worktree on `PYTHONPATH` for **both** off and on.
- Install sgl-amd-mono editable in the container. If git reports "dubious ownership": `git config --global --add safe.directory "*"`.
- Before any GPU work, run `rocm-smi` and `docker ps` on the node. Recheck before every long run.
- Set `ulimit -c 0` and a fresh, existing `FLYDSL_RUNTIME_CACHE_DIR` in every script that may fault or that compares kernel variants.



## Step 3: Implement

- Keep the decode path bitwise unchanged unless the change is intentional.
- Graph safety: anything a step table or kernel keys by pointer must live in fixed per-width buffers; fill them with `copy_`, never by rebinding new tensors.
- Pad rows and pad requests must be harmless: zeroed outputs, no state writes.
- Fallback: every step the kernel can't serve (too many rows, wrong tokens per request, unsupported mode) must route to SGLang's path, and the server log must show which batch sizes are served.



## Step 4: Gate ladder

Pass each gate on the **exact revision** before the next. After any kernel
change, restart from gate 3. Record each result with its commit.

```text
- [ ] G1 Lint (pinned pre-commit versions), unit + contract tests (CPU)
- [ ] G2 Compile every shape the spec will serve
- [ ] G3 Layer correctness: bitwise vs the existing monokernel path, same width (e.g. verify == L decode launches); and vs SGLang's own layer within `spec.NUMERICS` (`SGLANG_AMD_MONO_CHECK=1`, eager server, zero violations)
- [ ] G4 Limits: long context (default 400K), max rows, pad rows, chain-internal cases
- [ ] G5 Microbenchmark across the workload's context range, before vs after, and vs main
- [ ] G6 E2E accuracy on vs off: same flags, the user's harness, parallelism low enough that the kernel is used
- [ ] G7 Fixed-length serving on vs off (e.g. 8K/1K at conc 1, plus one point past the row limit)
- [ ] G8 Ten-minute smoke of the real workload: no faults, server log shows routing
- [ ] G9 Full agentic benchmark with a same-node baseline, full duration
```

GLM-5.3-Flash commands (adapt for other packages):

```bash
pytest test/unit -k glm53_flash
torchrun --nproc-per-node 4 test/kernels/glm53_flash/check_layer.py --kind dsa --rows 8 --verify 4
L="--kind dsa --max-len 409600 --max-pools 262144"   # 400K rows, SGLang's 1M-context pools
torchrun --nproc-per-node 4 test/kernels/glm53_flash/check_layer.py $L --rows 8 --launches 2 --lens 262150,331001,400001,65537,3000,1,0,0
torchrun --nproc-per-node 4 test/kernels/glm53_flash/check_layer.py $L --rows 8 --verify 4 --lens 400001,300003
for p in 8191 131071 393215; do torchrun --nproc-per-node 4 test/kernels/glm53_flash/time_layers.py --kind dsa --rows 8 --pos $p; done
pytest test/kernels -k glm53_flash        # 8 GPUs, full suite, not only the new test
```

Gate rules:

- G3's bitwise check compares at the **same width**. Different widths change tiling and give 1e-7 differences that aren't bugs.
- G3 against SGLang can't be bitwise: the fused layer rounds and accumulates in a different order. It is a tolerance check per output, state and cache, and it is required for a new model or any change to the math; a bitwise match only proves the new path equals the old monokernel path. If `check.py` doesn't cover the new step type (glm53_flash's covers decode only), say so in the PR.
- G4 lengths must be representative of the workload, not test defaults: 8K is far too short for agentic traces, so test 400K tokens unless the user gives another length. The exact maximum varies run to run; don't chase it. Include lengths just past power-of-two pool and tile boundaries (e.g. 262,148 tokens completes pool 65,536).
- G5 runs before and after every fix. A fix that removes a fault can still add a hotspot.
- G6: use the user's harness, or GSM8K by default (Step 0). Set `--parallel` so every step fits the row limit: rows per step = requests × tokens per request, so at high parallelism the kernel may never run (GLM-5.3-Flash at TP4: parallel 2 for MTP3, 1 for MTP5).



## Step 5: When results surprise you

- A regression is a bug until profiling says otherwise. Do not report an explanation you haven't measured. Label hypotheses as hypotheses.
- If two results disagree (e.g. microbenchmark faster, agentic slower), reconcile them before reporting either.
- Profile inside the kernel: stamp `common/ops.py` `clock()` (100 MHz realtime) per CTA per stage into a scratch region, then compare stage times. See [pitfalls.md](pitfalls.md).
- Memory faults: reproduce with the kernel test, bisect the boundary, then use rocgdb with precise memory on rank 0. See [pitfalls.md](pitfalls.md).
- Before calling a failure pre-existing, reproduce it on `main` on an idle node.



## Step 6: Reporting

- Lead with the outcome and the comparison it came from (same-node or CSV baseline, duration, error rate, request count).
- Tag every number with commit, image, SGLang commit and node. If a number predates the current head, say so or rerun it.
- Before the user sends anything outside (PR, manager, colleagues), verify every claim about the repo against `main` and all PRs, open and closed, and cite them.
- Keep a results ledger file under shared storage; chat summaries lose details.
- For PRs to sgl-amd-mono, use the `sgl-amd-mono-pr` skill.



## Run hygiene

- One chat per node. If another job appears on your GPUs, stop, tell the user, and discard results that overlapped.
- Stop servers with SIGINT or SIGTERM; match processes with a bracketed pattern (`pgrep -f '[s]glang.launch_server'`) so the command doesn't match itself.
- You cannot act after your reply ends. Chain follow-up runs into the driver script, or tell the user exactly what is left to start.
- Check job and hold expiry before planning multi-hour work.

