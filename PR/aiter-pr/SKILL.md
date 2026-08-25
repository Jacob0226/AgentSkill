---
name: aiter-pr
description: Write pull request descriptions for ROCm/aiter in the Motivation / Test Plan / Test Result format, each section capped at 100 words with the detail carried by tables. Use when opening, drafting, or rewriting an aiter PR body, especially gfx950 kernel-config and tuning work.
---
# aiter PR Description

Three sections, in this order: **Motivation**, **Test Plan**, **Test Result**. Nothing else.

No `Technical Details` section. Motivation already states the mechanism, and the diff of a
config PR *is* the technical detail — restating it burns the reader's budget twice.

## The 100-word rule

Each section's **prose** is capped at 100 words. Tables and code blocks do not count.

The cap is the format. A reviewer should get three short paragraphs and the tables that carry
the evidence. When a section runs long, move detail into a table rather than shaving words off
every sentence.

Every PR needs at least one table, and it must show **what the PR improved** — a before/after
column pair over the swept dimension, not a description of the change.

## What each section carries

### Motivation

Name the exact lookup that is wrong and what the code does instead. aiter dispatches through
tuned tables keyed by shape, so the mechanism is almost always "shape X has no entry / a stale
entry, so it falls through to Y, which is tuned for a different regime." Say which key, which
fallback, and why the fallback is wrong for this shape.

For a model-driven PR, name the model dims that produce the shape, so a reviewer can confirm
the shape is real rather than synthetic.

### Test Plan

Name the hardware, the ROCm and Triton versions, and the docker image tag — a tuning result is
meaningless without them. Then give the **exact command**, and explain the argument convention
if it is not obvious (`--shape` is `B M N K`; `B=16` is GLM-5.2 TP4's local head count).

State how the two arms are produced: for a config PR, that is usually moving the JSON or CSV in
and out of the config directory, which is the cleanest one-variable A/B available.

When a tuner run produced the new config, the section holds **two commands with two different
jobs**, and a reader who cannot tell them apart will ask why you ran the same thing twice. Label
them as ordered steps and say outright that the tuner's output *is* the diff:

    **Step 1, produce the rows** — the diff here is this run's output. <tuner command>
    **Step 2, check they are faster** — <benchmark command>

Step 1 earns its place by making the PR reproducible, not by proving anything; say which tuner
entry point, because a reader who reruns the default one will not reproduce your rows. Step 1
never doubles as evidence — see the tuner-`us` warning above.

### Test Result

Lead with the kernel table, swept over the dimension the entry is gated on (usually M or token
count). Then, if the PR is motivated by a model, the end-to-end table.

State explicitly that no swept point regressed, or name the ones that do and why they are
acceptable. A reviewer's first question about a tuning change is "what did you make slower?"

## Rules

- **Never hard-wrap the body.** One line per paragraph, however long. GitHub soft-wraps to the
  reader's width, so inserting your own newlines only fixes the text to a width nobody has and
  makes every later edit rewrap the whole paragraph. This applies to the PR body and to
  comments; the commit message still wraps at 72.
- **Measured numbers, never adjectives.** "409.2 → 163.0 us", not "much faster".
- **One variable.** Both arms share the image, the versions, and everything but the config
  under test. If anything else moved, the table is not evidence.
- **Sweep, never a single point.** A tuning entry is gated on a range; show the whole range,
  including the points just below the gate that must not change.
- **Name the gate.** `M_GEQ_512`, `block_m=16`, the CSV key — a reviewer greps for it.
- Bold the wins. Leaving noise-level rows unbolded is itself information.
- **Show the untouched region.** "every M <= 511 resolves to the same entry as before, field
  for field" is what makes a config PR safe to approve.
- If the change fixes a regression, link the PR that introduced it and show its own recorded
  numbers — see below.

## Tuned-CSV numbers are not evidence on their own

aiter's tuned tables record a `us` column, and it is tempting to quote it. Do not, without
checking which tuner wrote it.

The fused-MoE table is written by three mutually exclusive tuner entry points
(`FmoeTuner` by default, `Mxfp4FlydslTuner` under `--mxfp4-flydsl`, `GroupedFmoeTuner` under
`--grouped-gemm`), and they do not time the same thing: the mxmoe path times one end-to-end
launch with `moe_sorting` inside the timed region and reports it as `us1` with `us2=0`, while
the default path times stage1 and stage2 separately with sorting precomputed outside. Rows from
different entry points are therefore not comparable, even though they share a column.

Worse, the dedupe key excludes `block_m` and the kernel name, and `update_tunedf` overwrites
in place with `drop_duplicates(..., keep="last")` and no regression guard unless
`--compare --update_improved` is passed. A sweep that enumerates only one kernel family will
silently replace a faster entry from another family.

So: quote the incumbent's recorded `us` only to show *that the table disagrees with itself*,
and always carry your own apples-to-apples measurement as the real evidence.

## Worked example

[#4453](https://github.com/ROCm/aiter/pull/4453), condensed. A pure-config PR — this is the
shape one should take.

```markdown
## Motivation

gfx950 `BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` has no specialized file for the GLM-5.2 MLA absorb BMMs (`w_kc` N=512/K=192, `w_vc` N=256/K=512), and the generic table stops at `M_LEQ_256`, so large-M prefill falls through to `"any"` (32x128 / `num_warps=8`) — a smaller tile than `M_LEQ_256` already uses. A prefill-shaped M runs on a decode-shaped tile.

Two JSON configs, no code. A per-(N,K) file replaces the generic dict outright, so `M_LEQ_*` and `"any"` are copied verbatim and nothing below the new bucket changes.

## Test Plan

MI355X (gfx950), ROCm 7.2, Triton 3.7, `rocm/sgl-dev:v0.5.17-rocm720-mi35x-20260811`. Kernel A/B with aiter's own benchmark, moving the two JSONs in and out of `aiter/ops/triton/configs/gemm/`:

    BENCH=op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py
    python3 $BENCH --shape 16 16384 512 192 --metric time --no-bias                 # w_kc
    python3 $BENCH --shape 16 16384 256 512 --metric time --no-bias --transpose-bm  # w_vc

`--shape` is B M N K; B=16 is GLM-5.2 TP4's local head count, and the absorb path takes no bias.

## Test Result

**Kernel, us/call** — the command above, swept over M:

| M     | w_kc without | w_kc with | w_vc without | w_vc with |
| ----- | ------------ | --------- | ------------ | --------- |
| 512   | 17.7         | **12.1**  | 14.5         | 14.4      |
| 1024  | 30.7         | **16.1**  | 25.0         | **18.2**  |
| 2048  | 56.0         | **26.5**  | 46.2         | **23.3**  |
| 4096  | 107.7        | **47.2**  | 87.9         | **43.2**  |
| 8192  | 210.6        | **87.7**  | 171.9        | **80.7**  |
| 16384 | 409.2        | **163.0** | 321.8        | **142.2** |

2.5x / 2.3x at M=16384, and no M is slower than before — `w_vc` at M=512 still resolves to `"any"`, unchanged.
```

Note what the M=512 row buys: it is the point where the new entry is *not* a win, and shipping
it in the table is why the gate moved to `M_GEQ_1024` for `w_vc`. A sweep that only showed
M=16384 would have hidden that.

## Opening the PR

Work in a throwaway worktree off `upstream/main`:

```bash
cd ~/PR/aiter
git fetch upstream main
git worktree add -b jacob/<topic> ~/PR/wt-aiter-<topic> upstream/main
```

Commit with the identity explicitly, since a fresh worktree inherits no local git config and
`~/.gitconfig` may be unset. Pass it per-command rather than writing config:

```bash
git -c user.name="Jacob0226" -c user.email="jacchang@amd.com" commit -F - <<'EOF'
...
EOF
git push origin jacob/<topic>
gh pr create --repo ROCm/aiter --draft --base main \
    --head Jacob0226:jacob/<topic> --title "..." --body "$(cat <<'EOF'
...
EOF
)"
```

The commit message carries the long-form reasoning the 100-word sections cannot.

Do **not** hand-write the component tags in the title — a bot rewrites it to
`[Triton/Gluon] [Config] [gfx950] ...` from the paths touched. Write the plain descriptive
title and let it prefix.

Open as **draft** until CI is green.

### CI labels

CI runs pre-checks, aiter op tests (gfx942 + gfx950), and Triton tests on MI35X automatically.
Everything else is opt-in via labels, added in the sidebar or with
`gh pr edit <n> --add-label <label>`:

| Label            | Runs                                                       |
| ---------------- | ---------------------------------------------------------- |
| `ci:triton-300x` | Extra Triton job on MI300X                                 |
| `ci:sglang`      | SGLang integration: DeepSeek-R1-MXFP4, Qwen 3.5 accuracy   |
| `ci:atom`        | ATOM benchmark: DeepSeek-R1-0528, GPT-OSS-120B             |
| `ci:atom_full`   | ATOM accuracy suite — **only** for FlyDSL or Triton upgrades |
| `ci:vllm`        | vLLM benchmark: GPT-OSS-120B, DeepSeek-R1-0528, Kimi-K2.5  |
| `ci:all`         | All standard extended tests (excludes `ci:atom_full`)      |

Read `ci:atom_full`'s restriction literally: **upgrades** to FlyDSL or Triton, meaning the
library or compiler itself. A tuned config that happens to select FlyDSL kernels is not an
upgrade and must not carry it.

Default to no extra label and justify each one you add. A tuned table is keyed by shape, so it
can only affect the model whose shape it carries — a `glm5_fp4_tuned_fmoe.csv` row cannot reach
`ci:sglang`'s DeepSeek-R1 and Qwen 3.5 or `ci:atom`'s DeepSeek-R1-0528 and GPT-OSS-120B, and
asking for them burns machine time on a run that cannot observe the change. Add a label when
the change reaches a model the suite actually runs, or when it touches dispatch rather than a
single keyed row.
