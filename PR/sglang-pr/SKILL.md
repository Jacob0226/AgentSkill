---
name: sglang-pr
description: Write pull request descriptions for sgl-project/sglang in the Summary / Scope / Test plan format, each section capped at 100 words with the detail carried by tables. Use when opening, drafting, or rewriting an SGLang PR body, especially for AMD/ROCm performance work on MI355X.
---
# SGLang PR Description

Three sections, in this order: **Summary**, **Scope**, **Test plan**. Nothing else.

No `Changes` section. Summary already says what the change does; a file-by-file list only
restates the diff the reviewer is about to read.

## The 100-word rule

Each section's **prose** is capped at 100 words. Tables and code blocks do not count.

That cap is the format. A reviewer should get three short paragraphs and three tables, not
three pages. When a section runs long the fix is to move detail into a table, not to shave
words evenly off every sentence.

## What each section carries

### Summary

The mechanism, then the change. Most SGLang PRs are performance work, so lead with *why the
current code costs what it costs* — a reviewer who understands the mechanism can predict the
speedup without reading the diff. Symptom-first summaries ("X is slow") waste the budget.

Carry the evidence in a table: the kernel, the key, or the counter that moves.

### Scope

Which hardware, model, and code path the change touches, what it is gated on, and —
explicitly — what keeps current behaviour. Reviewers approve narrow changes, so state the
narrowness: other GPUs, other quantization paths, and ungated defaults all deserve a mention
when they are untouched.

### Test plan

Name the exact base first, or the numbers mean nothing: the docker image tag and the PR stack
applied on top of it. Then state which **single variable** separates the two arms.

Tables follow: end-to-end latency, then kernel-level attribution if the change is a kernel
change, then accuracy.

## Rules

- **Measured numbers, never adjectives.** "163.5 → 136.6 ms", not "significantly faster".
- **One variable.** Both arms share an image, a model, a TP, and a PR stack; say so. If
  anything else moved, the table is not evidence.
- **Name the thing.** The kernel symbol, the tuning key, the env var — a reviewer greps for it.
- Bold the deltas worth reading. Leaving noise-level rows unbolded is itself information.
- Report **mean TTFT** and **median TPOT**, and sweep concurrency rather than quoting one
  point. **Name the statistic in the table header**, never a bare "TTFT": `bench_serving`
  prints mean, median and P99 for each, and they diverge — one GLM-5.2 run at i8k/conc4 reads
  492.08 mean against 389.20 median TTFT, so an unlabelled column is unreproducible.
- Always include accuracy, with the caveat: "GSM8K 0.933 → 0.927, single scoring pass each —
  within what one pass resolves." A small accuracy move that goes unmentioned reads as hidden.
- Explain a trend when the table shows one — e.g. a per-forward saving grows with concurrency
  because deeper queues wait behind more forwards. One sentence, inside the word budget.
- If the change fixes a regression, link the PR that introduced it.

Numbers come from `~/SGLang-benchmarks/` bench logs and the step3 trace workbooks; see the
`sglang-benchmark` skill for how to produce them.

## Worked example

[#32888](https://github.com/sgl-project/sglang/pull/32888), condensed. A chunked-prefill
change: this is the shape a performance PR should take. Its sections run near 50 words, well
inside the cap — treat 100 as the ceiling, not the target.

```markdown
## Summary

A prefill batch never reaches `chunked_prefill_size`: `_update_prefill_budget` charges the
page-ceiled extend length, and `add_one_req` floors the last truncation back to a page. On
gfx95, landing 16 tokens short doubles the MLA absorb bmm — its `EVEN_MN` heuristic is a
compile-time `constexpr`, so a misaligned M selects a different compiled binary.

|                        | before             | after              |
| ---------------------- | ------------------ | ------------------ |
| trace token annotation | `toks=16350`       | `toks=16384`       |
| MLA absorb bmm         | 311.2 us @ M=16368 | 155.8 us @ M=16384 |
| instructions           | 3819               | 1562               |

`compute_charge` moves the compute budgets in raw tokens while the KV budgets stay page-ceiled.

## Scope

gfx95 (MI355X) chunked prefill, gated on `SGLANG_EXACT_CHUNK_FILL` **and** gfx95. CUDA and
other AMD parts keep upstream behaviour on every path. The dLLM path is excluded.

## Test plan

Baseline `rocm/sgl-dev:v0.5.15.post1-rocm720-mi35x-20260714` plus six AMD PRs (#30519, #30575,
#30715, #30808, #31323, #31324); this PR adds ROCm/aiter#4453. GLM-5.2-MXFP4, MI355X TP4,
in8192 / out1024.

| Concurrency | Mean TTFT base | this PR | Δ         | Median TPOT base | this PR | Δ         |
| ----------- | -------------- | ------- | --------- | ---------------- | ------- | --------- |
| 4           | 512.25         | 484.58  | **−5.4%** | 13.04            | 13.00   | −0.3%     |
| 8           | 727.92         | 681.18  | **−6.4%** | 16.41            | 16.23   | −1.1%     |
| 16          | 999.91         | 930.41  | **−7.0%** | 21.91            | 21.61   | **−1.4%** |
| 32          | 1558.20        | 1442.90 | **−7.4%** | 31.00            | 30.37   | **−2.0%** |
| 64          | 2742.82        | 2481.36 | **−9.5%** | 49.83            | 48.39   | **−2.9%** |

Mean TTFT improves monotonically with concurrency, which is what a shorter prefill should look
like: the saving is per prefill forward, so the deeper the queue, the more of them a request
waits behind.

Prefill kernel attribution, one forward at in8192 / concurrency 64:

| Category                  | baseline (ms) | this PR (ms) | Δ          |
| ------------------------- | ------------- | ------------ | ---------- |
| dense/linear GEMM         | 173.15        | 107.18       | **−65.97** |
| sparse-MLA attn           | 177.39        | 179.49       | +2.10      |
| all-reduce/comm           | 134.31        | 134.79       | +0.48      |
| MoE moe1 + moe2 + routing | 143.81        | 144.59       | +0.78      |
| DSA indexer + topk        | 24.91         | 25.15        | +0.24      |
| rmsnorm/quant/act         | 26.09         | 25.83        | −0.26      |
| Σ kernel                  | 709.34        | 646.71       | −62.62     |

Only dense/linear GEMM improves, as expected; the other categories shift within noise.

**Accuracy:** GSM8K 0.933 → 0.927, single scoring pass each — within what one pass resolves.
```

Note what the attribution table buys: it shows the saving landed *where the mechanism
predicted* and nowhere else. A summary that claims a GEMM win and a table that shows attention
also moving means the arms differed in something else.

## Opening the PR

Work in a throwaway worktree off `upstream/main`, never in a tree that has unrelated changes:

```bash
cd ~/PR/sglang
git fetch upstream main
git worktree add -b jacob/<topic> ~/PR/wt-<topic> upstream/main
```

Commit with the identity explicitly, since a fresh worktree inherits no local git config and
`~/.gitconfig` may be unset. Pass it per-command rather than writing config:

```bash
git -c user.name="Jacob0226" -c user.email="jacchang@amd.com" commit -F - <<'EOF'
...
EOF
git push origin jacob/<topic>
gh pr create --repo sgl-project/sglang --draft --base main \
    --head Jacob0226:jacob/<topic> --title "[AMD] ..." --body "$(cat <<'EOF'
...
EOF
)"
```

The commit message carries the long-form reasoning the 100-word sections cannot. Title starts
with `[AMD]`, plus `[gfx95]` when the change is gated on the architecture.

Open as **draft** until CI is green. A maintainer adds the `amd` and `run-ci` labels; CI does
not run on the AMD boxes without them.
