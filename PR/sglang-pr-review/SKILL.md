---
name: sglang-pr-review
description: >-
  Review an sgl-project/sglang pull request for object-oriented structure,
  software engineering, and SGLang conventions, including AMD gates
  (is_hip, _use_aiter, is_gfx95_supported, _use_aiter_gfx95) and registered
  tests for new kernels. Use when the user asks to review an SGLang PR,
  a sglang pull request, or sgl-project/sglang changes. Draft the review
  locally; do not post it to GitHub until the user approves.
---

# SGLang PR Review

Draft the review in the chat. Do not run `gh pr review`, `gh pr comment`, or any
review/comment API until the user explicitly says to post it.

Review the diff for three things: object-oriented structure, ordinary software
engineering, and the way this repo already writes the same kind of change.
Correctness and measured performance stay in scope only when they fall out of
those three.

## Fetch the change

```bash
gh pr view <number> --repo sgl-project/sglang --json title,files,additions,deletions
gh pr diff <number> --repo sgl-project/sglang
```

Read the surrounding function in a local sglang tree (or the PR head), not only
the added lines. The gate that matters is the one already used in that file.

## AMD-only code is gated

If the change is AMD-only, every new branch in a file that also runs on CUDA,
CPU, NPU, or MI300 must be unreachable unless the hardware gate is true.
`SGLANG_USE_AITER` alone is not a gate: it does not imply HIP, and HIP does not
imply gfx95. An env selector alone is not a gate either.

Use the flag the file already imported. Do not invent a parallel check.

| Intent | Symbol | Defined in |
| --- | --- | --- |
| HIP build | `is_hip()` / `_is_hip` | `sglang.srt.utils` |
| aiter installed and selected | `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip` | module level, or `deepseek_common.utils` |
| gfx95 (false on every non-HIP build) | `is_gfx95_supported()` | `sglang.srt.utils.common` |
| aiter on gfx95 | `_use_aiter_gfx95` | `sglang.srt.models.deepseek_common.utils` |
| aiter gfx95 and ROCm ≥ 7.2 (bpreshuffle GEMM) | `_use_aiter_bpreshuffle_gfx95` | same module |

`is_gfx95_supported()` already returns false off HIP, so a gfx95 kernel does not
need a second `is_hip()` at the launch site. `_use_aiter` must still be ANDed
with `is_hip()`, because aiter is a HIP package. `glm5_next.py` and the other
DeepSeek-family models already import `_use_aiter_gfx95`; a new GLM/DeepSeek
AMD path uses that flag.

Gate the decision where the structure changes, not only where the kernel
launches. Disabling a fused module, swapping `quant_method`, or allocating a
second weight all happen in `__init__` or `process_weights_after_loading`. A
load-time check that leaves the module graph already rewritten still changes
CUDA when the env var leaks.

Default behavior is the bar. Env unset, CUDA, MI300, and aiter off must keep
the same modules and the same `apply`. If the opt-in is set on the wrong
platform, raise. Do not silently unfuse or fall through.

On a shared hot path (`UnquantizedLinearMethod.apply`, a linear forward, an
attention forward), the first condition is a module-level bool such as
`_use_aiter_gfx95`. CUDA then pays one constant check. A new helper call in
front of that bool runs on every GPU.

Import `aiter` at module level under `if _use_aiter:`, not inside `forward`.

## Object-oriented shape

Match the extension point that already exists. A new linear behavior is a
`LinearMethodBase` subclass swapped onto that layer. It is not a set of private
attributes (`_fp8_ptpc_ready`, `_glm53_*`) that `UnquantizedLinearMethod.apply`
interprets for every model.

Keep the split the repo already uses:

| Layer | Owns |
| --- | --- |
| Model module | Which projections opt in, shared activation quant, fusion on or off |
| Quant method | Weight repack, when this layer consumes FP8, the GEMM call |
| Kernel function | The math, plus a `can_use_*` predicate that refuses shapes it did not specialize |

Do not drop `x: torch.Tensor` so a `(fp8, scale)` tuple can pass through
`LinearMethodBase.apply`. Name the union on the subclass that actually accepts
it. `ColumnParallelLinear.forward` and `RowParallelLinear.forward` assume a
tensor before `quant_method.apply` whenever sequence parallel is on or
`input_is_parallel` is false; a tuple that only works for the default
constructor is part of the review.

One policy, one place. A token-count threshold checked in the model and again
with a different predicate inside `apply` will drift.

## New kernels need a registered test

A new Triton or JIT kernel under `python/sglang/kernels/` ships with a test
under `test/registered/`. Registration is an AST marker; the runtime call is a
no-op, and the file still needs `if __name__ == "__main__"` with a real call.

Pick the suite from the hardware the test needs, not from the directory or a
default. A test registered on a runner that fails its own `skipUnless` is
skipped on every CI run, which is the same as having no test.

`stage=..., runner_config=...` resolves to suite `{stage}-test-{runner_config}`.
Look up that suite's `runs-on` in `.github/workflows/pr-test-amd.yml` (and
`pr-test-amd-extra.yml`) at the PR head before accepting it:

| Test needs | Registration | Runner (`runs-on`) |
| --- | --- | --- |
| Any HIP GPU, MI300 is fine | `register_amd_ci(est_time=..., stage="jit-kernel-unit", runner_config="amd")` | `linux-{mi300\|mi325}-1gpu-sglang` |
| gfx95 (MI355X), 1 GPU | `register_amd_ci(est_time=..., stage="stage-b", runner_config="1-gpu-small-amd-mi35x")` | `linux-mi35x-gpu-1` |
| gfx95 (MI355X), 8 GPU | `register_amd_ci(est_time=..., stage="stage-c", runner_config="large-8-gpu-amd-mi35x")` | `linux-mi35x-gpu-8` |
| Selector / gate / fallback, no GPU | `register_cpu_ci(est_time=..., suite="base-a-test-cpu")` | CPU |

AMD suites without `-mi35x` in the name run on `inputs.runner_arch`, whose
choices are `mi300` and `mi325`; none of them reach gfx95. So a test guarded by
`is_gfx95_supported()` must sit on a `-mi35x` suite, and a test on
`jit-kernel-unit-test-amd` must not require gfx95. Match the neighbors in the
same directory that already test gfx95 kernels. The legacy `suite="..."` string
reaches the same job as the stage/runner pair; either is fine as long as the
runner matches.

The test has to hit the specialized shapes: constexpr heads, local K, and the
M boundaries where the code switches back to BF16. A toy `N` that is not the
projection size does not cover the GEMM. Include the platform where the gate
must refuse (env set, not gfx95, aiter off), especially when `__init__` changes
which modules exist.

## What to look for

- Shared file, AMD behavior, no `_use_aiter_gfx95` (or the file's existing flag) around the structural change.
- Extra Python work in `UnquantizedLinearMethod.apply` before a module-level AMD bool.
- Model-specific protocol stuffed into the generic unquantized method.
- `import aiter` inside a forward or `apply`.
- New kernel with no `test/registered/...` file, or a test that skips the real shape.
- A gfx95-only test (`skipUnless(is_gfx95_supported())`) registered on a suite whose runner is MI300/MI325, so CI always skips it.
- Type annotation removed to hide a second input kind.
- Two copies of a threshold, a scale layout, or a module-name set that have to stay in sync.

## Draft format

Write the surrounding explanation in the user's language. Write the paste-ready
GitHub comments in English. Do not post them.

```markdown
# Review draft: <title>
<url>

Not posted.

## Verdict
Request changes / Comment / Approve — one sentence.

## Must fix
1. **<short title>** — `<file>` `<symbol>`. What changes off the AMD gate, and what to do instead.

## Should fix
1. **<short title>** — same shape, smaller blast radius.

## Nits

## What already matches the repo

## Comments ready to paste
### `<file>` — `<symbol>`
<comment a reviewer could leave as-is>
```

Each finding names the symbol and the platform it affects (CUDA with the env
unset, CUDA with the env set, gfx95 with aiter off). "Add a check" is not a
finding; say which existing flag, at which call site.

## Keep it short

Every paste-ready comment, and the top-level review body, stays under 100
words. One comment carries one finding: the symbol, what breaks on which
platform, and the fix. Do not restate the PR, praise it, or hedge. Keep a table
or a measurement if it carries the point, and cut the prose around it. If a
finding does not fit in 100 words, split it or cut it. Do not repeat a point
another reviewer already left inline; reference it in one line.
