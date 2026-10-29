# New model: from SGLang's forward to a monokernel package

Read this after SKILL.md Step 0 classifies the job as a new model. It adds three
phases in front of SKILL.md's Steps 1 to 6: an architecture reference, a choice of
base package, and a written design. Each phase ends with a document the user reviews
before the next phase starts. Then bring the package up in the order below; the gate
ladder in SKILL.md Step 4 still applies.

Not yet tested on a real new-model port. When one is done, write its case study and
add what went wrong to pitfalls.md.

## Phase A: Architecture reference from SGLang (before any kernel code)

A monokernel reimplements SGLang's forward for one model. Every detail it gets wrong
(a quantization format, a routing tie-break, a missing RoPE, an extra residual
stream) becomes a numerics bug that is found late. So first write the model's
architecture reference, as a skill folder like `AgentSkill/glm53-flash-architecture/` (its
`SKILL.md` and `layer-structure.md` are the template). Cite SGLang functions and
classes, not line numbers.

The reference must contain:

- [ ] Source files: the SGLang model file, config parser, attention backends, draft
      model, and the SGLang commit they were read at
- [ ] Config values the kernel builds in (hidden size, heads, experts, top-k, ranks,
      norms' eps, scaling factors)
- [ ] Layer types: which layers have which attention and which MLP, as a table
- [ ] Per layer type, every op in order from SGLang's forward: norms, projections,
      attention, residual handling, router, experts, all-reduces
- [ ] Each module's weight dtype and layout, read from the checkpoint's
      `quantization_config` (`exclude`, per-layer configs) and from how SGLang keeps
      it after loading (preshuffled or row-major, fused or separate shared expert)
- [ ] State and caches each layer reads and writes: KV pages, compressed or pooled
      caches, recurrent and conv state, rings, and their pools in SGLang
- [ ] Step types SGLang runs for this model and the speculative algorithm, if any
- [ ] A decode trace at concurrency 1 and the target TP: the kernel list per layer
      type with times, and the launch gaps. This is the gain the monokernel can win

## Phase B: Choose the base package

Map each component to a package that already implements it, then start from the one
that covers the most. Read the PR that added that package in full (`gh pr diff`); it
is the template for the file set. For example, glm53_flash's first PR
([sgl-amd-mono#6](https://github.com/sgl-project/sgl-amd-mono/pull/6)) added
`spec.py`, `check.py`, `gfx950/{kernel,runner,layout,weights,config,packing,tiles}.py`,
and unit, kernel and e2e tests.

Merged packages (from the repo README; check it for the current list):

| Package | Model | Launches per layer | What it covers |
| --- | --- | --- | --- |
| `glm53_flash` | GLM-5.3-Flash | 1, plus a tail launch | KDA linear attention, MLA / DSA with a k-pool indexer, 4-stream mHC residual, MXFP4 MoE, FP8 dense MLP, embedding, LM head and logits |
| `glm53` | GLM-5.3 | 1 | Sparse MLA with an indexer and exact top-2048, FP8 MoE, MTP verify |
| `deepseek_v41` | DeepSeek-V4.1-Flash | 2 (K1, K2), plus K2i on indexer layers | Residual-stream mixing, sparse attention, MoE, DSpark verify and draft stages, up to 48 rows |
| `deepseek_v4` | DeepSeek-V4-Pro | 1 | KV compressors and a CSA lightning indexer with exact top-k, TP8 |
| `minimax_m3` | MiniMax-M3 | 1 | The sparse MoE layers: block-selection sparse attention, MoE, EAGLE3 verify, up to 16 rows; the dense layers stay in SGLang |
| `minimax_m31` | MiniMax-M3.1-Flash | 1 | o_proj and the FFN only (NVFP4 experts); attention stays in SGLang |

Open PRs worth reading for components not yet merged (unreviewed code):
GDN linear attention ([#33](https://github.com/sgl-project/sgl-amd-mono/pull/33)),
Qwen3.8-Flash-Next ([#29](https://github.com/sgl-project/sgl-amd-mono/pull/29)),
Kimi-K3 ([#13](https://github.com/sgl-project/sgl-amd-mono/pull/13)), a dense model
([#28](https://github.com/sgl-project/sgl-amd-mono/pull/28)). Search again with
`gh pr list -R sgl-project/sgl-amd-mono --state all --search "<component>"`.

A component no package has (a new attention type, a new quantization format) is the
risk of the port. Say so to the user in this phase, with an estimate of its size.

A partial first version is a valid milestone: `minimax_m31` covers only o_proj and
the FFN and leaves attention to SGLang. Propose one when a component is new.

## Phase C: Design document

Write it before the kernel and get the user's sign-off:

- Launches per layer and the stage order in each; which stages are copied from the
  base package and which are new
- Per stage: tasks, rows per task, and what each task reads that another task or rank
  writes in the same launch (that sets the waits)
- LDS per row and the row limit per TP size (`WORK_WORDS <= 32768` per CTA)
- All-reduce points and their sizes
- Weights read in place, and weights copied at startup with their size per rank
- SGLang state and caches the kernel reads and writes, and the pad-row rule for each
- Step types served first (decode only is the usual first PR; verify follows, as a
  step-type job) and every fallback

## Bring-up order

Each step passes before the next. Keep one width and one TP size until step 4.

1. **Package skeleton.** `spec.py` with `Requirements`, `config_disable_reasons` and
   `model_disable_reasons`, `SGLANG_READS`; register the spec in `SPECS` in
   `sgl_amd_mono/__init__.py`; add `.github/CODEOWNERS` and README rows. Unit and
   contract tests pass on CPU. The checklist is in the contributing guide
   ([#81](https://github.com/sgl-project/sgl-amd-mono/pull/81), "Adding a Model").
2. **One layer type.** Kernel test against SGLang's reference ops in
   `test/kernels/`, then `check.py` (`SGLANG_AMD_MONO_CHECK=1`) against SGLang's own
   layer on the real weights. Write `check.py` now, not at the end: it is the only
   per-layer comparison on real weights and caches.
3. **All layer types**, then the first and tail launches if covered. Eager server
   with check mode on: zero violations. A short GSM8K, on vs off.
4. **CUDA graphs, every width and TP size** the spec declares.
5. **SKILL.md gates G4 to G9.**

## Learning FlyDSL and the shared runtime

There is no FlyDSL guide in this repo; learn from the base package's kernel and
`sgl_amd_mono/common/`:

| Module | Role |
| --- | --- |
| `plan.py` | Execution model, layouts, TP shards, build keys |
| `ops.py` | Wave ops, hardware math, packing, buffer access |
| `sync.py` | Tagged mailbox between CTAs (device scope) and GPUs (system scope) |
| `guard.py`, `epoch.py` | Wait deadlines, sticky error words, launch epochs |
| `peer.py` | Symmetric peer memory for in-kernel TP collectives |
| `mx.py` | FP8 / MX arithmetic |
| `stamps.py` | Per-stage clock marks for profiling (debug builds) |

Reuse these and AITER's FlyDSL helpers (`aiter.ops.flydsl.kernels`) instead of
copying them.

## What first ports missed

Fixes that followed glm53_flash's first PR; check each in a new port:

- Router ties and long rows' keys must follow SGLang
  ([#27](https://github.com/sgl-project/sgl-amd-mono/pull/27))
- Index-cache checks must compare dequantized keys
  ([#54](https://github.com/sgl-project/sgl-amd-mono/pull/54))
- A step's first launch needs half the watchdog as its wait deadline
  ([#61](https://github.com/sgl-project/sgl-amd-mono/pull/61))
