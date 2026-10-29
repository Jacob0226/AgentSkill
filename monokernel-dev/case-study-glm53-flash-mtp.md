# Case study: adding MTP to the GLM-5.3-Flash monokernel with an agent

How one engineer, new to monokernels, directed a coding agent to make the
sgl-amd-mono GLM-5.3-Flash FP4 monokernel serve MTP verify steps, and what the
back-and-forth looked like. Times are UTC+8. The companion guide,
[directing-the-agent.md](directing-the-agent.md), turns the mistakes below into
rules.

## Summary

| | |
|---|---|
| Task | Extend an existing kernel, not build one: the GLM-5.3-Flash decode monokernel already existed ([#6](https://github.com/sgl-project/sgl-amd-mono/pull/6), [#22](https://github.com/sgl-project/sgl-amd-mono/pull/22), [#23](https://github.com/sgl-project/sgl-amd-mono/pull/23)). Added MTP verify steps of up to 8 rows: MTP3 at batch size 1 and 2, MTP5 at batch size 1 (TP4). |
| Outcome | Draft PR [sgl-amd-mono#71](https://github.com/sgl-project/sgl-amd-mono/pull/71), 7 files, +753 / −77, plus a test fix |
| Agent | Claude Opus 5.5, High effort, Cursor agent mode, one chat from Oct 21 to Oct 29 (summarized several times) |
| Human input | A one-sentence request, a node, a baseline file, and about ten decisions or pushbacks. No implementation guidance. |
| Time | First working end-to-end result 76 minutes after the request. Draft PR about 8 hours after the request. |
| GPU cost | Roughly 85 MI355X GPU-hours (8 GPUs on node 068 for about 7.7 h, plus reruns on 168 and 097). Estimate from log timestamps. |

**Final numbers (PR head, MI355X TP4)**

| Test | Monokernel off | On | Change |
|---|---|---|---|
| AgentX conc 1, P90 interactivity (1/ITL p90, tok/s/user) | 288.3 | 343.2 | +19% |
| AgentX conc 1, throughput per GPU (tok/s) | 6308 | 6861 | +9% |
| 8K/1K conc 1, median TPOT, MTP3 (ms) | 2.21 | 1.19 | −46% |
| 8K/1K conc 1, median TPOT, MTP5 (ms) | 1.70 | 1.13 | −34% |
| GSM8K 1319, MTP3 accuracy / accept length | 0.918 / 3.31 | 0.916 / 3.32 | parity |
| GSM8K 1319, MTP5 accuracy / accept length | 0.912 / 3.99 | 0.913 / 4.03 | parity |

The off column for AgentX is a same-node baseline. The CSV baseline it was
checked against (`pr1026`, MI355X 1026 image + Oct 26 PRs) is 290.4 / 6245.

## What made a one-sentence request enough

The request worked because four things were already in place. A new model has
fewer of them.

| Already in place | Why it mattered |
|---|---|
| **A working decode kernel** for the same model | The work was a feature on top of a tested kernel, not a new kernel. |
| **A precedent for the same feature**: GLM-5.3 MTP ([#15](https://github.com/sgl-project/sgl-amd-mono/pull/15)) and Qwen GDN + MTP (#29) | The agent's first step was to read both and copy their spec declarations, verify fallbacks and scratch-row indexing. |
| **Test harnesses in the repo**: `check_layer.py`, `time_layers.py`, `test_contract.py` | The agent could prove "one verify launch equals L decode launches, bitwise" instead of arguing it. |
| **A week of shared context** in the same chat (below) | The agent already had the server scripts, nodes, images, harnesses and limits (8 rows, LDS budget). It had also diagnosed that MTP was blocked only by the missing verify path. |

## Before the request: a week of context (Oct 21–27)

| Date | Human asked | What came of it |
|---|---|---|
| Oct 21 | Forwarded the manager's message: "Some monokernel of GLM is in sgl-amd-mono repo. Please give it a shot, check accuracy and perf after integration." Then corrected the model: GLM-5.3-Flash, not GLM-5.3. | Integrated the plugin. 8K/1K TPOT dropped 29–35% at conc 1–4, with no gain at conc 16, because the kernel only covers up to 8 rows. |
| Oct 22 | "My accuracy is about 0.97, yours is 0.91. How did you test?" | **Mistake caught by the human:** the agent had used `few_shot_gsm8k` (5-shot, no chat template). The user's `GLM.sh` uses `sgl-eval` with thinking on. Re-graded with the same harness, both arms scored about 97%. |
| Oct 22 | "Is it true the repo doesn't support GLM MTP? It would look dumb if I'm wrong." | Checked `main` and every PR before the message went out. |
| Oct 26–27 | "A week later, does the repo support GLM-5.3-Flash FP4 + MTP?" Then a series of questions on how a monokernel works: rows vs batch size, the LDS budget, why DSv4.1 reaches 48 rows. | Learned the limits: rows = requests × draft tokens (MTP5 = 6 rows per request), 8 rows at TP4, LDS holds each row's activations. |

## The build, turn by turn (Oct 28)

| Time | Human | Agent | Result |
|---|---|---|---|
| 09:55 | "你能去做GLM5.3-Flash FP4 + MTP的monokernel嗎? 先讓row<=8可以work 你可以用這個node crsuse2-m2m-068" (Can you build it? Get rows ≤ 8 working first. Use node 068.) | Read #15 and #29, then SGLang's verify semantics (intermediate SSM and conv-window scratch, commit after accept, k-pool tail ring, pad slots). Designed KDA verify (one task per request and head, L tokens in order) and DSA verify (in-chain ordering of cache writes). Pulled an image and wrote the code. | |
| ~10:30 | | Compile-only script for every verify shape. Unit tests. Server failed: the image's SGLang predated `register(model_type)`. Switched both arms to SGLang `main` through `PYTHONPATH`. | |
| ~10:40 | | New check: one verify launch against L decode launches. Bitwise equal at the same width, for KDA and DSA, MTP2, 3 and 5, with pad rows and pools closing mid-chain. Graph capture failed (step table keyed by tensor pointers). Fixed with fixed per-width buffers. | |
| 11:11 | | GSM8K full set and 8K/1K for MTP3 and MTP5, on and off. | **Working.** GSM8K parity; TPOT 1.5–1.9× faster at conc 1. |
| ~11:30 | "我要你去跟這個比較: HANDOFF.md … 只跑conc1" (Compare against the pr1026 AgentX baseline, conc 1 only.) | Rebuilt the baseline's environment (1026 image, PR stack) plus one SGLang commit the plugin needs. Started a same-node baseline on GPUs 4–7. Added long-context kernel checks up to 131k tokens: all passed. | |
| 12:08 | | Mono AgentX crashed after 4 minutes: GPU memory fault. Real agentic contexts reached 330k tokens; the kernel test defaults stop at 8192. | |
| 12:08–13:00 | | Bisected the length: 262,147 passes, 262,148 faults (more than 65,536 pools). Two side problems: env-gated experiments silently reused stale FlyDSL kernels, and 118 GB of GPU core dumps filled the home quota. Under rocgdb with precise memory, the fault was an iscore store whose index register was clobbered after a divergent loop exit (an LLVM codegen issue). First fix: every lane stores, masked lanes to one discard slot. | |
| 14:08 | | Second mono AgentX finished. Reported: "monokernel is 39% worse at conc 1", with an explanation about a single-CTA radix select at long context. | **Wrong conclusion**, presented as a result. |
| ~14:15 | Chose "profile" from the agent's options. | Added an in-kernel `clock()` stage timeline. iscore was the hot stage. The cause was its own first fix: 48 of 64 lanes on every CTA stored to the same discard address. Second fix: a bounded buffer resource, so the hardware drops out-of-range stores. DSA layer at 393k tokens: 1960 → 290 µs. | |
| 15:35 | | Third mono AgentX finished. | **+19% P90 interactivity**, no faults through 397k-token contexts. |
| ~15:40 | "I am confused. 你原本說monokernel在conc1比原本差3X%，現在又變好了? P90變成343比之前所有的MI355X和B200還要好?" | Explained the −39% as its own bug and withdrew it. Compared against every CSV line: 343 is the best conc 1 line, only 4% above B200, within run-to-run noise. | |
| | "Did the 343 run last the full hour? For conc 2 (rows 2 × 8 = 16), can we cap decode at bs 8 to stay on the monokernel?" | Confirmed the full hour. Corrected the row count (MTP5 conc 2 = 12 rows) and explained why capping the batch would not help. | User dropped conc 2. |
| ~16:00 | "做成draft PR … 收集PR template，做成skill" (Open a draft PR; collect the repo's PR template into a skill.) | Wrote the `sgl-amd-mono-pr` skill. Linted with the repo's pinned tool versions. Ran the kernel tests on 068, which collided with an AgentX job the user's other chat had started on the same node. The agent first called one failure "flaky on main", then withdrew that. | Draft PR #71 opened (~17:45). |
| Evening | "Delete the E2E-normalized row." "Use node 168 to rerun what's pending." "Would conc 4 + MTP3 be faster?" | Noticed while writing the PR that its GSM8K and 8K/1K numbers predated the final kernel. Reran everything at the PR head on 168. The rerun found one of its own test bugs (`check_embed` cloning a new `None` field). Read the conc 4 batch distribution: 84.5% of decode batches have 1 request. | PR updated with head-of-branch numbers. |
| Evening | "Check that 'until now the repo didn't support MTP at all' is true. If not, I'll look bad." | It was false: GLM-5.3, DeepSeek-V4/V4.1 and MiniMax-M3/M3.1 already have speculative decoding. Only GLM-5.3-Flash lacked it. | Wording fixed before it was sent. |
| Oct 29 08:37 | Picked node 097 and the CSV baseline. | The conc 4 runs had not started the night before: the agent cannot act after its reply ends, and the node's hold expired. Reran on 097: MTP3 and MTP5, conc 4. | See below. |

## Where it went wrong, and who caught it

| Problem | Cost | Caught by | Gate that would have caught it earlier |
|---|---|---|---|
| Wrong GSM8K harness (0.91 vs 0.97) | A false accuracy gap, almost sent to colleagues | Human, who knew their own number | Agree on the accuracy harness and its expected score before testing |
| Long-context fault above 262,144 tokens | One crashed one-hour run | AgentX itself | A kernel check at the workload's real maximum context (agentic contexts reached 397k) |
| Stale FlyDSL kernels in env-gated experiments | About 30 minutes of invalid bisection | Agent: a "compile" took 8 s instead of 30 s | A fresh cache directory per variant, created up front |
| 118 GB of GPU core dumps in the repo directory | Home quota full; writes failed | Agent | `ulimit -c 0` before any fault reproduction |
| A fix created a memory hotspot, reported as "monokernel −39%" | A wrong conclusion reported to the user, plus a one-hour run | Profiling, which the human chose | Kernel microbenchmark across the workload's context range, before and after the fix |
| GPU contention with the user's own job from another chat | 8 false test failures, a false "flaky on main" claim, and a possibly perturbed conc 64 run | Agent, from OOM messages | Check that the node is idle, and give each chat its own node |
| PR numbers came from a revision before the last kernel fix | Stale numbers in a draft PR | Agent, while writing the PR | Tag every result with its commit and rerun after every kernel change |
| The agent's own test bug in `check_embed` | One failing kernel test | The full kernel suite on 168 | Run the full kernel suite, not only the new test |
| "The repo didn't support MTP at all" | Would have been wrong in front of the manager | Human asked for verification | Verify any claim about the repo against `main` and every PR before sending it |
| Follow-up runs never started | A day of a node hold | Nobody until the next morning | Chain follow-up jobs into the driver script |

## What the human contributed

- **Scope:** rows ≤ 8, conc 1 first, TP4.
- **Resources and rules:** which node, which nodes not to touch, "don't edit the CSV", "ask before adding a line".
- **The comparison:** a specific baseline file and the metric (AgentX P90 interactivity).
- **Decisions at forks:** profile rather than guess, skip conc 2, use the CSV baseline for conc 4.
- **Skepticism:** "my accuracy is 0.97", "I am confused", "did it run the full hour?", "check before I send this".

The human never described how to implement anything. Every one of these
inputs changed what the agent did next.

## Conc 4 follow-up (Oct 29)

The question was whether the 8-row limit makes the monokernel useless at
conc 4. The agent first read the baseline's server log: 84.5% of decode
batches had only 1 request. It then ran both MTP settings with the
monokernel on node 097, against the CSV line (no same-node baseline).

| AgentX conc 4, 3600 s | CSV `pr1026` (MTP5, off) | MTP3, on | MTP5, on | B200 NVFP4 (CSV) |
|---|---|---|---|---|
| P90 interactivity (tok/s/user) | 247.2 | **327.4 (+32%)** | 255.8 (+3%) | 267.8 |
| Throughput per GPU (tok/s) | 8776 | 9125 (+4%) | 9047 (+3%) | 9252 |
| ITL p50 / p90 (ms) | 2.6 / 4.0 | 2.09 / 3.05 | 2.28 / 3.91 | |
| Decode batches the kernel can serve | | 97.9% (1–2 requests) | 85.0% (1 request) | |
| Mean accept length (server log) | | 3.19 | 4.10 | |
| Errors / profiled requests | | 0.3% / 697 | 0.3% / 688 | |

MTP3 wins because 1 and 2 requests both fit in 8 rows. A longer tail of
MTP5 steps falls back to SGLang's path. The MTP3 gain mixes two changes,
the monokernel and MTP5 → MTP3, since no MTP3 baseline without the
monokernel was run.
