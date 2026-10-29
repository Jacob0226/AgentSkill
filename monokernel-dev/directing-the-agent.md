# Directing an agent to build or extend a monokernel

A guide for the human. It comes from one project: adding MTP to the
GLM-5.3-Flash monokernel ([case study](case-study-glm53-flash-mtp.md)). Every
rule below traces to something that cost time there. The agent-side version
of the same workflow is [SKILL.md](SKILL.md); point your agent at it.

**Tested environment:** Cursor agent mode with Claude Opus 5.5 at High effort.
No other model was tried. A weaker model may struggle most where this project
was hardest; see [Model choice](#8-model-choice).

## 1. Know which kind of job you're asking for

| Job | Example | Relative effort | Biggest risk |
|---|---|---|---|
| Add a step type to an existing kernel | MTP verify on GLM-5.3-Flash (this project) | 1× | Ordering and state semantics the decode path never needed |
| Raise the row limit | GLM-5.3-Flash beyond 8 rows | Several × | The LDS budget: each CTA holds every row's activations. 16 rows don't fit without redesign. |
| A new model | A model with no package in sgl-amd-mono | Much larger; not done here | Everything: weights, layouts, every stage, every contract |

This project was the first kind. For a new model, start from the package of
the closest architecture (`glm53_flash` for KDA + DSA hybrids, `deepseek_v41`
for wide verify), and expect each section below to take longer.

## 2. Prepare before the first prompt

The agent did well on a one-sentence request because these already existed.
Hand over whatever is missing:

- [ ] **Repos and branches:** sgl-amd-mono, SGLang, and the SGLang commit the plugin needs.
- [ ] **Precedent PRs** that added the same feature to another model (here, #15 and #29). This is the biggest lever.
- [ ] **Environment:** image tag, model path, how containers are set up.
- [ ] **Nodes the agent owns exclusively.** Say which nodes other chats or people are using.
- [ ] **The baseline:** exact numbers, the file they come from, and how they were measured (here, a handoff doc and a CSV line).
- [ ] **The accuracy harness and its expected score** (here, `sgl-eval` with thinking: about 0.97; `few_shot_gsm8k` gives about 0.91 on the same server).
- [ ] **The workload's shape:** batch-size distribution and maximum context. AgentX contexts reached 397k tokens; the kernel tests stopped at 8192.
- [ ] **House rules:** where large files go, what not to touch, when to ask first.

## 3. The kickoff prompt

```text
Goal: make the <model> monokernel in sgl-amd-mono serve <feature> for <scope,
e.g. verify steps of up to 8 rows at TP4>. Use the monokernel-dev skill.

Starting point: <existing package / precedent PRs #A, #B>.
Environment: image <tag>, SGLang <commit or branch>, model <path>.
Nodes: <node>, all 8 GPUs are yours. Do not touch <nodes>.
Baseline: <file and line>, metric <P90 interactivity>, measured with <script>.
Accuracy: <harness>, expected <score>.
Workload: contexts up to <N> tokens; decode batches mostly <k> requests.

Pass every gate in the skill on the exact revision before any agentic
benchmark. Tag every result with commit, image and node. Ask before
<long runs / pushing / editing shared files>.
```

## 4. Insist on the gate ladder

The common instinct is right: unit tests and microbenchmarks must pass
before the agentic benchmark. The fuller rule is that each gate is cheaper than the next
and catches a different class of bug. Never skip ahead, and after **any**
kernel change go back to gate 3.

| Gate | What | Time | What it caught, or would have caught, here |
|---|---|---|---|
| 1 | Lint, unit and contract tests (CPU) | Minutes | Changed constructor arguments; fakes in tests |
| 2 | Compile every shape the spec will serve | Minutes | FlyDSL trace errors |
| 3 | **Bitwise equivalence** against the existing path (verify = L decode launches) | Minutes | The core correctness proof |
| 4 | **Correctness at the workload's real limits**: max context, max rows, pad rows | Minutes | **The fault above 262,144 tokens**, which instead crashed an AgentX run |
| 5 | **Kernel microbenchmark** across the workload's range, before and after the change | Minutes | **The −39% "regression"**: at 393k tokens the DSA layer took 1960 µs with the first fix and 290 µs with the second |
| 6 | End-to-end accuracy, on vs off, same flags, low parallelism so the kernel is actually used | 10–20 min | The wrong-harness mix-up, if the expected score is known |
| 7 | Fixed-length serving benchmark (8K/1K), on vs off | 10 min | Whether the gain is real before spending an hour |
| 8 | Ten-minute smoke run of the real workload | 10 min | Crashes and fallbacks, cheaply |
| 9 | Full agentic benchmark (AgentX, 1 hour) with a same-node baseline | 1+ hour | The number you report |

Two refinements:

- **Gate 5 must cover the workload's range, not one point.** The hotspot was
  invisible at 8k context and dominant at 393k.
- **Gates 4 and 5 rerun after every fix.** The −39% came from a fix to a bug
  that gate 4 found. The fix itself was never benchmarked.

## 5. Where you should step in

| When the agent… | Say | Why |
|---|---|---|
| Explains a regression with a story | "Profile it before concluding." | The radix-select story was plausible and wrong. Profiling found the real cause in minutes. |
| Reports a result that contradicts an earlier one (8K/1K 1.5× faster, AgentX 39% slower) | "Reconcile these two first." | One of them is a bug. Here it was the AgentX run. |
| Calls a failure "pre-existing" or "flaky on main" | "Show me main failing on an idle node." | The "flake" was contention with another job. |
| Is about to start a run longer than 30 minutes | "Which gates passed on this exact commit?" | Prevents hour-long runs on unverified code. |
| Writes anything you will send to others | "Verify every claim against main and all PRs; cite sources." | "The repo didn't support MTP" was false. |
| Finishes a fix | "Which earlier results does this invalidate?" | The PR's first numbers predated the final kernel. |
| Gives you a number | "Same-node baseline? Full duration? Error rate?" | "Did it run the full hour?" is the right habit. |

## 6. Run hygiene the agent should follow (and you should check)

- **One chat per node.** A second chat started AgentX on the node mid-test;
  both sets of results were suspect.
- **Fresh compile caches for experiments.** FlyDSL's cache key ignores
  environment variables read at trace time, and it silently falls back if the
  cache directory doesn't exist.
- **`ulimit -c 0` before reproducing faults.** GPU core dumps reached 118 GB.
- **Large files go to shared storage,** not a quota-limited home.
- **Tag every result** with commit, image, SGLang commit and node, in a results
  ledger file. Long chats get summarized, and details are lost from the
  summary but not from files.
- **One variable per comparison.** Same image, same SGLang, same node, same
  flags except the monokernel switch.

## 7. Managing a long-running agent

- **The agent stops when its reply ends.** It cannot start the next job after
  a run finishes. Ask it to chain follow-ups into the driver script, or come
  back and ask. Here a node's 16-hour hold expired with the conc 4 runs never
  started.
- **Ask for decisions as choices.** The agent offers options at forks (profile
  vs rerun, which baseline). Picking one takes seconds and keeps you in
  control of GPU time.
- **Ask for verified vs assumed.** A good habit is "what did you verify, and
  what are you assuming?" after any surprising result.
- **Keep progress updates sparse.** Hourly is enough for hour-long runs.

## 8. Model choice

This project used Claude Opus 5.5 at High effort in Cursor's agent mode. The
hard parts were reading GPU ISA under rocgdb, reasoning about memory ordering
across CTAs, and instrumenting a persistent kernel. They need a strong
reasoning model with a long context. No other model was tried, so this guide
can't say how others compare. Record the model and effort level with your
results so the next person can.
