---
name: sgl-amd-mono-pr
description: Write and open pull requests for sgl-project/sgl-amd-mono (AMD monokernels for SGLang) in the repo's Motivation / Modifications / Accuracy Tests / Speed Tests and Profiling / Checklist template, each section's prose capped at 100 words with the detail carried by tables. Use when opening, drafting, or rewriting an sgl-amd-mono PR body or title, especially monokernel work on MI355X (glm53_flash, glm53, deepseek_v41, minimax, qwen).
---
# sgl-amd-mono PR Description

The repo ships a template (`.github/pull_request_template.md`). Use its sections, in this order, and nothing else:

1. **Motivation**
2. **Modifications**
3. **Accuracy Tests**
4. **Speed Tests and Profiling**
5. **Checklist**

Drop the template's HTML comments and its **Review and Merge Process** section; merged PRs keep neither. A short **Not covered** section before Checklist is allowed when the limits do not fit in Modifications (#56 does this).

The comments are still the requirements; reviewers quote them ("as the template asks"):

| Section | The template's comment |
| --- | --- |
| Accuracy Tests | "If this pull request changes kernel or forward code, give accuracy results with the monokernels on and off, on the same server." |
| Speed Tests and Profiling | "If this pull request affects speed, give per-user decode results with the monokernels on and off, including past the largest width." |

"Past the largest width" means a control point: a load where every step is wider than the monokernel runs, so both sides run SGLang's path and must match. It bounds the run-to-run noise of the other points (#95's "off (control)" rows, #71's c16 MTP3).

## Write for a reviewer who has not read the code

This rule outranks the rest. The code owners (`.github/CODEOWNERS`) review every model's PRs, and they did not write your stage. Every symbol they must look up is a reason to put the tab aside.

**Open Motivation with what the user of the server sees,** in plain words, before any symbol: the step that runs SGLang's path instead of the monokernel, the fault, the slower TPOT. Then the cause, then the fix. #61 is the model: it opens with "a false fail-stop" and the 3-in-5 cold boots that died, and only then names `fcntl.flock` and the deadline.

| Symptom | Fix |
| --- | --- |
| The first sentence names a kernel stage or a private function | Move it down. Open with what fails or what is slow. |
| One sentence carries three or more symbols | Split it, or say what two of them do |
| Following it needs another file open | Say what that file does, in words |
| The design arrives before the problem | Reverse them: problem, cause, fix |

Simple means fewer unexplained symbols, never vaguer claims. The measured-numbers rule below is not relaxed by this one.

## Simple English

Short sentences, common words. If a sentence needs a second read, split it or use a plainer word. The repo's own bodies read this way: "A launch whose wait times out stops", "The kernel does not change."

## The 100-word rule

Each section's **prose** is capped at 100 words. Tables, code blocks and the Checklist do not count.

Merged PRs here often run 10,000+ characters. Do not copy that length: copy what they *cover* (bitwise coverage, every TP, past the largest width) and put it in tables. When a section runs long, move detail into a table; do not shave words evenly off every sentence.

## What each section carries

### Motivation

What the user sees, why the current code costs or breaks it, and what this PR changes. End with the dependencies:

- **SGLang:** the exact commit, plus any SGLang PR the plugin needs (`sgl-project/sglang#NNNNN`). The checklist asks for this, and a reviewer cannot rerun anything without it.
- **Stacked PRs:** "Depends on #N. This branch is stacked on it, so review only the last commit, `abc1234`." (#51)
- **Merge order** against other open PRs on the same model, when they touch the same files (#62).

### Modifications

One bullet or one short subsection per change, by file, saying *what changes in behaviour*, not restating the diff. Things reviewers here always look for:

| Item | Say |
| --- | --- |
| Which steps the monokernel now runs, and which still run SGLang's path | Name the forward modes, widths, TP sizes, and the fallback condition |
| New SGLang internals read | Each one is declared in the module's `SGLANG_READS`; `test/unit/test_contract.py` checks them on CPU |
| New `Requirements` (`server_args`, `env`) or `features` on the `MonoSpec` | One line each, with the reason SGLang's path is needed otherwise |
| Kernel change | Whether any value changes. "Bitwise equal to `main`" is a claim to prove in Accuracy Tests. |
| Scratch / layout change | Bytes, and what it costs in KV-pool tokens if it moves free memory (#62) |
| Fault-injection site numbers | Whether `SGLANG_AMD_MONO_FAULT_INJECT`'s site numbers move |
| Test changes | Which test, and its negative control: "with X broken, the check fails" (#62, #60) |
| New model | Its `.github/CODEOWNERS` line and README row |

### Accuracy Tests

The template asks for results **with the monokernels on and off, on the same server stack**. Name the setup first: GPU, image tag, SGLang commit (and PR stack), checkpoint, TP. Then the evidence, strongest first:

| Evidence | Tool |
| --- | --- |
| Launch equals a reference bitwise | `test/kernels/<model>/check_layer.py`, via `test/kernels/test_<model>_kernels.py` (8 GPUs, TP2/4/8) |
| Layer check against SGLang's forward | `SGLANG_AMD_MONO_CHECK=1`: checks, violations, worst cosine |
| End to end | `test/e2e/test_<model>_server.py` cases passed |
| Task accuracy | GSM8K (or the model's eval) on vs off, same flags, with spec accept length if MTP |

State a small accuracy move instead of hiding it: "0.912 on vs 0.907 off, one scoring pass each".

An accuracy run or e2e test counts only for the steps that reach the kernel. Rows a step = requests × draft tokens (MTP3: 4, MTP5: 6), so with a largest width of 8, MTP3 runs 2 requests at a time and MTP5 runs 1. At 12 requests, #71's `TestMonokernelMtp` put about 14% of the verify work on the kernel, and its accept-length bound could not fail (the code owner's request 2). Say what share of the steps ran the kernel, and make each bound one that fails when the kernel path breaks.

Accept length is evidence only with real acceptance. AgentX's golden accept length (`SGLANG_SIMULATE_ACC_LEN`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`) fixes it regardless of the verify, so it says nothing about correctness.

For DSA models, GSM8K prompts are shorter than the index top-k, so every key is selected and the sparse selection is not tested. A change to DSA, or a new width whose launch is close to but not bitwise equal to its reference, also needs a long-context check with real acceptance at the new widths, e.g. RULER NIAH at 300K and 640K (#98).

### Speed Tests and Profiling

The template asks for **per-user decode, on and off, including past the largest width** (where both sides run SGLang's path and must match). Name the setup (image, SGLang commit, flags, workload, rounds), then tables:

| Table | Columns |
| --- | --- |
| End to end | Concurrency, off, on, change. Median TPOT (ITL in parentheses), or AgentX P90 interactivity and tok/s/GPU |
| Kernel only | `test/kernels/<model>/time_layers.py --kind <kind> --rows ...`, µs a launch, `main` → this PR |

Say the run-to-run spread when it is comparable to the gain ("c1 screens at 900 s move by several percent between identical runs", #56).

**AgentX.** The code owner (`@jhinpan`) reruns AgentX tables and asks for everything below when it is missing (#71, both reviews):

| Item | Say |
| --- | --- |
| Same-configuration A/B | Same image, flags and MTP depth. Run off and on one at a time, or at the same time on GPUs 0–3 and 4–7 with a second round that swaps the sets; one round on one node shares HBM, fabric and power. Never off from another node or day. A best-against-best row (off at MTP5, on at MTP3) may be added as a separate row, never in place of the same-MTP pair (#98). |
| Window | 3,600 s, as InferenceX runs it. Name it: 900-s windows move by as much as the gain. |
| Columns | P90 interactivity (1 / ITL p90), P90 E2E-normalized interactivity, P90 TTFT, total tok/s/GPU, and requests completed by each side (a side that finishes more turns in the window shifts the P90s) |
| Control point | One concurrency past the largest width (see the template's comment above) |
| Server command | In a `bash` block: the full `launch_server` line and every exported env var, `SGLANG_SIMULATE_ACC_*` included. Say that off drops `--monokernel-backend amd`. |
| Client | aiperf version and source commits (InferenceX and its `utils/aiperf` submodule), and the full `aiperf profile` line in a `bash` block |
| Aggregate JSONs | The aggregate JSON of each point (`*_local-mi355x.json`), in a gist linked from the body |

When the points differ only in a few values (concurrency, MTP depth, golden accept length), give one point's commands and a table of the differences.

When the code owner's rerun disagrees because it used shorter windows, keep the 3,600-s target rather than rerunning at their window. Give the commands so the two runs can be compared.

A metric the change should not move (TTFT for a change to verify launches only) that moves shows the noise of the setup; give a replicate pair so that its size is known (#100).

### Checklist

Copy the template's five items. Tick what holds; for an unticked item say why in parentheses, e.g. "(No new model.)" (#60).

```markdown
- [x] `pre-commit run --all-files` passes.
- [x] Tests under `test/` cover the change.
- [x] The SGLang commit or PR this was tested against is named above.
- [x] Accuracy and speed results are attached when kernel or forward code changes.
- [ ] A new model adds one line to `.github/CODEOWNERS` that names its owner. (No new model.)
```

## Rules

- **Never hard-wrap the body.** One line per paragraph. GitHub soft-wraps; the commit message still wraps at 72.
- **Measured numbers, never adjectives.** "2.534 → 2.433 ms (−4.0%)", not "faster".
- **Every number in the prose is in a table,** with the command that measured it. #98's Motivation gave 183 µs for the per-row MoE at 16 rows, and no table showed it.
- **State a slower cell,** with its size and whether a threshold would avoid it (#100: 4 rows at pos 8191, +2.4%).
- **One variable.** On and off share an image, SGLang commit, checkpoint, TP and flags except `--monokernel-backend amd`; say so.
- **Never write "arm".** Say "off" / "on", or "`main`" / "this PR".
- **Name the statistic** in every header: median TPOT, P90 interactivity, mean TTFT.
- **Name the thing:** the stage, the `SGLANG_READS` key, the env var. A reviewer greps for it.
- **Arrows read old → new** (`main` → this PR, off → on), as every merged PR here does.
- **No new env flag per optimization.** A verified faster path replaces the old one; the shape or config check picks the fallback.
- **One change per PR.** A fix that does not need the feature (#71's decode fault past 262,144 tokens) goes in its own `[Fix]` PR, merged first (#97).
- **Test on upstream SGLang.** Name the commit, and each open SGLang PR the change needs; a stack of unrelated AMD PRs is not a setup the code owner can rerun.
- **Refuse at startup.** A server setting the monokernel cannot run is a config disable reason (glm53_flash: `_config_disable_reasons` in `spec.py`), with a unit test, not an error at the first step.
- **Read declared SGLang attributes directly.** When SGLang declares one with a default, `getattr(x, "name", None)` hides a rename that `SGLANG_READS` would catch.
- **Check what the change relies on in SGLang.** If the kernel depends on an SGLang behaviour (#100: a verify request's page-table row repeated for each of its rows), add a check that fails when SGLang stops doing it.
- **Tests import the server's constants** (`GROUPED_FROM`), never restate them (`rows > 8`), so that the tests build what the server builds (#98).
- **Code comments say what the code cannot show** (#43–#46 trimmed the rest). Never where a change came from or why the diff is right.

## Title

Checked by the required `Check PR title` workflow (`.github/workflows/pr-title.yml`):

```
[Type] scope: Title
[Type] [Hardware] scope: Title
```

- **Type:** `Feat`, `Fix`, `Refactor`, `Perf`, `Doc`, `Test`, `CI`, `Chore`.
- **Hardware:** optional, one uppercase tag without spaces. Here it is `[MI355X]` for gfx950 kernel work; `[AMD]` adds nothing in an AMD-only repo.
- **Scope:** the module or model package name as it is on disk: `glm53_flash`, `glm53`, `deepseek_v41`, `dsv4`, `minimax_m31`, `qwen38_flash_next`, `common`, `mono`. Letters, digits and `._/-`, no space before the colon.
- **Title:** sentence case, imperative, plain words, no trailing period. One space between fields and after the colon.
- **At most 12 words**, not counting the `[...]` tags; `glm53_flash:` is one word. Drop file and kernel names; the body says them.
- The title becomes the squash commit on `main`, so it must read as a commit subject.

| Fails or too long | Passes |
| --- | --- |
| `[Feat][MI355X] glm53_flash: ...` (no space between tags) | `[Feat] [MI355X] glm53_flash: Serve MTP verify steps on the monokernel` (8) |
| `[Perf] [mi355x] glm53: ...` (lowercase hardware) | `[Perf] [MI355X] glm53: Score the paged indexer keys per wave` (8) |
| `[Fix] GLM-5.3 Flash: ...` (space in scope) | `[Fix] glm53_flash: Check the index cache by its dequantized keys` (9) |

## Opening the PR

There is no push access to `sgl-project/sgl-amd-mono`; PRs come from forks (`MarloweAI:perf/...`, `jhinpan:glm53-fp8-kv`). Use the gh wrapper without the tokens in the environment:

```bash
GH="env -u GH_TOKEN -u GITHUB_TOKEN /home/jacchang/bin/gh"
$GH repo fork sgl-project/sgl-amd-mono --remote --remote-name fork --clone=false
cd ~/PR/sgl-amd-mono
git fetch origin main && git switch -c jacob/<topic> origin/main   # or rebase the work branch
```

Lint before committing. `pre-commit` is not installed on the login node and `python3 -m venv` fails there, but the hooks' exact versions (isort 7.0.0, ruff 0.15.1, codespell 2.4.1) are in `/shared_nfs/jacchang/fmtpkgs`:

```bash
export PYTHONPATH=/shared_nfs/jacchang/fmtpkgs; B=/shared_nfs/jacchang/fmtpkgs/bin
F=$(git diff --name-only origin/main)
python3 -m isort --check-only $F
$B/ruff check --select=F401,F821,F823,UP037 $F && $B/ruff format $F
python3 -m codespell_lib --config .codespellrc $F
```

The `no-commit-to-branch` hook refuses commits on `main`; work on a branch. Commit with the identity passed per command, push to the fork, open as draft:

```bash
git -c user.name="Jacob0226" -c user.email="jacchang@amd.com" commit -F /tmp/msg.txt
git push fork jacob/<topic>
$GH pr create --repo sgl-project/sgl-amd-mono --draft --base main \
    --head Jacob0226:jacob/<topic> --title "[Feat] [MI355X] <scope>: ..." --body-file /tmp/body.md
```

CI is GitHub-hosted only: `lint` (pre-commit on all files) and `Check PR title`. No GPU tests run in CI, so the Accuracy and Speed sections are the only evidence a reviewer gets. Then ask the model's code owner from `.github/CODEOWNERS` to review (glm53 and glm53_flash: `@jhinpan`). Merge is "Squash and merge".

## Tests

| Directory | Runs on | Command |
| --- | --- | --- |
| `test/unit` | CPU, needs SGLang importable | `python -m pytest -q test/unit` |
| `test/kernels` | 8 × gfx950 (skips otherwise) | `python -m pytest -q test/kernels/test_<model>_kernels.py` |
| `test/e2e` | a TP server on the node | `python -m pytest -q test/e2e/test_<model>_server.py` |

Run GPU tests with `ulimit -c 0` and `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`; a faulting test otherwise writes a GPU core dump of tens of GB into the working directory. Give each FlyDSL experiment a fresh, existing `FLYDSL_RUNTIME_CACHE_DIR` (`mkdir -p` it first): the cache key ignores env vars read at trace time, and a missing directory silently falls back to `~/.flydsl/cache`.
