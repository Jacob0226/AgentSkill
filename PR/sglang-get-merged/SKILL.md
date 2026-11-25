---
name: sglang-get-merged
description: >-
  Get an approved sgl-project/sglang PR merged by a maintainer with merge rights (BBuf), based on what
  BBuf actually checked before merging #44415: NVIDIA CI green, a GLM-5.3-Flash accuracy result for the
  nightly test /rerun-test cannot run, and a root cause for every red check. Use when the user asks who
  can merge an SGLang PR, wants to ping BBuf, asks whether a PR is ready to hand over for merge, or wants
  a merge-request comment drafted for PRs such as #44416 / #44417.
---
# Getting an SGLang PR merged

Teammates (Raiden-Makoto, EricKing626) can approve but cannot merge. A Merge Oncall has to merge, and
the PR has to be ready **before** the ping: the reviewer's agent answers in minutes, so every missing
piece costs a round trip of a day or more.

## Who to ask

| Person | Use | Why |
| --- | --- | --- |
| **BBuf** | Default merger | Merge Oncall. Answers through "BBuf Review Agent". Bypasses CODEOWNERS: merged #44415 with `reviewDecision` still `REVIEW_REQUIRED`, and #50196 with only his own approval |
| kevin-mii | Do not ask | No reply on #44417 for 6 days (11/19 → 11/25) |
| HaiShaw | Do not ask | The user's manager; no time |

**Never ask BBuf to rerun CI.** `/tag-and-rerun-ci`, `/rerun-failed-ci` and `/rerun-test` go to another
maintainer the user knows. BBuf's time is spent only on review and merge.

## Ready checklist (all must hold before pinging BBuf)

```
- [ ] Head commit has NVIDIA CI (PR Test Base) all green
- [ ] A teammate approved (any commit is fine)
- [ ] The PR's own new tests pass on every suite they are registered in (AMD included)
- [ ] GLM-5.3-Flash accuracy test run locally at the current head (below)
- [ ] Every red check has a one-line root cause with evidence
- [ ] The diff shows only this PR (merge main after a stacked PR lands)
```

| Item | Detail |
| --- | --- |
| NVIDIA CI | Every new head commit needs a maintainer `/tag-and-rerun-ci`, or `pr-test-finish` fails with "head commit has no run-ci approval" and all tests show skipped. So push everything at once, then get one approval. A low-permission actor also hits `pr-gate`'s 120-minute cooldown when pushing several PRs in a row |
| Own tests | A new test registered on `base-b-test-1-gpu-small-amd` also runs on MI300 (gfx942). A gfx950-only path must skip there, not assert. BBuf's agent reads every failure, and one caused by the PR blocks the merge |
| Accuracy | BBuf runs `/rerun-test test/registered/amd/accuracy/mi35x/test_glm53_flash_eval_mi35x.py` on any GLM-5.3-Flash change. The bot refuses it (nightly suite). He accepted a local run at the PR head with sha, image, score and threshold. For DSA indexer, k-pool, KV cache, MoE or quant changes, also have RULER 32K / 128K ready (`sglang-pr` skill) |
| Red checks | AMD failures count. For each one give the cause and something he can verify: the fix PR on main, the same failure on unrelated PRs, or a cancelled run. #44415: "`test_nixl_transfer_engine_e2e.py` crash fixed by #51525"; he checked #51525 and merged with the job still red |
| Stacked PRs | SGLang squash-merges, so after the base PR lands the stacked PR's diff still shows the base PR's lines. `git merge upstream/main` removes them; it is a review-size fix, not a conflict fix |

## The comment

One comment, `@BBuf`, everything in it. His agent does not reread old comments: on #44415 he ran the
accuracy `/rerun-test` twice more on 11/21 although the result had been posted on 11/19, and accepted
it within a minute once it was in a comment that tagged him.

```markdown
@BBuf Could you help review and merge this PR? <one sentence: what it does, gated to what>. @Raiden-Makoto approved it but doesn't have merge permission.

`/rerun-test` can't run `test_glm53_flash_eval_mi35x.py` (nightly suite), so I ran it locally at `<sha>` on 8x MI355X (`<image>`): GSM8K <score> (<n>/1319) against the 0.92 threshold; main records 0.975.

NVIDIA CI is green. The only red check is `<job>`: <cause>, <evidence link or PR>.
```

- Keep it this short; #37941 got merged 4 minutes after a three-line ask.
- Never hard-wrap; one line per paragraph.
- No "you can either review or just merge": say who approved, and let him decide.
- If the PR is ready and nothing happens for 1–2 days, one short follow-up with any new evidence.

## Running the accuracy test locally

Follow [accuracy-run.md](accuracy-run.md). About 25 minutes on one 8-GPU MI355X node.

## Reference: #44415 (request 11/19, merged 11/24)

| Date | Event |
| --- | --- |
| 11/19 | Ask posted. BBuf Review Agent approves ("LGTM. Let's wait for CI."), tries the accuracy `/rerun-test` (refused), runs `/rerun-failed-ci` |
| 11/21 | Tries the same `/rerun-test` twice more |
| 11/23 | User tags him with the local GSM8K 0.9727 → "That covers the accuracy check for me" (1 min). User explains the disaggregation failure with #51525 → he verifies and accepts (2 min) |
| 11/24 | `/rerun-failed-ci`; disaggregation still red; he merges |
