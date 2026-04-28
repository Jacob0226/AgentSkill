---
name: sglang-ci-repro
description: >-
  Reproduce SGLang AMD CI failures locally on MI35x / MI325 boxes using
  ~/SGLang-benchmarks/tools/repro_ci.sh. Use when a sglang PR's AMD CI fails
  (stage-a/b/c-*, nightly-8-gpu-mi35x-*, etc.) and the user wants to debug
  locally before pushing again, or asks "how do I rerun this CI failure
  locally". Covers the CI architecture (workflow → stage → suite → test
  file mapping), how to map a failing GitHub Actions job to repro_ci.sh
  flags (--single-file, --suite, --partition-id, --nightly, --hostname for
  GPU arch detection), edge cases like multimodal-gen tests that use a
  different driver, and how to test alternative aiter versions.
---

# SGLang AMD CI Repro Workflow

## Quick start (90% case)

CI shows `python3 .../test_xxx.py` failed → just rerun that one file:

```bash
bash ~/SGLang-benchmarks/tools/repro_ci.sh \
    --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 \
    --single-file registered/amd/test_qwen3_instruct_mxfp4.py
```

By default the script uses the image's self-contained `/sgl-workspace/sglang`
(no host mount). Add `--sglang-dir "$HOME/PR/sglang"` only if you've edited
local sglang code that needs to be picked up.

`bash ~/SGLang-benchmarks/tools/repro_ci.sh --help` has the full reference
including a "QUICKSTART" / "CONCEPT REFERENCE" / "EXAMPLES" block at the
top of the script. **Always read it first** when in doubt about a flag.

## CI architecture (what maps to what)

| Layer | Lives in | Purpose |
|-------|----------|---------|
| Workflow | `sglang/.github/workflows/{pr-test-amd,nightly-test-amd*}.yml` | Decides which jobs run, on which runner, on what trigger |
| Stage / job | YAML job (e.g. `stage-b-test-small-1-gpu-amd`) | One row in the GitHub Checks UI; runs container + install + test |
| Suite | `sglang/test/run_suite.py` `PER_COMMIT_SUITES` / `NIGHTLY_SUITES` | Named group of test files |
| Test file | `sglang/test/registered/**/*.py` | Real test, registered with `register_amd_ci(suite="...")` |

CI's per-stage shell sequence (mirrored by `repro_ci.sh`):

```bash
bash scripts/ci/amd/amd_ci_start_container.sh        # spawn container
bash scripts/ci/amd/amd_ci_install_dependency.sh     # rebuild aiter + sgl-kernel
bash scripts/ci/amd/amd_ci_exec.sh -w /sglang-checkout/test \
    python3 run_suite.py --hw amd --suite <SUITE> ...   # run the suite
```

`repro_ci.sh` skips step 2 by default (image already self-contained); add
`--run-install` if you've changed sglang core / sgl-kernel / aiter source.

## Mapping CI surface → flags

When you click into a failing job in the GitHub Actions UI, note these:

1. **Workflow file** (top of page, e.g. `PR Test (AMD)` → `pr-test-amd.yml`)
2. **Stage / job name** (e.g. `stage-b-test-small-1-gpu-amd`)
3. **Matrix entries** (e.g. `(linux-mi325-gpu-1, 5)` — runner + part_id)
4. **Failed test file** in the log (search for `Begin (` to find lines like
   `Begin (3/12):\npython3 /sglang-checkout/test/registered/.../test.py`)

| CI surface clue | Flags to pass |
|-----------------|---------------|
| One test file failed (you can see it in the log) | `--single-file PATH` (90% case) |
| Whole nightly job failed (1 runner, no partition) | `--suite NAME --nightly` |
| Whole per-commit stage failed (no partition matrix) | `--suite NAME` |
| One shard of a partitioned per-commit stage<br>e.g. `(linux-mi325-gpu-1, 5)` | `--suite NAME --partition-id N --partition-size M`<br>(M = `matrix.part` length, see workflow YAML) |
| MI35x-specific runner (e.g. `linux-mi35x-gpu-8`) | Add `--hostname linux-mi35x-gpu-N` so `amd_ci_exec.sh` sets `GPU_ARCHS=gfx950` |
| `multimodal-gen-test-*-gpu-amd` job | **Not supported by `--suite`** — see "Multimodal-gen tests" below |

## Picking the docker image

The image must match the CI runner's GPU arch / ROCm version:

| Runner label | Image pattern | Example |
|--------------|---------------|---------|
| `linux-mi35x-gpu-N` | `rocm/sgl-dev:vX.Y.Z-rocm720-mi35x-YYYYMMDD` | `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211` |
| `linux-mi325-gpu-N` | `rocm/sgl-dev:vX.Y.Z-rocm700-mi30x-YYYYMMDD` | `rocm/sgl-dev:v0.5.8.post1-rocm700-mi30x-20260211` |

Find the exact tag CI used by reading the YAML's "Start CI container" step
(it calls `amd_ci_start_container.sh` which has hostname-based image
selection logic). For ROCm 7.2.x runs on `nightly-test-amd-rocm720.yml`,
add `--rocm-version rocm720` if invoking `amd_ci_start_container.sh`
directly; `repro_ci.sh` takes the image directly via `--docker`.

## Common workflow modes

```bash
# 1. Smallest repro: single failing file
bash repro_ci.sh --docker IMAGE --single-file registered/amd/test_xxx.py

# 2. Same file but with YOUR host edits
bash repro_ci.sh --docker IMAGE \
    --sglang-dir "$HOME/PR/sglang" \
    --single-file registered/amd/test_xxx.py

# 3. Full per-commit stage partition (CI showed "(...gpu-1, 5)" with matrix
#    part: [0..13])
bash repro_ci.sh --docker IMAGE \
    --suite stage-b-test-small-1-gpu-amd \
    --partition-id 5 --partition-size 14

# 4. Peek at which files partition 5 contains, before committing 30 min
bash repro_ci.sh --docker IMAGE \
    --suite stage-b-test-small-1-gpu-amd \
    --partition-id 5 --partition-size 14 --list-only

# 5. Full nightly suite (always single runner, no partition)
bash repro_ci.sh --docker IMAGE \
    --suite nightly-8-gpu-mi35x-qwen3-235b-mxfp4 --nightly

# 6. Drop into bash with CI env preset (manual debug)
bash repro_ci.sh --docker IMAGE --shell

# 7. After editing sglang core / sgl-kernel / aiter source, force rebuild
bash repro_ci.sh --docker IMAGE \
    --sglang-dir "$HOME/PR/sglang" --run-install \
    --single-file registered/amd/test_xxx.py
```

## Multimodal-gen tests (special case)

`multimodal-gen-test-1-gpu-amd` and `multimodal-gen-test-2-gpu-amd` use a
**different** test driver (`python/sglang/multimodal_gen/test/run_suite.py`,
not `test/run_suite.py`) with different flags (`--total-partitions` instead
of `--auto-partition-size`) and 8 extra diffusion env vars. `repro_ci.sh`
does NOT wrap that driver. Use `--shell` to set the container up, then
paste the CI's `docker exec` command verbatim. The exact recipe is in the
"MULTIMODAL-GEN NOTE" section at the top of `repro_ci.sh` — read its `--help`.

## Testing alternative aiter versions

To swap the aiter version inside an existing sgl-dev image (e.g. test
`v0.1.13.dev0` or `v0.1.12.post2`):

```bash
docker exec -it ci_sglang_repro bash
cd /sgl-workspace/aiter
git fetch --tags origin
git checkout <TAG>                             # e.g. v0.1.12.post2
git submodule update --init --recursive
pip uninstall -y amd-aiter
GPU_ARCHS=gfx950 PREBUILD_KERNELS=1 python3 setup.py develop
```

Build is 10-30 min depending on csrc/ delta. Cross-version `git checkout`
between v0.1.11 and v0.1.12 fails because of an upstream history rewrite —
the two tags share zero git ancestry; you can only checkout to/from the
post-rewrite chain.

When testing accuracy regressions across aiter versions, see
[bisection.md](bisection.md) for the file-overlay A/B helper that avoids
having to rebuild aiter from source.

## Common gotchas

- **Hostname matters for GPU arch detection.** The CI scripts parse
  `linux-mi35x-gpu-N` / `linux-mi325-gpu-N` to set `GPU_ARCHS=gfx950` /
  `gfx942`. `repro_ci.sh --hostname linux-mi35x-gpu-8` (default) works for
  most MI35x cases. For mi325 jobs, pass `--hostname linux-mi325-gpu-1`.

- **`--single-file` path is relative to test tree** (`/sglang-checkout/test`
  or `/sgl-workspace/sglang/test`). Pass `registered/amd/test_xxx.py`,
  NOT the full absolute path or `python3 ...py`.

- **Partition is rarely needed.** If you know the failing file, always
  prefer `--single-file`. Only use `--partition-id/--partition-size` when
  reproducing the exact CI partition is required (e.g. flaky cross-test
  state issue).

- **`--list-only` is free.** Run it before any 30-min suite job to confirm
  which files the partition would dispatch.

- **Container is kept by default.** The script doesn't auto-rm the
  container. Reuse it with `docker exec -it <name> bash` for follow-up
  experiments. Pass `--rm-container` if you want auto-cleanup.

- **Triton cache is flushed on each invocation.** Don't be surprised if a
  fresh run rebuilds triton kernels (~30s extra) — this is intentional so
  Python source swaps actually take effect.

## Reference files

- Script: `~/SGLang-benchmarks/tools/repro_ci.sh` — full `--help` is the
  authoritative reference for flags
- For aiter v0.1.11 vs v0.1.12 module-level bisection workflow, see
  [bisection.md](bisection.md)
- Investigation doc with empirical data:
  `~/PR/sglang/docs/developer_guide/debug/aiter_2857_mi35x_qwen3_watchdog_race.md`
