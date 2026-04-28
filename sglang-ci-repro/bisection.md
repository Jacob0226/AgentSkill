# Module-level bisection between aiter versions

Used when commit-level `git bisect` doesn't work (e.g. aiter v0.1.11 ↔
v0.1.12 share zero ancestry due to upstream history rewrite, so `git
bisect bad/good` walks back to 2024-11-11 "Initial commit").

## Workflow

1. Pre-extract both aiter trees from existing images (one-time, ~2 min):

   ```bash
   mkdir -p /tmp/aiter_diff/v011 /tmp/aiter_diff/v012
   docker run --rm <V011_IMAGE> tar cf - -C /sgl-workspace aiter \
       | tar xf - -C /tmp/aiter_diff/v011
   docker run --rm <V012_IMAGE> tar cf - -C /sgl-workspace aiter \
       | tar xf - -C /tmp/aiter_diff/v012
   ```

2. Compute the diff:

   ```bash
   cd /tmp/aiter_diff
   (cd v011/aiter && find aiter -type f \! -path "*/jit/*" \! -name "*.so" \
       | sort > /tmp/aiter_diff/v011_files.txt)
   (cd v012/aiter && find aiter -type f \! -path "*/jit/*" \! -name "*.so" \
       | sort > /tmp/aiter_diff/v012_files.txt)

   # Source-only diff (drop .hsaco / .json compiled artifacts):
   comm -23 v011_files.txt v012_files.txt | grep -vE "\.(hsaco|json)$"  # removed
   comm -13 v011_files.txt v012_files.txt | grep -vE "\.(hsaco|json)$"  # added

   # Files in both with content diff:
   comm -12 v011_files.txt v012_files.txt | while read f; do
       diff -q "v011/aiter/$f" "v012/aiter/$f" >/dev/null 2>&1 || echo "$f"
   done > /tmp/aiter_diff/changed_files.txt
   ```

3. Group changes by directory to identify suspect modules:

   ```bash
   awk -F'/' '{n=NF; if(n>=4) print $1"/"$2"/"$3; else print $1"/"$2}' \
       /tmp/aiter_diff/changed_files.txt | sort | uniq -c | sort -rn
   ```

4. Run A/B with the helper. The `exp.sh` helper at `/tmp/repro_2857/exp.sh`
   (created during the aiter#2857 investigation) overlays specific files
   from one tree onto the other image and runs GSM8K:

   ```bash
   bash exp.sh LABEL <relative_path1> [<relative_path2> ...]
   # e.g.
   bash exp.sh fused_moe_only aiter/fused_moe.py
   bash exp.sh fp4_utils_only aiter/utility/fp4_utils.py
   ```

5. For testing patched files (e.g. revert just one function body in v012's
   fused_moe.py), put the patched copy at
   `/tmp/aiter_diff/exp_<label>/aiter/aiter/<file>.py` then run
   `EXP_SRC_TREE=/tmp/aiter_diff/exp_<label>/aiter bash exp.sh ... <file>`.

## When this works

- Suspect is in **Python source** (Triton kernels, dispatcher logic,
  module wrappers). Swapping a `.py` file plus flushing Triton cache
  picks up the change without rebuilding.

## When this doesn't work

- Suspect is in **compiled C++/HIP** (`module_*.so` files in
  `/sgl-workspace/aiter/aiter/jit/`). Python overlay can't change kernel
  behavior. Need to either:
  1. Rebuild aiter from source with `python3 setup.py develop`
     (10-30 min), or
  2. Test with a different pre-built image entirely.

- Cross-version *Python overlay* breaks because of API drift (e.g. v012
  imports `from aiter import mxfp4_moe_sort_fwd` which doesn't exist in
  v011). Symptoms: `ImportError` at server start, or `SIGABRT` later
  when downstream HIP kernels read a Triton-produced tensor with subtly
  different layout/strides. In this case the regression is in
  co-designed Python+C++ pair and can't be cleanly bisected at Python
  level — see the aiter#2857 investigation doc for an example.

## Reference

- Investigation doc with the actual aiter v0.1.11 → v0.1.12 GSM8K
  regression bisection (Exp A/B/C/D + v0.1.13.dev0 attempt):
  `~/PR/sglang/docs/developer_guide/debug/aiter_2857_mi35x_qwen3_watchdog_race.md`
  ("Module-level delta-debugging" section)
