# Triton specialization cliffs (why one kernel has several speeds)

A Triton kernel decorated with `@triton.heuristics` does not branch at runtime.
Each heuristic is a `tl.constexpr`, so Triton compiles **a separate binary per
combination** and picks one by the shapes you pass. Change a dimension by 16
elements and you can land on a different binary with a different instruction
count.

That makes these flags behave like cliffs, not slopes, and it makes them
invisible to anyone reading the Python source and reasoning about "how much
could a modulo cost".

---

## Reading the specialization off a trace

The flag values are mangled into the kernel name, so a trace tells you which
binary ran without any instrumentation:

```
_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel
  _HAS_BIAS_0_BLOCK_SIZE_M_64_BLOCK_SIZE_N_256_BLOCK_SIZE_K_128_GROUP_SIZE_M_1
  _EVEN_K_0_EVEN_MN_0_cache_modifier_CG
                ^^^^^^^^^^^^
```

If one workload shows several variants of "the same" kernel, they are different
binaries and their per-call times are **not** comparable to each other — they
usually differ in M as well.

---

## EVEN_MN: worth 2x, and not for the reason it looks like

```python
@triton.heuristics({
    "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
                        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
})
```

```python
if EVEN_MN:
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
else:
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
...
if EVEN_MN:
    tl.store(c_ptrs, c)
else:
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
```

Reading that, the honest expectation is "a modulo outside the K loop plus a
masked store, call it 10-20%". Measured on the MLA absorb bmm, BLOCK_M=64,
BLOCK_N=256, everything else pinned:

| | M=16368 (`EVEN_MN_0`) | M=16384 (`EVEN_MN_1`) |
|---|---|---|
| w_kc (K=192) | 311.2 us | 155.8 us |
| w_vc (K=512) | 211.4 us | 129.6 us |

**2x from 16 tokens.** The grid is the same 256 M-tiles either way, so the work
is identical.

### The ISA says where it goes

Dump both `.amdgcn` from the Triton cache and diff them (script below). Same
tile, same `num_warps`, only `EVEN_MN` differs:

| | `EVEN_MN=0` | `EVEN_MN=1` | ratio |
|---|---|---|---|
| total instructions | 3819 | 1562 | **2.44x** |
| `v_mfma*` (the real math) | 32 | 32 | 1.00x |
| `buffer_load*` | 24 | 24 | **1.00x** |
| `buffer_store_short` | **64** | **0** | — |
| `v_cndmask*` (predication) | 297 | 89 | 3.34x |
| `s_mul_hi_u32` (division seq) | 124 | 6 | 20x |
| `s_mul_i32` | 151 | 11 | 14x |
| `s_nop` (pipeline stalls) | 180 | 2 | 90x |

The instruction ratio (2.44x) tracks the runtime ratio (~2x). Three things to
take from this, two of which contradict the obvious guess:

1. **Loads are unaffected.** 24 either way. "The modulo stops loads coalescing"
   is the natural explanation and it is wrong here.
2. **The store collapses.** `buffer_store_short` 64 → 0. The masked store cannot
   be widened, so a tile that was one wide store becomes 64 separate 16-bit
   stores.
3. **`% M` is not one instruction.** M is not a power of two, so the modulo is a
   multiply-high / shift / multiply / subtract sequence, and the compiler cannot
   prove the result is monotonic so it recomputes and predicates all over the
   kernel. `s_nop` 180 vs 2 is the scheduler admitting it could not hide the
   latency it created.

The MFMA count being identical is the whole point: **2.44x the instructions
wrapped around exactly the same math.**

---

## EVEN_K exists too, and is second-order — do not assume symmetry

```python
"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
```

Unlike `EVEN_MN`, this one lives *inside* the K loop: when false, every load in
every iteration carries a mask, not just the tail iteration.

It looks like the same kind of cliff. It is not. On the absorb bmm, w_kc has
K=192 with `BLOCK_SIZE_K=128`, so `EVEN_K=0` and the 2-iteration loop computes
256 of K to use 192. Forcing `BLOCK_SIZE_K=64` makes 192 divide exactly:

| `BLOCK_SIZE_K` | `EVEN_K` | w_kc @ M=16384 |
|---|---|---|
| 64 | 1 | 158.4 us |
| **128 (shipped)** | **0** | **155.3 us** |
| 256 | 0 | 154.6 us |

**Flipping `EVEN_K` to 1 made it slower.** Fewer, larger K iterations beat the
masking cost. Do not "fix" `EVEN_K` by analogy with `EVEN_MN`; measure it.

---

## How to diff two specializations

```python
import glob, os, re
from collections import Counter

STEM = "..._BLOCK_SIZE_M_64_BLOCK_SIZE_N_256_..._EVEN_MN_"   # pin everything else

found = {}
for f in glob.glob(os.path.expanduser("~/.triton/cache/**/*.amdgcn"), recursive=True):
    src = open(f, errors="ignore").read()
    for mn in ("0", "1"):
        if STEM + mn in src:
            found.setdefault(mn, []).append(src)

# several cached copies per variant; take the largest (the full kernel, not a stub)
for mn in ("0", "1"):
    src = max(found[mn], key=len)
    body = [l.strip() for l in src.splitlines()
            if l.startswith("\t") and l.strip() and not l.strip().startswith((".", ";"))]
    ops = Counter(l.split()[0] for l in body)
```

Pin **every** other constexpr in the stem. The cache holds every config from
every sweep you have ever run, so matching on `EVEN_MN` alone will happily
compare a 64x256 kernel against a 32x128 one and hand you a meaningless ratio.

`v_mfma*` is the control: if it differs between the two, you are not comparing
the same work and the instruction ratio means nothing.

---

## Consequences for tuning and for scheduling

- **A tile sweep at a round M measures a binary production never runs.** SGLang's
  scheduler delivers 16289-16368, not 16384, unless you make it exact. Read the
  M off the `prefill[... tok=N ...]` trace annotation and sweep at that.
- **Alignment and tile choice are not independent, but neither gates the other.**
  On this kernel, 32x128 → 64x256 was worth −33.8 ms at `EVEN_MN=0`, and
  `num_warps` 8 → 4 was worth −18 to −22% at `EVEN_MN=0` and −29% at
  `EVEN_MN=1`. Alignment amplifies both; it does not enable them. Do not report
  a config win as "only works once aligned" without measuring it unaligned.
- **Getting `EVEN_MN=1` is a scheduler property, not a kernel property.** No
  amount of kernel tuning produces it; something upstream has to hand the kernel
  an M that divides the tile.
