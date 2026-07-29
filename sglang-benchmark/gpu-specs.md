# GPU Specifications Reference

## AMD Instinct MI355X

- **Architecture**: CDNA 4
- **Process**: Advanced packaging (3D chiplet)
- **Compute Units**: 304 CUs
- **Memory**: 288 GB HBM3e, 8 TB/s bandwidth
- **Peak Performance**:
  - FP64: 1.18 PFLOPS
  - FP32: 2.36 PFLOPS
  - FP16/BF16: 2.36 PFLOPS
  - FP8/INT8: 4.7 PFLOPS (2:4 sparsity)
- **Interconnect**: 8x Infinity Fabric links, 896 GB/s aggregate
- **TDP**: 500W (liquid-cooled)
- **Form Factor**: OAM (8-GPU nodes: MI355X × 8)
- **Software**: ROCm 7.2+, HIP, hipBLASlt, Composable Kernel, aiter
- **Key Features**:
  - Native FP8 (E4M3/E5M2) matrix cores
  - Structured sparsity (2:4) for INT8/FP8
  - Infinity Fabric for GPU-GPU direct interconnect (no NVLink equivalent)

## NVIDIA B200

- **Architecture**: Blackwell (GB202)
- **Process**: TSMC 4NP
- **CUDA Cores**: 18432
- **Tensor Cores**: 5th gen (576 cores)
- **Memory**: 192 GB HBM3e, 8 TB/s bandwidth
- **Peak Performance**:
  - FP64: 40 TFLOPS
  - FP64 Tensor: 80 TFLOPS
  - FP32: 80 TFLOPS
  - TF32 Tensor: 2.25 PFLOPS (with sparsity: 4.5 PFLOPS)
  - FP16/BF16 Tensor: 4.5 PFLOPS (with sparsity: 9.0 PFLOPS)
  - FP8 Tensor: 9.0 PFLOPS (with sparsity: 18 PFLOPS)
  - FP4 Tensor: 18 PFLOPS (with sparsity: 36 PFLOPS)
- **Interconnect**: NVLink 5 (1.8 TB/s), PCIe Gen6
- **TDP**: 1000W (liquid-cooled, SXM)
- **Form Factor**: SXM (GB200 NVL72 rack scale)
- **Software**: CUDA 13+, cuBLAS, TensorRT-LLM, FlashInfer, DeepGEMM
- **Key Features**:
  - FP4 (new in Blackwell) for inference
  - 2nd gen Transformer Engine (FP4/FP8 auto-casting)
  - NVLink 5 domain: up to 72 GPUs fully connected
  - Confidential Computing, RAS+

## Cache Hierarchy (matters for micro-benchmarks)

| Level | MI355X | B200 |
|-------|--------|------|
| L1 (per CU/SM) | 32 KB vector | 256 KB unified L1/shared |
| L2 | 4 MB per XCD (8 XCDs) | — |
| Last level | **256 MB Infinity Cache (MALL)**, memory-side, shared by all XCDs | 126 MB L2 |
| HBM | 288 GB HBM3e | 192 GB HBM3e |

MI355X's Infinity Cache is the L3-equivalent, but it is *memory-attached*: it
fronts the memory controllers and caches everything headed to HBM, whichever XCD
issued the request. Any benchmark whose working set fits in 256 MB runs at
Infinity-Cache bandwidth, not HBM bandwidth — see
[microbench-pitfalls.md](microbench-pitfalls.md).

## Comparison for LLM Serving

| Metric | MI355X | B200 |
|--------|--------|------|
| HBM Capacity | 288 GB | 192 GB |
| HBM Bandwidth | 8 TB/s | 8 TB/s |
| FP8 TFLOPS (dense) | 2,300 | 9,000 |
| FP8 TFLOPS (sparse) | 4,700 | 18,000 |
| GPU-GPU BW (8-GPU) | 896 GB/s IF | 1,800 GB/s NVLink5 |
| TDP | 500W | 1000W |
| KV Cache Advantage | More HBM (288 GB) | Higher compute density |
| Attention Backend | tilelang (NSA) | trtllm (NSA) |
| MoE Backend | aiter / CK | flashinfer_trtllm |
| AllReduce | ROCM_QUICK_REDUCE (INT4) | flashinfer allreduce fusion |

### SGLang Implications

- **MI355X**: Larger KV cache (288 GB), favors long-context / large-batch decode. ROCm stack uses aiter, Composable Kernel, tilelang.
- **B200**: Higher raw FP8 throughput, NVLink5 enables fast AllReduce. Uses TensorRT-LLM kernels, FlashInfer, JIT DeepGEMM.
- **MoE Dual Stream**: On MI355X, shared-experts-fusion can be disabled (`--disable-shared-experts-fusion`) to enable dual-stream MoE overlap via HIP streams.
