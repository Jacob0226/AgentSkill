# 啟動、調參與實測

## 兩個 memory pool 要一起看

GLM-5.3-Flash 有 **兩個獨立的 pool**，concurrency 取兩者的較小值：

| Pool | 內容 | 隨 seq_len 成長？ |
|---|---|---|
| Paged KV pool（`DSATokenToKVPool`） | 11 層 MLA latent KV（512 元素/token/layer）+ pooled indexer FP8 K cache | **是** |
| KDA state pool（`MambaPool`） | 34 層的 conv state + recurrent state | **否**，每 request 固定 |

TP4 下的具體數字：

- KDA state：**≈ 35.2 MiB（36.9 MB）/ sequence / rank**（固定，見 `kda-linear-attention.md`）
- MLA KV + indexer：**≈ 11.3 KiB / token**（BF16 KV），64K context ≈ 727 MiB / sequence
- 開 MTP 時還有一塊 intermediate SSM cache，量到約 16.6 GB / rank，跟 `--max-running-requests` 成正比

哪個 pool 先滿看 context 長度與 `--mamba-full-memory-ratio`：

- 短 context、高 concurrency：常常是 **KDA state pool 先撐爆**。啟動 log 顯示
  `max_running_requests` 被自動調降時，用 `--mamba-full-memory-ratio` 或
  `--max-mamba-cache-size` 加大 KDA pool，再回頭調 `--max-running-requests`。
- Agentic（長 context、高 prefix reuse）：反過來是 **KV pool 先滿**。conc 64、沒有 HiCache 時
  KV pool 100% 滿；conc 32 在 2× max-running 下只到 49%。解法是把 mamba ratio 調低（0.5）、
  max-running 從 128 降到 80（省下 MTP intermediate cache 給 KV），再開 HiCache。

## MI355X / ROCm 啟動範例（PR #36607 的量測設定）

```bash
export SGLANG_USE_AITER=1
export SAFETENSORS_FAST_GPU=1

python3 -m sglang.launch_server \
  --model-path /data/huggingface/hub/zai-org/GLM-5.3-Flash \
  --tp-size 8 \
  --context-length 65536 \
  --kv-cache-dtype bfloat16 \
  --moe-runner-backend aiter \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 1 32 \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache \
  --reasoning-parser glm45 \
  --tool-call-parser glm47
```

完整腳本：`~/SGLang-benchmarks/pr36607_glm53flash.sh`（`TP` 與 `KV_DTYPE` 可用環境變數覆寫）。

## CUDA / Blackwell 官方建議（cookbook）

`docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx` 的要點：

- **KV dtype 與 DSA backend 必須配對切換**。Blackwell 建議 FP8 KV + `trtllm` DSA
  （GB300 上量到 throughput +2.3–5.5%、同樣 pool bytes 下 KV token 容量約 1.8 倍，
  GSM8K 在誤差範圍內）；H100 / H200 預設 BF16 KV + `tilelang` DSA。
  **TileLang DSA + FP8 KV 在 CUDA 上不是合法組合。**
- MTP 用 `--speculative-algorithm NEXTN`，低延遲組合是
  `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --speculative-adaptive`（`index_share_for_mtp_iteration` 需要 eagle-topk = 1）。
  高 throughput 場景直接關掉 speculative decoding。
- 多模態：processor 用 2 FPS 取樣、video 上限 240,000 visual token，要裝 `torchcodec`。
  長影片在 4×GB300 上建議用 encoder disaggregation（`--encoder-only` + `--language-only`
  + `--encoder-urls`）把 vision encoder 的 memory spike 隔離開。
- **不要 override `linear_attn_config.gate_lower_bound`**，會改變數值。model 讀的是
  `linear_attn_config["gate_lower_bound"]`；頂層 `linear_lower_bound` 只在 config 沒有
  `linear_attn_config` 時才當 fallback，兩份 checkpoint 都有，所以改它沒有效果。
- PD disaggregation 目前只用 dummy weight 機械性驗證過（prefill/decode 都要搬 paged DSA KV
  *和* KDA recurrent state），且 PD + speculative decoding 起不來。

## 已知踩雷

### FP8 KV cache + 非 trtllm DSA backend 會炸

GLM-5.3-Flash 是 hybrid，attention backend 被包在 `HybridLinearAttnBackend` 裡。
FP8 KV 搭配非 trtllm 的 DSA backend 時，讀取會走
`_get_mla_kv_buffer_from_fp8_for_dsa()`，它向 `get_attn_backend()` 要
`.forward_metadata` —— 那個呼叫會 unwrap `TboAttnBackend` 但 **不會 unwrap hybrid wrapper**，
於是 raise。

觸發條件是「chunked-prefill 續傳 + prefix reuse」，所以 1024-token 的短 shape
量測不會踩到（`results/GLM-5.3-Flash/CRASHED-fp8kv-untagged-docker/` 就是踩到的紀錄）。
**ROCm 上請用 `--kv-cache-dtype bfloat16`。**

### 其他

- `--disable-shared-experts-fusion`：cookbook 的 GB300 recipe 有帶。ROCm 上的 shared expert
  fusion 來自 #41161（FP8 ckpt）和 #41870（Quark MXFP4 ckpt），**兩個都已 merge、在 1012 image 裡**。
  `shared_experts_fusion_disable_reason` 只在 AITER gfx950、TP4 / TP8、EP1、沒有 MoE a2a、
  E=288 / topk 8 / hidden 4096 / moe_intermediate 2048、clamped SiLU（limit 10.0），而且 routed 與
  shared expert 同一種量化 layout 時放行。啟動 log 會印 `Shared experts fusion optimization enabled.`，
  aiter fused_moe 的 key 變成 `E=289, topk=9`；MI355X 不要加這個 flag。
- 第一次呼叫時 fused k-pool 與 AITER mHC kernel 會走 JIT compile 路徑，
  量 conc-1 之前一定要先做 warmup run。
- `hc_*` 參數是 fp32 且列在 `quantization_config.modules_to_not_convert` 裡；
  MTP block（`model.layers.45.*`）在 hybrid FP8+NVFP4 checkpoint 裡是列在
  `quantization_config.ignore` 的 BF16 區塊，`_resolve_nextn_quant_config` 會處理，
  不要手動給它塞 FP8 quant config（會 load 到沒有 scale 的 BF16 payload 而壞掉）。

### `HSA_NO_SCRATCH_RECLAIM=1` + agentic conc 64 會機率性 crash

conc 64（HBM 幾乎滿）時，RCCL 的 `ncclDevKernel_Generic_1` launch 會失敗，錯誤是
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`，通常在開跑後 55–71 分鐘。這是機率性的，
不是 regression：`=1` 的 15 次完整 conc 64 run 裡 6 次 crash，而且 crash 跟 high-load
decode batch 的數量沒有對應關係。
`HSA_NO_SCRATCH_RECLAIM=0` 有 2 次完整 conc 64 run 都沒 crash，但 206 上那次 P90 interactivity
只有 19.17（`=1` 的 prstack-fix 是 22.92），所以 recipe 維持 image 預設的 `=1`，
conc 64 crash 時就重跑。crash 前剩餘 HBM 只有 0.28–1.5 GB（HSA 錯誤訊息裡的 Available Free mem）。

## 1012 image 與 PR stack

Image `rocm/sgl-dev:v0.5.21-rocm10-mi35x-20261012` 內建 sglang `d1357bf87b`、aiter
`e7d2453f2`（在 container 裡用 `git log -1` 確認過）。目前的 GLM-5.3 PR 狀態：

| PR | 內容 | 狀態 | 1012 image 裡？ |
|---|---|---|---|
| sglang #41161 | shared expert 併進 AITER MoE（FP8） | merged | **有** |
| sglang #41870 | Quark MXFP4 的 shared expert fusion + KDA projection fusion | merged | **有** |
| sglang #41615 | GLM-5.3 改用 Triton sparse MLA，新增 `aiter_sparse_mla` | open | 沒有，PR stack 疊 |
| sglang #41871 | MXFP4 small-M MoE kernel 支援 GLM-5.3 TP4 shape（intermediate 512、8/9 slots、≤16 token、`swiglu_limit`）+ small sort 融合 MXFP4 quant | open | 沒有，PR stack 疊（image 只有 Qwen3.5 的 10/11 slots，所以 GLM-5.3 走不到 small-M） |
| sglang #42020 | FFN all-reduce + `hc_post` 融合（`SGLANG_ROCM_FUSED_AR_MHC_POST`） | open | 沒有，PR stack 疊 |
| sglang #38764 | KDA projection opt-in PTPC FP8（`SGLANG_OPT_GLM53_KDA_PTPC_MODULES`） | open | 沒有，PR stack 疊 |
| sglang #39121 | KDA prefill kernel tuning | **closed，沒 merge** | 沒有，PR stack 也沒有 |
| aiter #5933 | `gemm_a16w16` N=288 K=4096 config（router GEMM） | merged | 沒有，手動複製 JSON |
| aiter #5982 | GLM-5.3 TP4 fused KDA in-proj BF16 GEMM tuning | merged | 沒有，手動複製 CSV |
| aiter #6276 | gfx950 `sparse_mla` bf16 KV-tile store 去 bank conflict | merged | 沒有，也沒補；只影響 `aiter_sparse_mla` |

PR stack 在 `~/PR/wt-glm53-1012-prstack`（branch `glm53-1012-prstack`，依序疊 #41615、#41871、
#42020、#38764）。啟動方式見 `~/SGLang-benchmarks/tmp/agentic_1014/run_triton.sh`：

- 把 `$TREE/python` 複製到 container 的 `/root/ab_trees/prstack`，用 `PYTHONPATH` 蓋過 image 的 sglang。
- 把 aiter#5933 的 `GEMM-A16W16-N=288-K=4096.json` 與 aiter#5982 的 `glm53_flash_bf16_{tuned,untuned}_gemm.csv`
  複製進 image 的 aiter。
- 環境變數：`SGLANG_ROCM_FUSED_AR_MHC_POST=1`、
  `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`。
- DSA backend：`--dsa-prefill-backend triton --dsa-decode-backend triton`（Agentic 腳本是 `--dsa-backend triton`）。

在 1012 image 的 container 裡可以這樣確認哪些 PR 生效：server log 有
`Shared experts fusion optimization enabled.` 與 `E=289, topk=9`；decode trace 有
`smallm_p2_i512_s9`（#41871）、`allreduce_mhc_post_large_m_kernel`（#42020）、
`_sparse_mla_decode_split_kernel`（Triton DSA）。

## MI355X TP4 agentic 設定與實測（0928 image recipe）

Recipe：`~/SGLang-benchmarks/Agentic/recipe_glm53flash_fp4_mi355x_sglang_mtp.sh`，
image `rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928` + 當時的 PR stack（DSA 還是 TileLang），checkpoint
`amd/GLM-5.3-Flash-Quark-MXFP4`，TP4 / EP1。

```bash
python3 -m sglang.launch_server \
  --model-path amd/GLM-5.3-Flash-Quark-MXFP4 --trust-remote-code \
  --tp 4 --ep-size 1 \
  --kv-cache-dtype bfloat16 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --moe-runner-backend aiter \
  --mamba-radix-cache-strategy extra_buffer \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --chunked-prefill-size 16384 --mem-fraction-static 0.85 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --max-running-requests $MAX_RUNNING --cuda-graph-max-bs-decode $CUDA_GRAPH_MAX_BS \
  --mamba-full-memory-ratio $MAMBA_RATIO
  # conc 64 再加：
  # --enable-hierarchical-cache --hicache-write-policy write_back --hicache-io-backend direct
  # --hicache-mem-layout page_first_direct --hicache-size 169
```

| conc | max-running | cuda-graph max bs | mamba ratio | HiCache |
|---|---|---|---|---|
| 1–32 | max(2 × conc, 8) | = max-running | 0.9 | 無 |
| 64 | 80 | 64 | 0.5 | 169 GB / rank |

結果（`bench-Agentic-TP4_EP1-mtp5-prstack-fix`，`HSA_NO_SCRATCH_RECLAIM=1`；
同設定重跑的差異大約 ±3%，conc 32 約 ±6%）：

| conc | P90 interactivity (tok/s/user) | total throughput (input + output, tok/s/GPU) |
|---|---|---|
| 1 | 259.18 | 6021 |
| 4 | 222.21 | 8894 |
| 8 | 187.30 | 15095 |
| 16 | 134.79 | 26902 |
| 32 | 73.59 | 53569 |
| 64 | 22.92 | 64432 |

## MI355X TP4 實測（舊資料：v0.5.18-rocm720-mi35x，zai-org FP8 checkpoint）

這節是早期 fixed-shape 量測，跟上面的 agentic 設定不同，只能當作參考。

Docker `lmsysorg/sglang:v0.5.18-rocm720-mi35x`，commit `aa8c950a3d`（+ local k-pool
routing revert），機器 `smci355-ccs-aus-m15-21`，FP8 權重 / BF16 KV，GSM8K accuracy **0.9712**。
資料來源：`~/SGLang-benchmarks/results/GLM-5.3-Flash/lmsysorg_sglang-v0.5.18-rocm720-mi35x/bench-Fixed-TP4/all_benchmarks.csv`

### in 1024 / out 1024

| conc | tok/s/user | tok/s/GPU | TTFT (ms) | TPOT (ms) | ITL (ms) |
|---|---|---|---|---|---|
| 4 | 94.8 | 180 | 103 | 10.6 | 10.3 |
| 8 | 84.1 | 326 | 104 | 11.9 | 11.4 |
| 16 | 66.8 | 517 | 109 | 15.0 | 13.8 |
| 32 | 56.0 | 857 | 111 | 17.9 | 15.6 |
| 64 | 39.3 | 1221 | 138 | 25.5 | 21.2 |

### in 8192 / out 1024

| conc | tok/s/user | tok/s/GPU | TTFT (ms) | TPOT (ms) | ITL (ms) |
|---|---|---|---|---|---|
| 4 | 90.8 | 768 | 287 | 11.0 | 10.3 |
| 8 | 77.5 | 1329 | 288 | 12.9 | 11.2 |
| 16 | 60.2 | 2043 | 289 | 16.6 | 13.0 |
| 32 | 45.0 | 3060 | 287 | 22.2 | 14.8 |
| 64 | 29.3 | 4083 | 442 | 34.1 | 18.7 |

### in 70000 / out 300

| conc | tok/s/user | tok/s/GPU | TTFT (ms) | TPOT (ms) | ITL (ms) |
|---|---|---|---|---|---|
| 4 | 35.4 | 6039 | 2477 | 28.2 | 10.2 |
| 8 | 18.1 | 6924 | 3022 | 55.2 | 11.2 |
| 16 | 9.6 | 7536 | 4423 | 103.9 | 13.0 |
| 32 | 4.8 | 7899 | 5640 | 208.9 | 14.7 |
| 64 | 2.3 | 7916 | 8070 | 434.1 | 18.9 |

值得注意的兩件事：

- **ITL 幾乎不隨 input length 成長**（conc 4 時 10.3 / 10.3 / 10.2 ms）。
  這正是 hybrid 架構的效果：34 層的 state 大小固定，只有 11 層的 KV 會隨 context 變長。
- 70K input 的 tok/s/GPU 在 conc 16 之後就飽和在 ~7.9K，
  瓶頸從 decode 轉到 prefill（TTFT 從 2.5 s 一路漲到 8 s）。

## 對照 profiling 用的 layer 權重

分析 trace 的 layer breakdown 時，記得三種 layer 的數量比是 **3 : 11 : 31**：

| layer type | 數量 | 佔比 |
|---|---|---|
| KDA + dense MLP | 3 | 6.7% |
| MLA/DSA + MoE | 11 | 24.4% |
| KDA + MoE | 31 | 68.9% |

所以：
- **indexer 的總成本 = 單層 indexer × 11**
- **MoE 的總成本 = 單層 MoE × 42**
- **KDA 的總成本 = 單層 KDA × 34**
- **mHC 的總成本 = 單層 4 次 × 45 = 180 次操作**（AITER gfx950 decode 每層 5 個 kernel，見 `mhc.md`）
