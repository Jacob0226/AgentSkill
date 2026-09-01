# 啟動、調參與實測

## 兩個 memory pool 要一起看

GLM-5.3-Flash 有 **兩個獨立的 pool**，concurrency 取兩者的較小值：

| Pool | 內容 | 隨 seq_len 成長？ |
|---|---|---|
| Paged KV pool（`DSATokenToKVPool`） | 11 層 MLA latent KV（512 元素/token/layer）+ pooled indexer FP8 K cache | **是** |
| KDA state pool（`MambaPool`） | 34 層的 conv state + recurrent state | **否**，每 request 固定 |

TP4 下的具體數字：

- KDA state：**≈ 36.8 MiB / sequence / rank**（固定）
- MLA KV + indexer：**≈ 11.3 KiB / token**（BF16 KV），64K context ≈ 727 MiB / sequence

所以短 context、高 concurrency 的場景常常是 **KDA state pool 先撐爆**。
如果啟動 log 顯示 `max_running_requests` 被自動調降，用
`--mamba-full-memory-ratio` 或 `--max-mamba-cache-size` 加大 KDA pool，
再回頭調 `--max-running-requests`。

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
- **不要 override `linear_lower_bound`**（KDA 的 `gate_lower_bound`），會改變數值。
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

- `--disable-shared-experts-fusion`：cookbook 的 GB300 recipe 有帶，
  shared expert fusion 在這個 checkpoint 上不適用。
- 第一次呼叫時 fused k-pool 與 AITER mHC kernel 會走 JIT compile 路徑，
  量 conc-1 之前一定要先做 warmup run。
- `hc_*` 參數是 fp32 且列在 `quantization_config.modules_to_not_convert` 裡；
  MTP block（`model.layers.45.*`）在 hybrid FP8+NVFP4 checkpoint 裡是列在
  `quantization_config.ignore` 的 BF16 區塊，`_resolve_nextn_quant_config` 會處理，
  不要手動給它塞 FP8 quant config（會 load 到沒有 scale 的 BF16 payload 而壞掉）。

## MI355X TP4 實測（v0.5.18-rocm720-mi35x）

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
- **mHC 的總成本 = 單層 4 次 × 45 = 180 次操作**（gfx95 融合後每層 3 次 launch）
