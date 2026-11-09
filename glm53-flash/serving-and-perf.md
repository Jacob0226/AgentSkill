# Memory pool、啟動設定與踩雷

## 目前的設定在哪

不在這裡抄 flag 和實測數字，會過時。以這三處為準：

| 內容 | 位置 |
|---|---|
| server flag | `~/SGLang-benchmarks/Agentic/recipe_glm53flash_fp4_{mi355x,b200}_sglang_mtp.sh` |
| 每個 conc 的 mem-fraction、chunk、mamba、HiCache、ReplaySSM，以及每項的實測理由 | `~/SGLang-benchmarks/Agentic/ix_agentx_glm53flash.sh` 的 `conc_defaults()` 和上方註解 |
| image、PR stack、env、手動步驟 | `pr-stack.md` |

## 兩個 memory pool

| Pool | 內容 | 隨 seq_len 成長？ |
|---|---|---|
| KV pool（`DSATokenToKVPool`） | 11 層 MLA latent KV（512 元素 / token / layer）+ indexer FP8 K（每 4 token 一個） | 是 |
| KDA state pool（`MambaPool`） | 34 層的 conv state + recurrent state，每個 slot 固定大小 | 否 |

TP4 每個 rank 的大小：

| 項目 | 大小 | 說明 |
|---|---|---|
| 一個 KDA slot | 36.9 MB | recurrent state 34 × `(16, 128, 128)` fp32 = 35.65 MB，conv state 1.25 MB |
| ReplaySSM ring | 每個 slot 再加 13.4 MB | 16 token（`--linear-replayssm-cache-len` 預設）× 34 層 × 24.6 KB |
| intermediate SSM cache | (min(max-running, slots / r) + 1) × draft token 數 × 35.65 MB | 只在 MTP 且沒開 ReplaySSM 時有；max-running 80、6 個 draft token 約 16 GiB |

啟動 log 的 `Mamba Cache is allocated` 印的 GB 其實是 GiB，所以用 GB ÷ slot 數會算出 33 MB，比實際的 36.9 MB 小。

### slot 怎麼被用掉

- 每個 running request 固定占 4 個 slot（`extra_buffer` 加 overlap scheduler）。所有 AgentX run 的 `mamba num / #running-req` 中位數都是 4.00。
- `Decode batch` log 的 `mamba num` 是 `capacity − available − evictable`，**不含 radix cache 裡可以 evict 的 prefix state**。所以 `mamba usage` 低，不代表 slot 給太多。
- prefix 要命中，KV 和 mamba state 必須同時還在。哪個 pool 先滿，就由哪個開始 evict。
- AgentX trace 的命中率上限約 95%。MI355X 的 KV 是 B200 的 3 倍，命中率也只有 95.0–95.3%。

### ratio 和固定 slot 數

兩種做法都是啟動時切一次，執行中不會再調整：

- `--mamba-full-memory-ratio r`：mamba 拿剩餘記憶體的 r / (1 + r)。沒開 ReplaySSM 時，intermediate cache 也從這份裡扣，但會被 max-running 截斷，用不完的部分回到 KV。開了 ReplaySSM 就沒有 intermediate，整份都拿去開 slot，KV 反而變少：B200 conc64 從 7.21M 掉到 5.47M token，P90 −7.3%。
- `--max-mamba-cache-size N`：直接指定 slot 數，剩下的全給 KV，優先於 ratio。

固定 slot 數的估法是 N ≈ 1.2 × 4 × decode batch 峰值 + 約 2 × conc，後半是留給 cache 的 session prefix state。改完要確認命中率沒有掉。只有在 KV 不夠的地方改才有好處：conc64 兩個平台都是，B200 conc32 也是（它會用 HiCache）。conc ≤ 32 的 MI355X 有 22M 以上的 KV，改了也沒收益。

AgentX 的 decode batch 遠小於 conc，但 conc64 會因為 subagent fan-out 超過 64：

| conc | decode batch p50 / p90 / 峰值 |
|---|---|
| 16 | 2 / 5 / 11 |
| 32 | 5 / 10 / 24 |
| 64 | 23–26 / 39–42 / 73–77 |

所以 conc64 的 max-running 80 夠用，但餘量不大。

## 已知踩雷

- **`--cuda-graph-max-bs-decode` 要 ≥ `--max-running-requests`。** decode batch 超過 graph 的上限會改跑 eager mode，conc64 的 HBM 幾乎全滿，eager 需要的額外記憶體會讓它 crash。以前 conc64 的機率性 crash（RCCL `HSA_STATUS_ERROR_OUT_OF_RESOURCES`）就是 graph 上限設 64、max-running 設 80 造成的，跟 `HSA_NO_SCRATCH_RECLAIM` 無關。recipe 現在兩個都設 80。
- **HiCache 吃的是 host memory。** TP4、`--hicache-size 216` 的一個 server 要約 900 GB host RAM（每個 rank 約 230 GB RSS）。同一個 node 跑兩條 lane 前，先用 `df -h /dev/shm` 和 `free -g` 確認空間，別人留下的 `/dev/shm` 檔（例如 `lmcache_l1_pool_*`）會讓 server 在配置 host pool 時被 OOM killer 殺掉（exit -9）。
- MI355X 不要加 `--disable-shared-experts-fusion`。gfx950 TP4 上 shared expert 會被併進 routed MoE（`E=289, topk=9`），細節見 `SKILL.md`。
- 第一次呼叫時，fused k-pool 和 AITER mHC kernel 會走 JIT compile，量 conc 1 之前一定要先 warmup。
- `hc_*` 參數是 fp32，列在 `quantization_config.modules_to_not_convert` 裡。MTP block（`model.layers.45.*`）在 hybrid FP8 + NVFP4 checkpoint 裡是 BF16（列在 `quantization_config.ignore`），`_resolve_nextn_quant_config` 會處理。不要手動塞 FP8 quant config 給它，會 load 到沒有 scale 的 BF16 payload 而壞掉。

## CUDA / Blackwell 官方建議（cookbook）

`docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx` 的要點：

- KV dtype 和 DSA backend 要配對切換。Blackwell 建議 FP8 KV + `trtllm` DSA；H100 / H200 預設 BF16 KV + `tilelang` DSA。CUDA 上 TileLang DSA + FP8 KV 不是合法組合。
- 多模態：processor 用 2 FPS 取樣，video 上限 240,000 visual token，要裝 `torchcodec`。長影片在 4×GB300 上建議用 encoder disaggregation（`--encoder-only` + `--language-only` + `--encoder-urls`）隔離 vision encoder 的 memory spike。
- PD disaggregation 只用 dummy weight 驗證過（prefill / decode 都要搬 paged DSA KV 和 KDA recurrent state），而且 PD + speculative decoding 起不來。

## profiling 時的 layer 權重

三種 layer 的數量比是 3 : 11 : 31（KDA + dense MLP / MLA/DSA + MoE / KDA + MoE），所以：

- indexer 的總成本 = 單層 × 11
- MoE 的總成本 = 單層 × 42
- KDA 的總成本 = 單層 × 34
- mHC 的總成本 = 每層 4 次 × 45 = 180 次（AITER gfx950 decode 每層 5 個 kernel，見 `mhc.md`）
