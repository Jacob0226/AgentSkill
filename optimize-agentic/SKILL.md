---
name: optimize-agentic
description: >-
  優化 InferenceX AgentX（Agentic benchmark）P90 interactivity 的方法：先找出設定 P90 的 tail
  request 實際跑在多大的 decode batch、它們的 TPOT 有多少是 decode step、多少是被 prefill
  卡住的 stall，再決定要 profile decode（哪個 batch、多長 context）還是 prefill。Use when
  optimizing, profiling or explaining Agentic / AgentX P90 Int (P90 interactivity, TPOT tail),
  comparing MI355X vs B200 on Agentic, or choosing what fixed-length profile represents an
  Agentic concurrency.
---

# 優化 Agentic 的 P90 Interactivity

## 先記住：Agentic 不是 fixed length

Agentic 的 conc 是同時在跑的 session（trajectory tree）數，不是 decode batch。session 之間有 tool call 的空檔、subagent fan-out、超長的 prefill（ISL 大多 100k 以上，prefix hit 約 95%），所以同一時間在 decode 的 request 遠少於 conc。

GLM-5.3-Flash TP4 實測（`Decode batch` log 的 `#running-req`）：

| conc | decode batch p10 / p50 / p90 / max |
|---|---|
| 16 | 1 / 2 / 5 / 11 |
| 32 | 2 / 5 / 10 / 24 |
| 64 | 5 / 26 / 42 / 77 |

conc 16 大部分時間的 decode batch 只有 1–4。所以 `BenchFixedLength` 的 conc 16（decode batch 固定 16）不能直接代表 Agentic 的 conc 16。要先知道真相，再決定 profile 什麼。

## P90 Int 是什麼

- `P90 Int = 1 / p90(TPOT)`，每個 request 的 `TPOT = (E2EL − TTFT) / (OSL − 1)`（AIPerf 的 `inter_token_latency`）。取的是最慢 10% 的 request（slow tail），不是平均。細節見 `sglang-benchmark/agentic-benchmark.md` 第 1 節。
- 優化 P90 就是處理極端案例。SGLang 預設 prefill 和 decode 分開跑，prefill 在跑的時候 decode 會停，所以 request 在 decode 期間碰到的 prefill，也會算進它的 TPOT。
- 所以要先回答：**tail 是 prefill 問題還是 decode 問題？如果是 decode，P90 來自 decode batch 多少？**

## 流程

### 1. 拆解 tail：`~/SGLang-benchmarks/tools/agentic_tail.py`

```bash
python3 ~/SGLang-benchmarks/tools/agentic_tail.py \
    <sweep>/glm5.3flash_tp4_conc16_kvnone_spec-mtp <B200 同一點> --label MI --label B200
```

工具讀 `server.log` 和 `aiperf_artifacts/profile_export.jsonl`，輸出：

- **P90 intvty** 自己重算一次，要等於報告的值，否則資料讀錯。
- **P90 intvty, stalls removed**：把 stall 拿掉後的上限。和實際值差很多，代表 prefill 是主因。
- **Decode batch over the run**：整段 run 的 decode batch 分布。
- **Per-user ms per token by decode batch**：從 server log 算的每個 decode batch 下的速度。p50 接近純 decode，p90 含 stall。沒有 jsonl 也能算，可以拿來跨平台比較。
- **Requests P85–P95 (sets P90)**：決定 P90 的那群 request。看它們的：
  - `step part` 和 `stall part`：TPOT 拆成 decode step 和 stall。gap 超過該 request 中位 gap 2 倍的部分算 stall（`--stall-mult` 可調）。
  - `decode batch p10/p50/p90`：它們 decode 期間實際的 decode batch。這就是要 profile 的 batch。
  - `ISL (mean)`：要 profile 的 context 長度。
  - `prefill tok in window`：它們 decode 期間 server 做了多少 prefill。

### 2. 判斷，再決定 profile 什麼

| P85–P95 的 stall share | 主因 | 要 profile 的東西 |
|---|---|---|
| < 20% | decode | fixed-length decode，conc = 該群的 decode batch p50（也看 p10 / p90），input ≈ 該群 ISL，MTP 和 golden AL 跟 Agentic 一樣 |
| 20–40% | 兩者都有 | decode 和 EXTEND 都做 |
| > 40% | prefill | EXTEND（chunked prefill，長 cached prefix）。knob：`--chunked-prefill-size`、`--prefill-max-requests`、mixed chunk、HiCache load-back |

### 3. 跨平台比較

- 兩邊跑同一個分析。B200 的 run 要保留 `profile_export.jsonl`，不然只剩 server log 那張 per-batch 速度表。
- 先比設定再怪 kernel。例如 B200 golden AL 線在 conc 32/64 用了 `--prefill-max-requests 1`、`--mamba-full-memory-ratio 1.5`、conc 32 就開 HiCache 216；MI355X 的 conc 32 沒有 HiCache。
- per-batch 速度相同、P90 卻輸，代表輸在 stall 或 batch 分布，不是 decode kernel。

### 4. 驗證

- 優化後一定要回 Agentic 跑 3600s 才能 quote throughput 和 TTFT。只看 interactivity 的話 1200s 就夠（見 `agentic-benchmark.md` 第 2 節）。
- 改完再跑一次 `agentic_tail.py`，確認 P85–P95 那群的 step part 或 stall part 真的變小，不是 batch 分布變了。

## 陷阱

- **warmup 和 error 的 record 不算**，只算 `benchmark_phase == profiling` 且有 `inter_token_latency` 的 record，跟 InferenceX 一樣。
- **回饋效應（Little's law）**：request 越慢，同時在 decode 的就越多，decode batch 越大，又更慢。MI355X 的 conc 32 在 decode batch 13–24 的時間占 5.3%，B200 只有 1.2%。比較 per-batch 速度時，要記得兩邊的 batch 分布也不同。
- **server log 的時間解析度**：`Decode batch` 每 40 個 decode step 印一行，timestamp 只到秒。單一 request 的 batch 是平均值，看 group 統計就好。
- **golden AL**：每步固定約 3.56 token（tokens / chunk），real AL 的 tail 會更差，兩種線不要混著比。
- **monokernel 的行數上限**：verify 最多 16 行。MTP5 每個 request 6 個 token，只能涵蓋 decode batch ≤2；MTP3 是 4 個 token，涵蓋 ≤4。目前 conc 16–64 沒有開 monokernel。
- **不考慮 multi stream。**

## GLM-5.3-Flash 實例（2026-11-08）

MI355X 1106 + PRs + Mono（conc 16–64 沒有 mono，FP8 KV，MTP5 golden 3.56）vs B200 v0.5.23 golden AL。資料：

- MI355X：`~/SGLang-benchmarks/results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.23-rocm10-mi35x-20261106/bench-Agentic-TP4_EP1-mtp5-prstack09111fe9a3-pr6435-1107-golden3.56-dsa-triton/`
- B200：`/shared_nfs/jacchang/b200_agentic/v0523-hc216-golden-line/`（只有 server log，沒有 `profile_export.jsonl`）

| | conc 16 | conc 32 | conc 64 |
|---|---|---|---|
| P90 Int MI / B200 | 220.9 / 232.1 | 143.1 / 158.0 | 59.5 / 62.5 |
| MI P85–P95 的 decode batch p10/p50/p90 | 3 / 6 / 9 | 8 / 16 / 19 | 32 / 47 / 61 |
| MI P85–P95 的 TPOT：step + stall（ms） | 3.93 + 0.61 | 5.43 + 1.70 | 9.67 + 7.37 |
| MI P85–P95 的 stall share | 13% | 24% | 43% |
| MI P90 Int 拿掉 stall | 256 | 180 | 99 |
| per-user ms/token p50，MI / B200 | bs 5–8：3.86 / 3.50 | bs 13–16：5.87 / 5.63 | bs 33–48：11.52 / 11.51 |

解讀：

- **conc 16 輸在 decode**。P90 來自 decode batch 約 6（3–9）、ISL 約 150k 的 request。MI 在 bs 1–8 每個 token 慢約 10%。
- **conc 32 兩者都有**。P90 來自 decode batch 約 16，stall 占 24%。
- **conc 64 輸在 prefill**。P90 來自 decode batch 約 47，stall 占 43%；bs 33–64 的 decode 速度和 B200 一樣。
