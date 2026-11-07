# PR stack（MI355X 重跑 benchmark 前必讀）

最後更新：2026-11-07（台灣時間）。「狀態」欄是當天用 `gh pr view` 查的，重跑前先照最後一節「重跑前檢查」更新一次。

MI355X 的數字一律來自「image + 自己疊的 PR stack + 手動步驟」三層；漏掉任何一層，結果就跟之前的報告對不上。B200 用 stock image，沒有 PR stack（見「B200 對照」）。

## 目前基準（1026 image，11/06 Agentic 數字用的這套）

| 項目 | 值 |
|---|---|
| Image | `rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261026`，image 內 sglang `5ef99013ac`、aiter `285579b62` |
| sglang tree | `/shared_nfs/jacchang/exp/sglang-fp8kv-1103`，branch `glm53-1105-bcg-hybrid-dsa`，HEAD `b1c1c92cdc`。底是 image 的 `5ef99013ac`，上面 14 個 commit；用 `PYTHONPATH=$SGL/python` 蓋過 image 的 sglang |
| mono tree | `/shared_nfs/jacchang/exp/sgl-amd-mono-iscore`（branch `glm53-flash-dsa-iscore-req`）也放在 `PYTHONPATH` 最前面；Agentic recipe 沒開 monokernel |
| 啟動鏈 | `/shared_nfs/jacchang/agentic_1106/lane_env_dur.sh`（`DUR` 秒數，預設 1800）→ `~/SGLang-benchmarks/Agentic/ix_agentx_glm53flash.sh --platform mi355x` → `recipe_glm53flash_fp4_mi355x_sglang_mtp.sh` |

## sglang：自己疊的 PR（依疊的順序）

stack 裡的 commit 是當時的 PR head 套在 `2815eed49c` 上再搬到 `5ef99013ac`；「stack head → 目前 head」不同時，代表 PR 之後有更新，stack 還是舊版。

| # | PR | 內容 | 狀態（11/07） | stack head → 目前 head | 生效怎麼看 |
|---|---|---|---|---|---|
| 1 | sglang #41615 | GLM-5.3 改用 Triton sparse MLA（gfx950） | open | `acca6a72db`（相同） | decode trace 有 `_sparse_mla_decode_split_kernel` |
| 2 | sglang #41871 | small-M MoE kernel + sort 融合 MXFP4 quant | open | `ac5057efad` → `2f3232f34d` | decode trace 有 `smallm_p2_i512_s9` |
| 3 | sglang #42020 | FFN all-reduce + `hc_post` 融合，env `SGLANG_ROCM_FUSED_AR_MHC_POST=1` | **closed，沒 merge**（draft 23 天沒更新，stale bot 10/25 自動關） | `bd5ddf3358`，改接 `ExitDecision(sum_in_reduce_scatter)` API | decode trace 有 `allreduce_mhc_post_large_m_kernel` |
| 4 | sglang #38764 | KDA projection opt-in PTPC FP8，env `SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj` | open | `1d33eed202` → `c8c963cf0f` | KDA projection 在 trace 裡是 FP8 GEMM，不是 BF16 |
| 5 | sglang #44415 | MoE correction bias 保持 fp32（bug fix，沒有效能影響） | open | `4272d975f1` → `2e8c7f9b39` | — |
| 6 | sglang #44416 | MTP verify：one-launch MoE router + 較窄的 KDA V tile（疊在 #44415 上） | open | `9b1d711bba` → `3d692f0a25` | — |
| 7 | sglang #44417 | MTP verify：減少 DSA k-pool indexer launch | open | `60336705fe` → `8499d23086` | — |
| 8 | sglang #46036 | mono-kernels 依 model type 註冊 spec | **merged**（在 image 之後） | `f73a82b40e` | 換到含它的 image 後可從 stack 拿掉 |
| 9 | sglang #46808 | HIP verify 的 sort-free top-p + 多 row softmax（stack 裡是 3 個 commit：`2514ded128`、`d41967ade8`、`42665af0e0`） | open | 內容同 PR 的 3 個 commit，PR 另有 merge main | — |
| 10 | 沒有 PR | `f0a7a579d5` GLM-5.3-Flash 在 gfx950 Triton sparse-MLA 允許 FP8 e4m3 KV（依賴 #41615） | 只在內部 tree | — | `--kv-cache-dtype fp8_e4m3` 能啟動且走 Triton DSA |
| 11 | sglang #48875 | BCG：hybrid KDA 包住的 DSA 也能 replay breakable prefill graph，graph 裡的 KDA extend 走 HIP | draft，open | stack `e4cd94409d`、`b1c1c92cdc`；PR 是 `3121fec7f8`、`c9670de08c`（同內容 rebase 到 main） | 99,840 prefix + 16 new tokens 的 prefill 約 37 ms（沒 replay 時約 99 ms） |

### 已在 1026 image 裡，不用疊

| PR | 內容 |
|---|---|
| sglang #41161 / #41870 | shared expert 併進 AITER MoE（log：`Shared experts fusion optimization enabled.`、`E=289, topk=9`）；#41870 另含 KDA projection fusion |
| sglang #43182 | GLM-5.3-Flash interleave BCG（batched prefill、DFlash2） |
| sglang #45710 / #45713 | gfx950 HIP KDA chunked-prefill / fused decode kernel（#48875 的 HIP extend 用的就是 #45710） |

## aiter

| PR | 內容 | 狀態 | 在 1026 image（aiter `285579b62`）？ | 要做什麼 |
|---|---|---|---|---|
| aiter #5933 | `gemm_a16w16` N=288 K=4096 config（router GEMM） | merged 10/02 | 有 | 不用 |
| aiter #5982 | GLM-5.3-Flash TP4 fused KDA in-proj BF16 GEMM tuning | merged 10/12 | 有 | 不用 |
| aiter #6276 | gfx950 `sparse_mla` bf16 KV-tile store 去 bank conflict | merged 10/13 | 有 | 不用（只影響 `aiter_sparse_mla`） |
| aiter #6435 | GLM-5.3-Flash MXFP4 fused-MoE tuned CSV（TP4） | merged 10/26 | **沒有** | 要改 code 並 rebuild `module_moe_mxfp4_aux`，見下一節 |

「在 image 裡？」是用 `git merge-base --is-ancestor <merge commit> 285579b62` 確認的。

## 手動步驟（recipe 沒做的）

**MoE tuned CSV。** recipe 沒有設 `AITER_CONFIG_FMOE`，忘了傳就是沒有 tuned CSV（30 分鐘 run 的 TPUT 和 P90 分別差約 1.8% 和 4.8%）。兩個 CSV 在 60 分鐘 run 的差異在 noise 內：

| 選項 | CSV | 額外步驟 |
|---|---|---|
| 我們 tune 的 | `~/SGLang-benchmarks/tmp/moe2_tune/glm53_flash_fp4_tuned_fmoe.csv` | 無 |
| aiter #6435 | `/shared_nfs/jacchang/aiter6435_1106/glm53flash_fp4_tuned_fmoe.csv` | container 裡先跑 `/shared_nfs/jacchang/aiter6435_1106/patch_aiter.sh`（加 6 個 shape、rebuild aux module） |

傳法：`AITER_CONFIG_FMOE` = image 預設清單（`/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv` 加上 `model_configs/` 下所有 `*fmoe*.csv`，用 `:` 串起來）再接上選的 CSV，當 `lane_env_dur.sh` 的第 5 個參數（`KEY=VALUE`）傳進去。完整字串見 `/shared_nfs/jacchang/agentic_1106/driver.log`。

**env（lane 腳本 export）。** `SGLANG_ROCM_FUSED_AR_MHC_POST=1`（#42020）、`SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`（#38764）。`HSA_NO_SCRATCH_RECLAIM=1` 維持 image 預設，不要改。

**driver 參數。** `--conc 64 --mtp-steps 5 --dsa-backend triton --mamba-ratio 0.5 --hicache-size 216 --hicache-io kernel`。BCG 的 `--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2560`、FP8 KV、INT4 quick-reduce 已經寫在 recipe 裡。

## B200 對照

B200 用 stock image，不疊 PR。60 分鐘的對照點是 `lmsysorg/sglang:v0.5.23-cu130`、dgx-023、10/30 單獨跑（InferenceX curve 那條線）。30 分鐘和 60 分鐘的 run 不能互比，MI355X 和 B200 要用同樣長度比。

## 重跑前檢查

1. 更新 PR 狀態，跟上面的表比對：

   ```bash
   GH="env -u GH_TOKEN -u GITHUB_TOKEN /home/jacchang/bin/gh"
   for p in 41615 41871 42020 38764 44415 44416 44417 46036 46808 48875; do
     $GH pr view $p --repo sgl-project/sglang --json number,state,headRefOid \
       --jq '"#\(.number) \(.state) \(.headRefOid[0:10])"'
   done
   ```

2. 換 image 時，先在 container 裡查 image 內的 sglang / aiter commit，再用 `git merge-base --is-ancestor` 判斷哪些 merged PR 已經在 image 裡，在的就從 stack 拿掉。
3. open PR 的 head 有更新時，先問使用者要不要改疊新 head；改了就要重跑 GSM8K。沒改就在報告寫明用的是舊 head。
4. 開跑後確認生效訊號（上表「生效怎麼看」），至少看 server log 的 `E=289, topk=9` 和一份 decode trace。
5. 重跑完更新這份文件的「最後更新」和狀態欄。

## 舊 image 的 PR stack（歷史紀錄）

### 1012 image

Image `rocm/sgl-dev:v0.5.21-rocm10-mi35x-20261012` 內建 sglang `d1357bf87b`、aiter `e7d2453f2`（在 container 裡用 `git log -1` 確認過）。當時的 GLM-5.3 PR 狀態：

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

PR stack 在 `~/PR/wt-glm53-1012-prstack`（branch `glm53-1012-prstack`，依序疊 #41615、#41871、#42020、#38764）。啟動方式見 `~/SGLang-benchmarks/tmp/agentic_1014/run_triton.sh`：

- 把 `$TREE/python` 複製到 container 的 `/root/ab_trees/prstack`，用 `PYTHONPATH` 蓋過 image 的 sglang。
- 把 aiter#5933 的 `GEMM-A16W16-N=288-K=4096.json` 與 aiter#5982 的 `glm53_flash_bf16_{tuned,untuned}_gemm.csv` 複製進 image 的 aiter。
- 環境變數：`SGLANG_ROCM_FUSED_AR_MHC_POST=1`、`SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`。
- DSA backend：`--dsa-prefill-backend triton --dsa-decode-backend triton`（Agentic 腳本是 `--dsa-backend triton`）。

在 1012 image 的 container 裡可以這樣確認哪些 PR 生效：server log 有 `Shared experts fusion optimization enabled.` 與 `E=289, topk=9`；decode trace 有 `smallm_p2_i512_s9`（#41871）、`allreduce_mhc_post_large_m_kernel`（#42020）、`_sparse_mla_decode_split_kernel`（Triton DSA）。
