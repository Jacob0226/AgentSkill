# PR stack（MI355X 重跑 benchmark 前必讀）

最後更新：2026-11-07（台灣時間）。「狀態」與「能不能套」是當天查的，重跑前先照最後的「重跑前檢查」更新一次。

MI355X 的數字來自四層：**image、自己疊的 sglang PR stack、monokernel（sgl-amd-mono）版本、手動步驟（env、CSV、server flag）**。漏掉任何一層，結果就跟之前的報告對不上。**把 PR 套進 tree 不等於它會生效**：很多 PR 要 env、server flag 或特定條件才會跑到，見「每個 PR 怎麼啟用」。B200 用 stock image，沒有 PR stack。

## 目前基準：1106 image

| 項目 | 值 |
|---|---|
| Image | `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261106`（digest `sha256:35750c54…`，11/07 在 node 150 pull） |
| image 內 sglang | `c06e39c6ca`（11/06 的 upstream main） |
| image 內 aiter | `285579b62`，**跟 1026 image 一樣，沒有更新** |
| image 內 sgl-amd-mono | `/sgl-workspace/sgl-amd-mono`，editable install，`e52803f`（11/04） |
| 其他 | FlyDSL 0.4.0、torch 2.11.0+rocm10.0.0、triton 3.8.0；image 預設 `HSA_NO_SCRATCH_RECLAIM=1`、`SGLANG_USE_AITER=1` |
| sglang stack | `~/PR/wt-glm53-1106-prstack`，branch `glm53-prstack-c06e39c6ca`，HEAD `09111fe9a3`：`c06e39c6ca` 上 10 個 commit（一個 PR 一個，全部是 11/07 的最新 head），見「1106 stack 的解衝突紀錄」 |
| mono | `/shared_nfs/jacchang/exp/sgl-amd-mono-main1107`（upstream main `25723f1`），見「Monokernel」 |
| 驗證 | `/shared_nfs/jacchang/prstack_1107/`（node 150）：BF16 KV、Triton DSA、BCG、MoE CSV、兩個 env、MTP5。GPU 0–3 用 `run_gsm.sh`（HiCache 216 GB，不開 monokernel）；GPU 4–7 用 `run_gsm_mono.sh`（不開 HiCache，開 monokernel） |

11/07 驗證結果（sgl-eval GSM8K，1319 threads、thinking、temperature 1.0、top-p 0.95；1026 stack 是 96.44%）：

| lane | sglang | GSM8K | stop rate | 99,840 prefix + 16 / 512 / 4096 new tokens 的 prefill |
|---|---|---|---|---|
| stack，BF16 KV（HiCache） | `94f7a7ece1`（沒有 #48806） | 96.66% | 99.55% | 38.8 / 52.9 / 137.4 ms |
| stack + monokernel，BF16 KV（無 HiCache） | `94f7a7ece1` | 97.19% | 99.85% | 37.4 / 51.9 / 135.2 ms |
| stack，**FP8 KV**（HiCache，即 recipe 設定） | `09111fe9a3`（含 #48806） | 97.12% | 99.24% | 37.9 / 52.3 / 133.3 ms |

FP8 KV 的 device KV 是 23,792,128 tokens，BF16 是 12,267,264（同樣 132.84 GB）。

1319 threads 下 decode batch 大多超過 2 個 request，monokernel 只在尾段接手，所以這個 GSM8K 主要驗證的是開 monokernel 時 stack 沒壞。

查法：`docker run --rm --entrypoint bash <image> -c 'pip list | grep -iE "sglang|aiter|amd-mono|flydsl"; git -C /sgl-workspace/sgl-amd-mono log -1'`（sglang 和 aiter 的 commit 在 pip 版本字串的 `+g<commit>` 裡）。判斷某個 merged PR 在不在 image：在本地 repo 跑 `git merge-base --is-ancestor <PR merge commit> <image commit>`。

## Merged PR：在 1106 image 裡嗎？

| PR | 內容 | 在 1106 image？ | 要做什麼 |
|---|---|---|---|
| sglang #41870 | Quark MXFP4 shared expert fusion + KDA projection fusion（ITL −9.8 ~ −13.6%） | 有 | 不用 |
| sglang #41161 | shared expert 併進 AITER MoE（gfx950） | 有 | 不用 |
| sglang #46036 | mono-kernels 依 model type 註冊 spec（monokernel 需要） | 有 | 不用；1026 的 stack 要自己疊，現在不用 |
| sglang #46980 | recurrent state ownership 重構（mono #71 需要） | 有 | 不用 |
| aiter #5933 | `gemm_a16w16` N=288 K=4096 config（TPOT −9%） | 有 | 不用 |
| aiter #5982 | GLM-5.3-Flash TP4 fused KDA in-proj BF16 GEMM tuning（TPOT −1.2%） | 有 | 不用 |
| aiter #6276 | gfx950 `sparse_mla` bf16 KV-tile store 去 bank conflict | 有 | 不用（只影響 `aiter_sparse_mla`） |
| aiter #6435 | GLM-5.3-Flash MXFP4 fused-MoE tuned CSV（別人的 PR） | **沒有** | 見「手動步驟」 |
| sgl-amd-mono #97 | sparse attention 超過 262,144 token 的 fault 修正 | 有 | 不用 |
| sgl-amd-mono #71 | MTP verify step 跑在 monokernel 上 | **沒有** | 見「Monokernel」 |
| sgl-amd-mono #98 | verify step 最多 16 rows | **沒有** | 同上 |
| sgl-amd-mono #100 | verify request 的 index tile 只算一次 | **沒有** | 同上 |

## Open PR：每個 PR 怎麼啟用

「怎麼啟用」是讀 PR diff 得出的：env 的預設值在 `environ.py`，flag 在 server args，其他是程式裡的條件。效果數字是各 PR 自己報的。

| PR | 內容（效果） | 怎麼啟用 | 生效怎麼看 |
|---|---|---|---|
| sglang #41615 | Triton sparse MLA（TTFT −10%） | **server flag**：`--dsa-prefill-backend triton --dsa-decode-backend triton`（driver 是 `--dsa-backend triton`；recipe 預設是 `tilelang`）。只有 #41615 時**只接受 BF16 KV**（FP8 KV 啟動就 `ValueError: Triton with index_kpool > 1 is only validated ... BF16 Q/KV`）；FP8 KV 要再疊 #48806 | decode trace 有 `_sparse_mla_decode_split_kernel` |
| sglang #48806 | Triton sparse MLA 接受 FP8 E4M3 KV（KV 容量 ×2；Agentic conc 64 HiCache write_back：tok/s +7.0%、P90 +8.4%）；取代內部的 `f0a7a579d5` | **server flag**：`--kv-cache-dtype fp8_e4m3` 加 #41615 的 Triton DSA flag。依賴 #41615，PR branch 帶著 #41615 的 commit，所以**只疊它自己的 `0ef818ced9` + `3d10677084`**（`gh pr diff` 會把 #41615 再疊一次）。PR 建議沒開 HiCache 時維持 BF16 KV；開 monokernel 時也只能 BF16 KV | 啟動沒有 `ValueError`；log 的 `kv_cache_dtype='fp8_e4m3'`，KV token 數約是 BF16 的 2 倍 |
| sglang #41871 | small-M MoE + sort 融合 MXFP4 quant（conc 4 ITL −4.6%） | 自動：Quark MXFP4、TP4 shape、每步 ≤16 token、`--moe-runner-backend aiter` | decode trace 有 `smallm_p2_i512_s9` |
| sglang #44415 | MoE correction bias 保持 fp32（bug fix） | 自動：`model_type == glm5_next_text` | — |
| sglang #44416 | one-launch MoE router + 較窄 KDA V tile（P90 +5.8 ~ 6.5%）；**已包含 #44415 的 commit，疊它就不用再疊 #44415** | 自動：gfx950、每步 ≤16 token、`SGLANG_USE_AITER=1`（image 預設）；`SGLANG_SIMULATE_UNIFORM_EXPERTS` / `ROUND_ROBIN_EXPERTS` 或 expert remapping 開著時退回原路徑 | decode trace 有 smallm router kernel，沒有獨立的 gate GEMM + topk |
| sglang #44417 | MTP verify 的 k-pool indexer launch 變少（P90 +8%） | 自動：HIP、MTP verify | — |
| sglang #46808 | sort-free top-p + 多 row softmax（P90 +4%） | 自動：HIP 的 verify sampling；request 要帶 top_p / temperature 才跑到（greedy 不會） | — |
| sglang #48875 | BCG replay + graph 內 HIP KDA extend（P90 +9%） | **server flag**：`--cuda-graph-backend-prefill breakable`，建議加 `--cuda-graph-max-bs-prefill 2560`（recipe 已設） | 99,840 prefix + 16 new tokens 的 prefill 約 37 ms（沒 replay 約 99 ms） |
| sglang #38764 | KDA projection PTPC FP8（TTFT −2.2%） | **env**：`SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj`（預設空，沒設等於沒開） | KDA projection 在 trace 裡是 FP8 GEMM |
| sglang #44659 | row-streaming KDA packed decode（ITL −0.53%） | 自動：gfx950。**只在 `forward_decode`（每個 request 1 token）**；MTP 時 target 走 verify，依程式判斷 MTP benchmark 跑不到它 | 不開 MTP 的 decode trace 有 `kda_packed_decode_rowstream_kernel` |
| sglang #44661 | KPool Hadamard + FP8 quant 融合（ITL −0.43%） | **env**：`SGLANG_DSA_FUSE_HADAMARD_QUANT=1`（預設 False）；另需 gfx950、indexer head_dim = block_size 且為 2 的次方 | trace 少了獨立的 Hadamard kernel |
### 從 stack 拿掉的（11/07 使用者決定）

| 項目 | 原本的作用 | 拿掉的影響 |
|---|---|---|
| sglang #42020（stale bot 10/25 關閉） | FFN all-reduce + `hc_post` 融合，env `SGLANG_ROCM_FUSED_AR_MHC_POST=1` | 這個 env 在新 stack 沒有作用，不用再設 |
| `f0a7a579d5`（內部 commit，沒有 PR） | 讓 #41615 的 Triton sparse MLA 接受 FP8 e4m3 KV | 被 #48806 取代，不要再疊 |

**FP8 KV。** 有 #48806 之後，recipe 寫死的 `--kv-cache-dtype fp8_e4m3` 配 driver 的 `--dsa-backend triton` 可以直接跑。要 BF16 KV 時（例如開 monokernel，或沒開 HiCache 的 conc），從 driver 多傳 `--server-arg --kv-cache-dtype --server-arg bfloat16`（額外參數接在 recipe 後面，後出現的生效）。

### 1106 stack 的解衝突紀錄

11/07 用 `tools/build_glm53_prstack.sh` 在 image 的 `c06e39c6ca` 上依序疊。#41871、#46808、#44659、#48875、#48806 直接套上；其餘的衝突這樣解（重建時照做）。疊完 #48806 後，`triton_sparse_mla.py` 跟 #48806 head 的版本只差 #41615 解衝突留下的縮排，直接 `git checkout pr/48806 -- python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py`（#48806 作者在自己的 merge `182ffa5f6f` 裡也是這樣解 #44503 和 #44322 的衝突）。

| PR | 衝突 | 怎麼解 |
|---|---|---|
| #41615 | `triton_sparse_mla.py`：upstream #44503 把 KV 載入改成 `_load_kv_group` / `_load_kv_tail`（多了 `KV_SCALED`）；PR 要讓 tail 只在 `D_TAIL > 0` 才載入 | 保留 upstream 的 helper，把 `_load_kv_tail` 包進 `if D_TAIL > 0:`（三個 kernel 各一處） |
| #41615 | `dsa_backend.py`：upstream #44322 在 `dsa_impl == "triton"` 裡加了 DCP；PR 改成 `in ("triton", "aiter_sparse_mla")` 並先試 aiter | 兩者都留：條件用 PR 的，順序是 DCP → aiter → Triton（prefill 和 decode 各一處）；`_forward_triton_dcp` 和 `_forward_aiter_sparse_mla` 兩個 method 都留 |
| #38764 | `fused_norm_gate.py` import（upstream #47782 改用 `current_arch`）、`environ.py` 相鄰新增 | import 用 upstream 的再加 PR 的 `fp8_dtype_to_triton`；env 兩個都留 |
| #44416 | `smallm_router_gfx950/__init__.py`：PR 已 merge 過 #48399（拿掉 `SGLANG_ROCM_SMALLM_ROUTER`、改用 `current_arch`），image 還有這個 env | 用 PR 的版本，補 `from sglang.kernels.arch import current_arch`，`smallm_router_supported` 保留 `envs.SGLANG_ROCM_SMALLM_ROUTER.get()` |
| #44417 | `dsa_metadata_manager.py`：同樣是 #48399 拿掉 `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION`，image 還有（預設 True） | `requested and supported and (is_cuda() or _is_hip)`，跟 1026 stack 一樣 |
| #44661 | `dsa_indexer_kpool.py`：import、新 method 相鄰、`_project_and_scale_head_gates` 被 #44417 改名 | import 用 #44417 後的；兩邊的新 method 都留；呼叫改 `_project_head_gates` |
| #44661 | **語意衝突，git 不會標**：#44417 把 `n_heads**-0.5` 從 `_project_head_gates` 移到 `_apply_q_scale_and_softmax_scale`，但 #44661 的 `fused_hadamard_act_quant(weights=head_weights, softmax_scale=...)` 假設 `head_weights` 已經乘過 | `_quant_query` 傳 `softmax_scale=self.softmax_scale * self.n_heads**-0.5`；漏了會讓 `SGLANG_DSA_FUSE_HADAMARD_QUANT=1` 時 indexer 分數差一個常數倍 |

拿 11/07 main `b750923a2a` 當 base 時，#44416、#44417 可以直接套，但 #44659 會在 `kda_triton.py` 衝突，其他幾個照樣衝突。

## 建 stack

```bash
# 預設依序疊：41615 41871 38764 44416 44417 46808 44659 44661 48875 48806:0ef818ced9,3d10677084
~/SGLang-benchmarks/tools/build_glm53_prstack.sh <image sglang commit> <worktree> [entry...]
```

- **永遠用 PR 的最新 head。** 重建時一律 fetch 當下的 head，不要沿用舊 stack 的 commit；head 變了就重建、重解衝突、重跑 GSM8K。
- 每個 PR 用 `gh pr diff`（PR 自己的改動），不會把 PR branch merge 進來的 main commit 帶進來。PR branch 帶著另一個 PR 的 commit 時（#48806 帶著 #41615），用 `<PR>:<commit>,<commit>` 只疊它自己的 commit；PR 有新的 commit 時，要把它自己的新 commit 加進清單（merge commit 不用，衝突自己解）。`c:<commit>` 是只在內部 tree 的 commit。
- 第一個套不上的 PR 會停下來，衝突留在 worktree 裡。解完 `git add` + commit 後，用剩下的 entry 再跑一次。
- base 用 image 的 sglang commit，跟 image 的 aiter、sglang-kernel 對得上。用更新的 main 衝突會變少，但要確認 image 的 aiter 撐得住。
- 解完衝突的 stack 要先過 GSM8K，再跑 benchmark。**1106 image 沒有 `sgl-eval`**，在 container 裡裝到獨立 venv（不要裝進 `/opt/venv`，會改到 openai 等套件）：`python3 -m venv /root/sgl_eval_venv && /root/sgl_eval_venv/bin/pip install "git+https://github.com/sgl-project/sgl-eval.git@a231b7a439b235090ff7baa30778fa2b514309ae"`，再 `cp -r /home/jacchang/.cache/sgl_eval /root/.cache/`，用 `/root/sgl_eval_venv/bin/sgl-eval` 的完整路徑呼叫（不要把 venv 的 `bin` 放進 `PATH`，會蓋掉 server 的 `python3`）。
- 用 `PYTHONPATH=<worktree>/python` 蓋過 image 的 sglang；baseline 和 A/B 的兩邊都要用同一個 `PYTHONPATH`。

## Monokernel（sgl-amd-mono）

1106 image 的 sgl-amd-mono 是 11/04 的 `e52803f`，沒有 #71、#98、#100，**也就是沒有 MTP verify**。我們用 MTP5，所以用 image 原本的 mono，monokernel 一個 step 都接不到。更新方式：

1. 用固定在 upstream main 的 worktree：`/shared_nfs/jacchang/exp/sgl-amd-mono-main1107`（`25723f1`，含 #71、#97、#98、#100）。要更新時：`git -C /shared_nfs/jacchang/exp/sgl-amd-mono-iscore fetch https://github.com/sgl-project/sgl-amd-mono.git main:refs/remotes/upstream/main`，再開新的 worktree。
2. 放在 `PYTHONPATH` 最前面：`PYTHONPATH=$MONO:$SGL/python`。image 的 mono 是 editable install，plugin entry point（`sglang.srt.plugins`）用模組名稱載入，所以會載到 `PYTHONPATH` 的版本；`pyproject.toml` 從 `e52803f` 到 `25723f1` 沒變，不用重裝。
3. sglang 那邊要有 #46036 和 #46980（1106 image 都有，用 `c06e39c6ca` 當 base 的 stack 也有）。
4. 啟動加 `--enable-amd-monokernel`。每條 lane 設各自的 `FLYDSL_RUNTIME_CACHE_DIR`。
5. container 裡確認：`python3 -c "import sgl_amd_mono; print(sgl_amd_mono.__file__)"` 要指到 worktree。

monokernel 只在下面條件都成立時接手，否則 server log 會印 `monokernel: <原因>: the normal forward`，整個 step 走 SGLang 原本的 forward：

| 條件 | 值 | 跟目前 Agentic recipe 的關係 |
|---|---|---|
| `--kv-cache-dtype` | `auto` 或 `bfloat16` | **recipe 是 `fp8_e4m3`，monokernel 會被關掉**；開 monokernel 時要改回 BF16 KV |
| KV offload（HiCache） | **不接受**：開 `--enable-hierarchical-cache` 時啟動就 `ValueError: --enable-amd-monokernel cannot run this setup: KV offload is in use and the spec does not accept it`（11/07 實測） | **recipe 在 conc 64 開 HiCache 216 GB，所以 conc 64 不能開 monokernel**；conc ≤32 沒開 HiCache，可以 |
| 每個 step 的 rows | ≤16（`25723f1` 的寬度是 1–16 每一個）；TP2 ≤4 | MTP5 一個 request 6 rows，**最多 2 個 request**（log：`runs target_verify steps of batch size [1, 2] x 6 tokens (2 of 29 captured decode graphs)`）；MTP3 最多 4 個。conc 64 的 decode batch 幾乎都超過，只有低 concurrency 用得到 |
| MTP 設定 | `speculative_eagle_topk 1`，draft tokens = steps + 1 | recipe 符合 |
| `--moe-runner-backend` | `auto` 或 `aiter` | 符合 |
| env | `SGLANG_SIMULATE_UNIFORM_EXPERTS`、`SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS`、`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK`、`SGLANG_ENABLE_PP_SPEC` 都要關 | recipe 的 `SGLANG_SIMULATE_ACC_*` 不在這個清單，沒關係 |
| 其他 | 不開 HiSparse、linear ReplaySSM；KDA verify backend 不是 FlashInfer；page size 256 或預設；Quark checkpoint；有 FlyDSL | 符合 |

有生效時，server log 會印 `monokernel glm53_flash_hip on gfx950 runs ... (N of M captured decode graphs)`。

## 手動步驟（recipe 沒做的）

**MoE tuned CSV。** 1106 image 的 aiter 還是 `285579b62`，沒有 #6435。recipe 沒有設 `AITER_CONFIG_FMOE`，忘了傳就是沒有 tuned CSV（30 分鐘 run 的 TPUT 和 P90 分別差約 1.8% 和 4.8%）。兩個 CSV 在 60 分鐘 run 的差異在 noise 內：

| 選項 | CSV | 額外步驟 |
|---|---|---|
| 我們 tune 的 | `~/SGLang-benchmarks/tmp/moe2_tune/glm53_flash_fp4_tuned_fmoe.csv` | 無 |
| aiter #6435 | `/shared_nfs/jacchang/aiter6435_1106/glm53flash_fp4_tuned_fmoe.csv` | container 裡先跑 `/shared_nfs/jacchang/aiter6435_1106/patch_aiter.sh`（加 6 個 shape、rebuild aux module） |

傳法：`AITER_CONFIG_FMOE` = image 預設清單（`/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv` 加上 `model_configs/` 下所有 `*fmoe*.csv`，用 `:` 串起來）再接上選的 CSV，當 `lane_env_dur.sh` 的第 5 個參數（`KEY=VALUE`）傳進去。完整字串見 `/shared_nfs/jacchang/agentic_1106/driver.log`。

**env 一覽**（沒設就等於 PR 沒開）：

| env | PR | 值 |
|---|---|---|
| `SGLANG_OPT_GLM53_KDA_PTPC_MODULES` | #38764 | `qkv_proj,f_a_proj,g_a_proj,o_proj` |
| `SGLANG_DSA_FUSE_HADAMARD_QUANT` | #44661 | `1` |
| `AITER_CONFIG_FMOE` | MoE CSV | 見上 |
| `HSA_NO_SCRATCH_RECLAIM` | — | 維持 image 預設 `1`，不要改 |

**server flag / driver 參數。** `--conc 64 --mtp-steps 5 --dsa-backend triton --mamba-ratio 0.5 --hicache-size 216 --hicache-io kernel`。BCG 的 `--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2560`、INT4 quick-reduce 已經寫在 recipe 裡。recipe 的 `--kv-cache-dtype fp8_e4m3` 靠 #48806 才能配 Triton DSA（見「FP8 KV」）。monokernel 的 `--enable-amd-monokernel` 要另外加，而且要配 BF16 KV、不開 HiCache。

## B200 對照

B200 用 stock image，不疊 PR。60 分鐘的對照點是 `lmsysorg/sglang:v0.5.23-cu130`、dgx-023、10/30 單獨跑（InferenceX curve 那條線）。30 分鐘和 60 分鐘的 run 不能互比，MI355X 和 B200 要用同樣長度比。

## 重跑前檢查

1. 更新 PR 狀態，跟上面的表比對：

   ```bash
   GH="env -u GH_TOKEN -u GITHUB_TOKEN /home/jacchang/bin/gh"
   for p in 41615 41871 38764 44415 44416 44417 46808 48875 44659 44661 48806; do
     $GH pr view $p --repo sgl-project/sglang --json number,state,headRefOid \
       --jq '"#\(.number) \(.state) \(.headRefOid[0:10])"'
   done
   ```

2. 換 image 時：查 image 內的 sglang、aiter、sgl-amd-mono commit，用 `git merge-base --is-ancestor` 判斷 merged PR 在不在；在的就從 stack 拿掉，mono 夠新就不用 `PYTHONPATH` 蓋。
3. 用 `tools/build_glm53_prstack.sh` 以**最新 head** 重建 stack（不用問）；#48806 這類只疊自己 commit 的 PR，先看 PR 有沒有新的非 merge commit。有衝突先問使用者要不要解、怎麼解。
4. 對照「每個 PR 怎麼啟用」：env 有沒有設、flag 有沒有加、條件有沒有成立（例如 FP8 KV 和 monokernel 互斥）。
5. 開跑後確認生效訊號，至少看 server log 的 `E=289, topk=9`、monokernel 的那行 log 和一份 decode trace。
6. 跑完更新這份文件的「最後更新」和表格。

## 舊 image 的 PR stack（歷史紀錄）

### 1026 image（11/06 Agentic 數字用的這套）

Image `rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261026`，image 內 sglang `5ef99013ac`、aiter `285579b62`。sglang tree `/shared_nfs/jacchang/exp/sglang-fp8kv-1103`，branch `glm53-1105-bcg-hybrid-dsa`，HEAD `b1c1c92cdc`（`5ef99013ac` 上 14 個 commit）。mono tree `/shared_nfs/jacchang/exp/sgl-amd-mono-iscore`（branch `glm53-flash-dsa-iscore-req`）放在 `PYTHONPATH` 最前面，但 Agentic recipe 沒開 monokernel。啟動鏈：`/shared_nfs/jacchang/agentic_1106/lane_env_dur.sh`（`DUR` 秒數，預設 1800）→ `ix_agentx_glm53flash.sh --platform mi355x` → `recipe_glm53flash_fp4_mi355x_sglang_mtp.sh`。

stack 裡是當時的 PR head（套在 `2815eed49c` 上再搬到 `5ef99013ac`），依序：

| PR | stack 用的 head | 11/07 的 head |
|---|---|---|
| #41615 | `acca6a72db` | 相同 |
| #41871 | `ac5057efad` | `2f3232f34d` |
| #42020 | `bd5ddf3358`，改接 `ExitDecision(sum_in_reduce_scatter)` API | 相同（closed） |
| #38764 | `1d33eed202` | `c8c963cf0f` |
| #44415 | `4272d975f1` | `2e8c7f9b39` |
| #44416 | `9b1d711bba` | `3d692f0a25` |
| #44417 | `60336705fe` | `8499d23086` |
| #46036 | `f73a82b40e` | merged |
| #46808 | 3 個 commit `2514ded128`、`d41967ade8`、`42665af0e0` | 內容相同 |
| 沒有 PR | `f0a7a579d5` FP8 KV on Triton sparse MLA | — |
| #48875 | `e4cd94409d`、`b1c1c92cdc` | `c9670de08c`（同內容 rebase 到 main） |

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

PR stack 在 `~/PR/wt-glm53-1012-prstack`（branch `glm53-1012-prstack`，依序疊 #41615、#41871、#42020、#38764）。啟動方式見 `~/SGLang-benchmarks/tmp/agentic_1014/run_triton.sh`：把 `$TREE/python` 複製到 container 的 `/root/ab_trees/prstack`，用 `PYTHONPATH` 蓋過 image 的 sglang；把 aiter#5933 的 `GEMM-A16W16-N=288-K=4096.json` 與 aiter#5982 的 `glm53_flash_bf16_{tuned,untuned}_gemm.csv` 複製進 image 的 aiter；env 同上。
