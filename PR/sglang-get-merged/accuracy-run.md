# Local run of `test_glm53_flash_eval_mi35x.py`

Runs the nightly GLM-5.3-Flash GSM8K test (TP8 + EP8, FP8 checkpoint, 1319 questions, threshold 0.92)
at a PR head, unchanged except for the model path. Measured at #44415's head: 0.9727, 661 s for the
test after a warm load.

## Inputs

| Item | Value |
| --- | --- |
| Tree | A worktree at the PR head, e.g. `git -C ~/PR/sglang fetch https://github.com/sgl-project/sglang.git pull/<N>/head:pr<N>-head && git -C ~/PR/sglang worktree add ~/PR/wt-pr<N> pr<N>-head` |
| Model | `/shared_nfs/jacchang/models/GLM-5.3-Flash-46a050`: weights symlinked from snapshot `3f1971b` (the HF `refs/main` snapshot `46a050b` has no weights; their index files are identical), newest `chat_template.jinja` from `46a050b` |
| Image | Newest `rocm/sgl-dev:v0.5.23-rocm10-mi35x-*` on the node. It lacks `sgl-eval`; install the version pinned in `python/pyproject.toml` (`sgl-eval==0.1.3 antlr4-python3-runtime==4.9.3`) |
| Node | Any idle 8-GPU node in the user's hold job (`reserve-crusoe` skill) |

## Steps

```bash
export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-006.crusoe.amd.com:6817
J=<jobid>; N=<PR number>; OUT=/shared_nfs/jacchang/pr${N}_eval; mkdir -p $OUT
cp ~/.cursor/skills/sglang-get-merged/run_glm53_eval.py $OUT/

timeout 120 srun --jobid $J --overlap bash -c "
  docker run -d --name jacchang_pr$N --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --network=host --shm-size 64g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v /home/jacchang:/home/jacchang -v /shared_nfs:/shared_nfs <image> sleep infinity &&
  docker exec jacchang_pr$N pip install -q sgl-eval==0.1.3 antlr4-python3-runtime==4.9.3 &&
  docker exec -d -e PYTHONPATH=$HOME/PR/wt-pr$N/python -e TREE=$HOME/PR/wt-pr$N \
    -e GITHUB_STEP_SUMMARY=$OUT/summary.md jacchang_pr$N \
    bash -c 'python3 $OUT/run_glm53_eval.py > $OUT/run.log 2>&1'
" </dev/null
```

Then poll `$OUT/run.log` for `gsm8k_score=` and `OK` / `FAILED`; `$OUT/summary.md` holds the result table.

## Pitfalls

| Trap | Avoid |
| --- | --- |
| `srun ... bash -s <<EOF` (script on stdin) | Leaves a step waiting for input forever. spur allows one interactive session per job per node and cannot cancel a single step, so the node is locked for that job until it ends. Always `bash -c '...' </dev/null` |
| `HF_HUB_OFFLINE=1` | sgl-eval downloads the GSM8K dataset; leave it unset. The model is a local path anyway |
| Nodes reachable by ssh | They are not (publickey denied); everything goes through `srun --jobid --overlap` |
| Kill a stuck server | `sglang-benchmark` skill, `kill_sglang.sh` |

Report the sha, the image tag, the score as `0.xxxx (n/1319)`, and the 0.92 threshold.
