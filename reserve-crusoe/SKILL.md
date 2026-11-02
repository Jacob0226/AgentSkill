---
name: reserve-crusoe
description: >-
  Reserve three exclusive 8-GPU nodes on the Crusoe amd-aifw-sglang partition for
  Monday–Friday, 04:00–00:00 Asia/Taipei (20h each). Use when the user asks to
  borrow, reserve, or hold Crusoe or amd-spur nodes, says 借 node / 預約 node,
  or asks to fix a Jacob_GLM_hold job stuck in JobHoldMaxRequeue.
---

# Reserve Crusoe nodes

Standing reservation. Do not invent a different start time, duration, or node count unless the user explicitly overrides it for that request.

| | |
|---|---|
| When | Monday–Friday, 04:00 Asia/Taipei |
| Length | 20 hours, ending 00:00 the next calendar day |
| Nodes | 3 exclusive nodes, 8 GPUs each |
| Partition | `amd-aifw-sglang` |
| Account / QoS | `amd-aifw-sglang` / `amd-aifw-sglang-qos` |
| Weekend | nothing |

Cluster timestamps are UTC. 04:00 Taipei is 20:00 UTC the previous day. QoS max wall is 24 hours, so 20 hours fits. Do not pass `--requeue`.

The batch script sleeps until 00:00 Taipei of that window and then exits. `--time=20:00:00` is only the Slurm cap, counted from the actual start. A job that starts at 11:00 still releases the node at midnight, so it is not still holding a node when the next day's three become eligible at 04:00.

`amd-aifw-di` is no longer on this user. Do not submit there. This account has no per-user job or node cap. Three running holds do not cause Spur to reject the next three. Those jobs are already pending from Sunday. The sglang group node cap is 8. If it is full the jobs stay pending; they are not cancelled.

`--begin` only delays eligibility. It does not reserve a node. Priority caps at 11000, tied with the rest of the queue, so submitting earlier does not make these jobs first. Say that when a job is pending `Resources`.

## Run the script

Do not hand-write `sbatch`. From any cwd:

```bash
python3 ~/AgentSkill/reserve-crusoe/scripts/reserve_crusoe.py
```

- No dates: cover weekday windows still open through Friday of this week. On Saturday or Sunday, cover next Monday–Friday.
- Every Sunday 08:00 Asia/Taipei a user timer on `crsuse2-slog-006` runs this script with no arguments, which books the coming Monday–Friday. Units: `~/.config/systemd/user/reserve-crusoe.timer` and `.service`. Leave it alone unless the user asks to change the schedule. Do not submit that week a second time by hand after the timer has run.
- A later end date: add `--through YYYY-MM-DD` (Taipei date, inclusive).
- Preview: `--dry-run`.

The script is idempotent. Pending `Jacob_GLM_hold_*` jobs for a target day that are not a 04:00 / 20h slot are cancelled and replaced. It never cancels `RUNNING` jobs. A window that already started is filled only for the time left until midnight, not a fresh 20 hours.

Job names are `Jacob_GLM_hold_YYYYMMDD_{a,b,c}` using the Taipei date. Attach with `srun --jobid=<id> --pty bash`.

## After it runs

Report every job: id, Taipei window, state, and projected start converted to Asia/Taipei. Controller `StartTime` is UTC.

## JobHoldMaxRequeue

Spur holds a job after 5 requeues. The script releases a held job whose window is still open (`scontrol release`) and cancels one whose window has ended. Do not release a stale hold by hand; it will take a node outside the reservation.
