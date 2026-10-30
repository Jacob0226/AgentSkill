---
name: reserve-crusoe
description: >-
  Reserve three exclusive 8-GPU nodes on the Crusoe amd-spur cluster for
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
| Partition | `amd-spur` |
| Account / QoS | `amd-aifw-di` / `amd-aifw-di-qos` |
| Weekend | nothing |

Cluster timestamps are UTC. 04:00 Taipei is 20:00 UTC the previous day. QoS max wall is 24 hours, so 20 hours fits. Do not pass `--requeue`.

`--begin` only delays eligibility. It does not reserve a node. Priority caps at 11000, tied with the rest of the queue, so submitting earlier does not make these jobs first. Say that when a job is pending `Resources`.

## Run the script

Do not hand-write `sbatch`. From any cwd:

```bash
python3 ~/AgentSkill/reserve-crusoe/scripts/reserve_crusoe.py
```

- No dates: cover weekday windows still open through Friday of this week. On Saturday or Sunday, cover next Monday–Friday.
- A later end date: add `--through YYYY-MM-DD` (Taipei date, inclusive).
- Preview: `--dry-run`.

The script is idempotent. Pending `Jacob_GLM_hold_*` jobs for a target day that are not a 04:00 / 20h slot are cancelled and replaced. It never cancels `RUNNING` jobs. A window that already started is filled only for the time left until midnight, not a fresh 20 hours.

Job names are `Jacob_GLM_hold_YYYYMMDD_{a,b,c}` using the Taipei date. Attach with `srun --jobid=<id> --pty bash`.

## After it runs

Report every job: id, Taipei window, state, and projected start converted to Asia/Taipei. Controller `StartTime` is UTC.

## JobHoldMaxRequeue

Spur holds a job after 5 requeues. The script releases a held job whose window is still open (`scontrol release`) and cancels one whose window has ended. Do not release a stale hold by hand; it will take a node outside the reservation.
