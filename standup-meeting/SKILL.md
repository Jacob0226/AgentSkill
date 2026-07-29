---
name: standup-meeting
description: Write daily standup meeting notes summarizing yesterday's work and today's plan, grouped by ticket. Use when the user asks to write, draft, or format a standup, daily update, or status report.
---

# Standup Meeting Notes

Write concise daily standup notes as a nested bullet list that pastes cleanly into Microsoft Teams.

## Format

```
<Name>, <MM/DD>
Last day:
- <TICKET-ID> <Project/Topic>
  - <short task title>
    <one or two continuation lines with details and measured outcome>
  - <second finding or outcome for the same ticket>
Today:
- <TICKET-ID> <Project/Topic>
  - <planned item>
  - <planned item>
```

Three levels are in play:

- Level 1 bullet: `<TICKET-ID> <Project/Topic>` only. No description on this line.
- Level 2 bullet: one per task, finding, or plan item.
- Continuation line (no bullet): details belonging to the level 2 bullet above it. In Teams this is a soft line break (Shift+Enter), so it stays inside the same bullet instead of starting a new one.

## Rules

- Write the entire standup in English only, regardless of the language used in the conversation or source material. Translate any non-English notes into English.
- Header is `<Name>, <MM/DD>`, then the `Last day:` and `Today:` sections. Both section labels are plain text lines, not bullets.
- Every item hangs off a ticket ID (e.g. `GPUAI-6521`). If the same ticket appears in both sections, repeat it as a level 1 bullet in each.
- Use bullets, never numbered lists.
- `Last day:` reports what was done plus the concrete result. Include the numbers that matter (latency, throughput, accuracy, before -> after) rather than saying "improved".
- `Today:` lists what is planned next, one short line per item, no results.
- Keep it terse: a level 2 bullet plus at most two continuation lines. No filler, no restating the ticket title.
- Group several findings on one ticket as sibling level 2 bullets rather than one long paragraph.

## Teams formatting

- Paste as plain text with `-` markers and two-space indent per level; Teams auto-converts `-` into bullets and indentation into nesting. Alternatively type it in Teams using Tab to indent and Shift+Tab to outdent.
- Ticket IDs are usually hyperlinked to Jira. Keep the link on the ID text only, not the whole line.
- Use `->` for before/after transitions; Teams may render `→`, which is fine.

## Example

```
Jacob, 07/29
Last day:
- GPUAI-6521 GLM-5.2 MXFP4
  - FP8 online quant for MLA projections
    Route Quark-excluded attention linears to Fp8LinearMethod. o_proj: 41.4ms bf16 -> 21.0ms fp8, dense GEMM bucket 187.7 -> 153.4ms, i8k/conc64 TTFT 665 -> 628ms, GSM8K 0.931 vs 0.933 baseline. q_b_proj is break-even (15.4ms both ways, quant kernel eats the GEMM win) so it is excluded
  - Hits the ATOM dense target (138.9ms vs ATOM 135ms) and cuts GPU-busy 897 -> 862ms, but the 562 extra quant kernels serialize the stream: EXTEND wall goes 712 -> 722ms, TTFT 674ms (worse than the 665ms baseline)
Today:
- GPUAI-6521 GLM-5.2 MXFP4
  - To remove the quant overhead, fuse quant and the proceeding rmsnorm like fused_rms_fp8_group_quant / _batched_gemm_a8w8_prequant
  - Look into rccl/allreduce inspired by ATOM
```
