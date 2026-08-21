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
  - <work item with its measured outcome, 30 words max>
  - <second work item or finding for the same ticket>
Today:
- <TICKET-ID> <Project/Topic>
  - <planned item>
  - <planned item>
```

Two levels are in play:

- Level 1 bullet: `<TICKET-ID> <Project/Topic>` only. No description on this line.
- Level 2 bullet: one per work item, finding, or plan item. This is where all the content lives.

A continuation line (a soft line break, Shift+Enter in Teams, so it stays inside the bullet above instead of starting a new one) is still available, but the 30-word cap counts the bullet and its continuation together, so it is rarely needed. Prefer a second bullet.

## Rules

- Write the entire standup in English only, regardless of the language used in the conversation or source material. Translate any non-English notes into English.
- Header is `<Name>, <MM/DD>`, then the `Last day:` and `Today:` sections. Both section labels are plain text lines, not bullets.
- Every item hangs off a ticket ID (e.g. `GPUAI-6521`). If the same ticket appears in both sections, repeat it as a level 1 bullet in each.
- Use bullets, never numbered lists. Every piece of work content is a bullet item; never write a work item as a bare paragraph.
- **30-word cap.** No work item may exceed 30 words, counting any continuation line attached to it. Split anything longer into sibling bullets instead of letting one item run on. The cap applies to `Last day:` and `Today:` alike.
- `Last day:` reports what was done plus the concrete result. Include the numbers that matter (latency, throughput, accuracy, before -> after) rather than saying "improved".
- `Today:` lists what is planned next, one short line per item, no results.
- Keep it terse: no filler, no restating the ticket title. Spend the 30 words on the numbers, not on setup or narration.
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
  - FP8 online quant for MLA projections: o_proj 41.4 -> 21.0ms, dense GEMM bucket 187.7 -> 153.4ms, i8k/conc64 TTFT 665 -> 628ms
  - GSM8K holds at 0.931 vs 0.933 baseline. q_b_proj excluded, break-even at 15.4ms either way
  - Hits the ATOM dense target, 138.9 vs 135ms, and cuts GPU-busy 897 -> 862ms
  - But 562 extra quant kernels serialize the stream: EXTEND wall 712 -> 722ms, TTFT 674ms, worse than baseline
Today:
- GPUAI-6521 GLM-5.2 MXFP4
  - Fuse quant into the preceding rmsnorm, like fused_rms_fp8_group_quant, to remove the quant overhead
  - Look into rccl/allreduce, inspired by ATOM
```

Each bullet above is under 30 words, and the one long paragraph from the old format is now four siblings.
