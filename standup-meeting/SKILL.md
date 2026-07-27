---
name: standup-meeting
description: Write daily standup meeting notes summarizing yesterday's work and today's plan, grouped by ticket. Use when the user asks to write, draft, or format a standup, daily update, or status report.
---

# Standup Meeting Notes

Write concise daily standup notes following the format below.

## Format

```
<Name>, <MM/DD>
Last day:
1. <TICKET-ID> <Project/Topic>: <short task title>
<one or two lines describing what was done and the outcome>
2. <TICKET-ID> <Project/Topic>:
<short task description>
Today:
1. <TICKET-ID> <Project/Topic>:
<short task description>
<additional bullet line if needed>
```

## Rules

- Write the entire standup in English only, regardless of the language used in the conversation or source material. Translate any non-English notes into English.
- Group every item under a ticket ID (e.g. `GPUAI-6363`) followed by the project/topic.
- `Last day:` lists what was worked on previously, with a short result/outcome line per item.
- `Today:` lists what is planned next.
- Keep each entry terse: a title line plus one or two description lines. No filler.
- Number items within each section.
- If the same ticket appears in both sections, repeat its ID.

## Example

```
Jacob, 06/25
Last day:
1. GPUAI-6363 GLM 5 FP8:  Investigate the benchmark gap between IX and our local run
No a quick fix can bring InferenceX benchmark script to align with our local run. Will raise an github issue to announce their script potentially downgraded the MI355X perf
2. GPUAI-6427 GLM 5/5.1 MXFP4: 
Profile B200 for the latest status
Today:
1. GPUAI-6427 GLM 5/5.1 MXFP4: 
Profile MI355X and project optimization priority.
Add 70K/300 benchmark
```
