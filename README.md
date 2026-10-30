# cursor-skills

Portable [Cursor](https://cursor.com) Agent Skills. Clone on any machine and run setup to install.

## Setup

```bash
git clone git@github.com:Jacob0226/cursor-skills.git ~/cursor-skills
cd ~/cursor-skills
./setup.sh
```

This creates symlinks from `~/.cursor/skills/<skill-name>` to the corresponding directory in this repo. Skills may sit at the top level or one directory down, so related ones can be grouped (`PR/sglang-pr`, `PR/aiter-pr`); the group directory itself is not linked, only the leaves.

## Skills

| Skill | Description |
|-------|-------------|
| `sglang-benchmark` | SGLang dev workflow, benchmarking & profiling on MI355X / B200 |
| `sglang-glm5-fp8-decode-timeline` | Decode layer dual-stream kernel timeline (MI355X vs B200, MLA+NSA+MoE models) |
| `sglang-ci-repro` | Reproduce SGLang AMD CI failures locally on MI35x / MI325 boxes via `~/SGLang-benchmarks/tools/repro_ci.sh` (workflow → stage → suite → file mapping, partition handling, multimodal-gen edge case, aiter-version A/B) |
| `glm53-flash-architecture` | GLM-5.3-Flash model architecture reference: hybrid KDA linear attention + MLA/DSA k-pool indexer, mHC 4-stream residual, 288-expert MoE, NoPE text tower, vision encoder, MTP |
| `PR/sglang-pr` | Write an sgl-project/sglang PR body: Summary / Scope / Test plan, 100 words per section, detail in tables |
| `PR/sglang-pr-review` | Review an SGLang PR for OOP, engineering, and repo conventions (AMD gates, kernel tests). Draft locally; do not post until approved |
| `PR/aiter-pr` | Write a ROCm/aiter PR body: Motivation / Test Plan / Test Result, 100 words per section, detail in tables |
| `PR/sgl-amd-mono-pr` | Write and open an sgl-project/sgl-amd-mono PR: the repo template (Motivation / Modifications / Accuracy Tests / Speed Tests and Profiling / Checklist), 100 words per section, title check, fork + lint workflow |
| `monokernel-dev` | Build or extend an sgl-amd-mono monokernel: scoping, precedents, the verification gate ladder (layer correctness, long context, microbench, GSM8K, AgentX), profiling, pitfalls; plus a GLM-5.3-Flash MTP case study and a guide for directing the agent. Tested only with Claude Opus 5.5 (High) |
| `standup-meeting` | Write daily standup notes (Last day / Today, grouped by ticket ID) |
| `reserve-crusoe` | Reserve 3 exclusive 8-GPU nodes on Crusoe amd-spur, Mon–Fri 04:00–00:00 Asia/Taipei (20h) |
