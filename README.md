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
| `GLM5.3-Flash` | GLM-5.3-Flash model architecture reference: hybrid KDA linear attention + MLA/DSA k-pool indexer, mHC 4-stream residual, 288-expert MoE, NoPE text tower, vision encoder, MTP |
| `PR/sglang-pr` | Write an sgl-project/sglang PR body: Summary / Scope / Test plan, 100 words per section, detail in tables |
| `PR/aiter-pr` | Write a ROCm/aiter PR body: Motivation / Test Plan / Test Result, 100 words per section, detail in tables |
| `standup-meeting` | Write daily standup notes (Last day / Today, grouped by ticket ID) |
