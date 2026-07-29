# cursor-skills

Portable [Cursor](https://cursor.com) Agent Skills. Clone on any machine and run setup to install.

## Setup

```bash
git clone git@github.com:Jacob0226/cursor-skills.git ~/cursor-skills
cd ~/cursor-skills
./setup.sh
```

This creates symlinks from `~/.cursor/skills/<skill-name>` to the corresponding directory in this repo.

## Skills

| Skill | Description |
|-------|-------------|
| `sglang-benchmark` | SGLang dev workflow, benchmarking & profiling on MI355X / B200 |
| `sglang-glm5-fp8-decode-timeline` | Decode layer dual-stream kernel timeline (MI355X vs B200, MLA+NSA+MoE models) |
| `sglang-ci-repro` | Reproduce SGLang AMD CI failures locally on MI35x / MI325 boxes via `~/SGLang-benchmarks/tools/repro_ci.sh` (workflow → stage → suite → file mapping, partition handling, multimodal-gen edge case, aiter-version A/B) |
| `standup-meeting` | Write daily standup notes (Last day / Today, grouped by ticket ID) |
| `glm5-dsa-layer-structure` | GLM-5.2 / GLM-5 DSA decoder-layer structure & shared/full indexer-topk pattern; why the DSA indexer isn't on every layer (for reading analyze_sglang_trace.py breakdowns) |
