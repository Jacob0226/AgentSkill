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
| `sglang-decode-layer-timeline` | Decode layer dual-stream kernel timeline (MI355X vs B200, MLA+NSA+MoE models) |
