#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="$HOME/.cursor/skills"

mkdir -p "$SKILLS_DIR"

# Clean up dangling symlinks (e.g. after skill renames)
find "$SKILLS_DIR" -maxdepth 1 -type l ! -exec test -e {} \; -print | while read -r link; do
    echo "Removing dangling symlink: $link"
    rm "$link"
done

# Skills live either at the top level or one directory down, so related skills
# can be grouped (e.g. PR/sglang-pr, PR/aiter-pr). The group directory itself
# has no SKILL.md and is not linked; only the leaves are.
for skill_dir in "$SCRIPT_DIR"/*/ "$SCRIPT_DIR"/*/*/; do
    [ -f "$skill_dir/SKILL.md" ] || continue
    skill_name="$(basename "$skill_dir")"
    target="$SKILLS_DIR/$skill_name"

    if [ -L "$target" ]; then
        echo "Updating symlink: $target -> $skill_dir"
        rm "$target"
    elif [ -d "$target" ]; then
        echo "WARNING: $target exists as a directory (not a symlink). Backing up to ${target}.bak"
        mv "$target" "${target}.bak"
    fi

    ln -s "$skill_dir" "$target"
    echo "Linked: $target -> $skill_dir"
done

echo "Done. Skills installed:"
ls -la "$SKILLS_DIR"/
