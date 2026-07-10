#!/usr/bin/env python3
"""
patch_training_scripts.py
Run from the repository root:
    python patch_training_scripts.py [--dry-run]

Fixes applied to every Training_script*.py under Zonal/ and Uniform/:
  Fix 1 — Insert `global EXTRA_FEAT_COLS` as the first executable statement
           inside def main(), immediately before the first print().
           Skipped if already present.
  Fix 2 — Remove stale INPUT_COLS module-level declarations and any
           INPUT_COLS dynamic-reassignment blocks that were left behind
           from an earlier (incomplete) migration attempt.
  Fix 3 — Update the build_enc_norm docstring from the legacy
           "per-INPUT_COL ... aligned to INPUT_COLS" wording to refer to
           EXTRA_FEAT_COLS so that the comment matches reality.
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_scripts(repo_root: Path):
    scripts = []
    for pattern in ["Zonal/*/PointNetMLPJoint/Training_script*.py",
                     "Uniform/*/PointNetMLPJoint/Training_script*.py"]:
        scripts.extend(sorted(repo_root.glob(pattern)))
    return scripts


def apply_fix1_global_extra_feat_cols(lines, path_label):
    """Insert `    global EXTRA_FEAT_COLS` as first executable line in main()."""
    # Already patched?
    for line in lines:
        if re.match(r'\s*global\s+EXTRA_FEAT_COLS\s*$', line):
            return lines, False, "SKIP (global already present)"

    new_lines = []
    inserted = False
    in_main = False

    for i, line in enumerate(lines):
        # Detect def main(
        if re.match(r'^def main\(', line):
            in_main = True
            new_lines.append(line)
            continue

        if in_main and not inserted:
            # Skip blank lines and comment-only lines inside main()
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                new_lines.append(line)
                continue
            # This is the first real executable line inside main()
            # Insert the global statement before it
            indent = re.match(r'(\s*)', line).group(1)
            new_lines.append(f"{indent}global EXTRA_FEAT_COLS\n")
            inserted = True

        new_lines.append(line)

    if inserted:
        return new_lines, True, "OK"
    return lines, False, "WARN: could not locate insertion point in main()"


def apply_fix2_remove_stale_input_cols(lines, path_label):
    """
    Remove:
     a) Module-level INPUT_COLS = [...] lines (legacy dead variable).
     b) The INPUT_COLS dynamic-reassignment comment block + assignment in main()
        (present in Zonal/Edge_zoneID only; identified by the comment
        "# [fix_ablations] INPUT_COLS dynamic").
    """
    changed = False
    new_lines = []

    # Pattern for module-level INPUT_COLS declaration
    module_level_re = re.compile(
        r'^INPUT_COLS\s*:\s*List\[int\]\s*=\s*\[.*\]'
        r'|^INPUT_COLS\s*=\s*\[.*\]'
    )
    # Pattern for the dynamic-reassignment block (Edge_zoneID artifact)
    dyn_block_re = re.compile(r'^\s*#\s*\[fix_ablations\]\s*INPUT_COLS\s+dynamic')
    dyn_assign_re = re.compile(r'^\s*INPUT_COLS\s*=\s*list\(range\(')

    in_dyn_block = False
    i = 0
    while i < len(lines):
        line = lines[i]

        # Module-level INPUT_COLS = [...]  — drop the line
        if module_level_re.match(line):
            changed = True
            i += 1
            continue

        # Dynamic block start marker
        if dyn_block_re.match(line):
            in_dyn_block = True
            changed = True
            i += 1
            continue

        # Inside the dynamic block: drop comment lines and the assignment
        if in_dyn_block:
            stripped = line.strip()
            if stripped.startswith('#') or dyn_assign_re.match(line):
                i += 1
                continue
            # Any non-comment, non-assignment line ends the block
            in_dyn_block = False

        new_lines.append(line)
        i += 1

    msg = "OK" if changed else "SKIP (no stale INPUT_COLS found)"
    return new_lines, changed, msg


def apply_fix3_docstring_update(lines, path_label):
    """Replace legacy build_enc_norm docstring wording."""
    old = 'Build per-INPUT_COL normalization (mean, std) vectors aligned to INPUT_COLS.'
    new = 'Build per-EXTRA_FEAT_COL normalization (mean, std) vectors aligned to EXTRA_FEAT_COLS.'
    changed = False
    new_lines = []
    for line in lines:
        if old in line:
            new_lines.append(line.replace(old, new))
            changed = True
        else:
            new_lines.append(line)
    msg = "OK" if changed else "SKIP (docstring already updated or not found)"
    return new_lines, changed, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Patch PointNetMLPJoint training scripts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be changed without writing files.")
    parser.add_argument("--repo-root", default=".",
                        help="Path to repository root (default: current directory).")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    scripts = find_scripts(repo_root)

    if not scripts:
        print(f"ERROR: No Training_script*.py files found under {repo_root}/Zonal or {repo_root}/Uniform")
        sys.exit(1)

    print(f"Found {len(scripts)} script(s) under {repo_root}\n")

    total_changed = 0

    for script_path in scripts:
        label = script_path.relative_to(repo_root)
        lines = script_path.read_text(encoding="utf-8").splitlines(keepends=True)

        results = {}

        lines, c1, m1 = apply_fix1_global_extra_feat_cols(lines, label)
        results["Fix1 global EXTRA_FEAT_COLS"] = (c1, m1)

        lines, c2, m2 = apply_fix2_remove_stale_input_cols(lines, label)
        results["Fix2 remove INPUT_COLS"] = (c2, m2)

        lines, c3, m3 = apply_fix3_docstring_update(lines, label)
        results["Fix3 docstring update"] = (c3, m3)

        file_changed = any(c for c, _ in results.values())
        total_changed += int(file_changed)

        prefix = "[DRY-RUN] " if args.dry_run else ""
        status_str = "MODIFIED" if file_changed else "unchanged"
        print(f"{prefix}{status_str}: {label}")
        for fix_name, (changed, msg) in results.items():
            marker = "+" if changed else "·"
            print(f"    {marker} {fix_name}: {msg}")

        if file_changed and not args.dry_run:
            script_path.write_text("".join(lines), encoding="utf-8")

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Done. {total_changed}/{len(scripts)} file(s) modified.")
    if args.dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
