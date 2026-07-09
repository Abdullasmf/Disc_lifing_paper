#!/usr/bin/env python3
"""
fix_ablations.py
Fixes two bugs in the Disc Lifing paper training scripts:

1. Full-variant scripts have EXPECTED_REPR = "mesh" but the H5 files store
   representation = "full". Fix: change EXPECTED_REPR to "full".
   Affected: every Training_script*.py under */Full/*/

2. PointNetMLPJoint scripts hardcode INPUT_COLS = [0, 1] (only x, r).
   This means the model always gets in_channels=2, ignoring all extra
   features in the H5 (arc_length, tangent_x, tangent_r, curvature, etc.).
   Fix: replace the hardcoded INPUT_COLS list with a dynamic version that
   reads the actual number of feature columns from the first H5 sample at
   runtime, BEFORE the model is constructed.
   Affected: every Training_script*.py under */PointNetMLPJoint/

Usage:
    python fix_ablations.py [--dry-run] [--root /path/to/Disc_lifing_paper]
"""

import argparse
import glob
import os
import re
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: EXPECTED_REPR "mesh" -> "full" in Full-variant scripts
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_REPR_PATTERN = re.compile(
    r'(EXPECTED_REPR\s*:\s*str\s*=\s*)["\']mesh["\']'
)


def fix_expected_repr(content, filepath, dry_run):
    """Change EXPECTED_REPR = "mesh" -> "full" only in Full-variant scripts."""
    if not re.search(r'[\\/]Full[\\/]', filepath):
        return content, False
    if not EXPECTED_REPR_PATTERN.search(content):
        return content, False
    new_content = EXPECTED_REPR_PATTERN.sub(r'\1"full"', content)
    print(f"{'[DRY-RUN] ' if dry_run else ''}{filepath}")
    print('  [EXPECTED_REPR] "mesh" -> "full"')
    return new_content, True


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Dynamic INPUT_COLS for PointNetMLPJoint scripts
# ─────────────────────────────────────────────────────────────────────────────

# Matches the hardcoded per-ablation config block line:
#   INPUT_COLS: List[int] = [0, 1]
# The list may contain 2+ ints but we only want to replace the *config-block*
# declaration (not other references). We anchor on the comment block that
# surrounds it in every script.
INPUT_COLS_DECL_PATTERN = re.compile(
    r'^(INPUT_COLS\s*:\s*List\[int\]\s*=\s*)\[.*?\]',
    re.MULTILINE,
)

# Sentinel so we don't inject twice
INJECT_SENTINEL = '# [fix_ablations] INPUT_COLS dynamic'

# Code block injected just before the model is constructed (before
# PointNetMLPJoint(...) call). It reads the width of the first sample,
# subtracts the 2 target columns appended by the loader, and builds
# INPUT_COLS = [0, 1, ..., width-3] covering all feature columns.
INJECT_CODE_TEMPLATE = '''
{sentinel}
# Derive INPUT_COLS dynamically from the first sample in the H5 file so that
# all feature columns (x, r, zone_id, arc_length, tangent_x, …) are used.
# The loader always appends 2 target columns (stress, log_life) at the end,
# so feature width = sample_width - 2.
_first_sample_width = int(PS_list_whole[0].shape[1])
_n_feature_cols = _first_sample_width - 2  # subtract stress + log_life
INPUT_COLS = list(range(_n_feature_cols))
print(f"[fix_ablations] INPUT_COLS set dynamically: {{INPUT_COLS}} ({{len(INPUT_COLS)}} channels)")
'''

# We inject AFTER the data-load block, identified by the line that prints how
# many datasets were loaded. This is reliably present in all scripts.
INJECT_AFTER_PATTERN = re.compile(
    r'print\s*\(.*?[Ll]oaded.*?dataset', re.IGNORECASE
)


def needs_input_cols_fix(content, filepath):
    """Only touch PointNetMLPJoint scripts that still have hardcoded INPUT_COLS."""
    if 'PointNetMLPJoint' not in filepath:
        return False
    if INJECT_SENTINEL in content:
        return False  # already patched
    return bool(INPUT_COLS_DECL_PATTERN.search(content))


def apply_input_cols_fix(content, filepath, dry_run):
    # Step A: replace the hardcoded declaration with a placeholder comment so
    # the name exists at module level (required for the normalization code that
    # references INPUT_COLS before main() runs).
    new_content = INPUT_COLS_DECL_PATTERN.sub(
        r'\1[0, 1]  # overridden dynamically below inside main()',
        content,
        count=1,
    )

    # Step B: inject the dynamic derivation block inside main(), right after
    # the line that prints "Loaded N datasets".
    inject_code = INJECT_CODE_TEMPLATE.format(sentinel=INJECT_SENTINEL)
    lines = new_content.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        if INJECT_AFTER_PATTERN.search(line):
            insert_at = i + 1
            break

    if insert_at is None:
        # Fallback: inject just before PointNetMLPJoint model construction
        for i, line in enumerate(lines):
            if 'PointNetMLPJoint(' in line:
                insert_at = i
                break

    if insert_at is not None:
        lines.insert(insert_at, inject_code)
        new_content = ''.join(lines)
        print(f"{'[DRY-RUN] ' if dry_run else ''}{filepath}")
        print(f'  [INPUT_COLS] will be set dynamically from H5 data at runtime')
        return new_content, True
    else:
        print(f'  WARNING: could not find injection point in {filepath}. Skipping INPUT_COLS fix.')
        return content, False


# ─────────────────────────────────────────────────────────────────────────────
# File processing
# ─────────────────────────────────────────────────────────────────────────────

def find_training_scripts(root):
    pattern = os.path.join(root, '**', 'Training_script*.py')
    return sorted(glob.glob(pattern, recursive=True))


def process_file(filepath, dry_run):
    with open(filepath, 'r', encoding='utf-8') as fh:
        content = fh.read()

    any_change = False

    content, changed = fix_expected_repr(content, filepath, dry_run)
    any_change = any_change or changed

    if needs_input_cols_fix(content, filepath):
        content, changed = apply_input_cols_fix(content, filepath, dry_run)
        any_change = any_change or changed

    if any_change and not dry_run:
        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(content)

    return any_change


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show changes without writing files')
    parser.add_argument('--root', default='.',
                        help='Root of the repo (default: current directory)')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f'Scanning for Training_script*.py under: {root}\n')
    scripts = find_training_scripts(root)

    if not scripts:
        print('No Training_script*.py files found. Check --root path.')
        sys.exit(1)

    print(f'Found {len(scripts)} script(s).\n')
    n_changed = 0
    for fp in scripts:
        if process_file(fp, args.dry_run):
            n_changed += 1

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Done. "
          f"{n_changed}/{len(scripts)} file(s) "
          f"{'would be' if args.dry_run else 'were'} modified.")
    if args.dry_run:
        print('\nRe-run without --dry-run to apply changes.')


if __name__ == '__main__':
    main()
