#!/usr/bin/env python3
"""
fix_ablations.py
Fixes two classes of bugs in the Disc Lifing paper training scripts:

1. PointNetMLPJoint in_channels hardcoded to 34 — crashes on edge-arc (51 feats),
   edge-arc-feat (119 feats), edge-zoneID, and any other variant whose feature
   count differs. Fix: derive in_channels from the loaded data at runtime.

2. Zonal-Full ablation uses discdatasetfullzonal.h5 but the script expects a
   'mesh'-representation HDF5. Fix: correct the H5 path in Full training scripts.

Usage:
    python fix_ablations.py [--dry-run] [--root /path/to/Disc_lifing_paper]

Arguments:
    --dry-run   Print what would be changed without writing anything.
    --root      Root directory of the repo (default: current working directory).
"""

import argparse
import glob
import os
import re
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — update these if your filenames differ
# ──────────────────────────────────────────────────────────────────────────────

# Wrong H5 filename used in Full-Zonal scripts and what it should be.
FULL_H5_BAD  = "discdatasetfullzonal.h5"
FULL_H5_GOOD = "discdatasetmeshzonal.h5"   # ← update if your mesh file differs

# ──────────────────────────────────────────────────────────────────────────────
# Patterns for in_channels=34 replacement
# ──────────────────────────────────────────────────────────────────────────────

IN_CHANNELS_PATTERNS = [
    # Handles:  in_channels=34  (possibly with spaces)
    (re.compile(r'\bin_channels\s*=\s*34\b'),
     'in_channels=in_channels_auto'),
    # Handles:  "in_channels": 34  (dict-style presets)
    (re.compile(r'"in_channels"\s*:\s*34\b'),
     '"in_channels": in_channels_auto'),
    # Handles:  'in_channels': 34
    (re.compile(r"'in_channels'\s*:\s*34\b"),
     "'in_channels': in_channels_auto"),
]

INJECT_MARKER = "in_channels_auto"

# Block injected once per file, after the dataset load, to detect feat dim.
INJECT_CODE = '''
# --- AUTO-INJECTED by fix_ablations.py: derive in_channels from dataset ---
with __import__('h5py').File(h5_path, 'r') as _hf:
    _sample_key = list(_hf.keys())[0]
    _feats = _hf[_sample_key]['edge_features'][:]
    in_channels_auto = int(_feats.shape[-1])
print(f"[auto] in_channels detected from data: {in_channels_auto}")
# --- END AUTO-INJECT ---
'''

# Inject the block after the first line matching this pattern.
INJECT_AFTER_PATTERN = re.compile(r'Loaded\s+\d+\s+datasets', re.IGNORECASE)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_training_scripts(root):
    pattern = os.path.join(root, "**", "Training_script*.py")
    return sorted(glob.glob(pattern, recursive=True))


def fix_full_h5_path(content, filepath, dry_run):
    if FULL_H5_BAD not in content:
        return content, False
    new_content = content.replace(FULL_H5_BAD, FULL_H5_GOOD)
    print(f"{'[DRY-RUN] ' if dry_run else ''}{filepath}")
    print(f"  [H5-PATH] {FULL_H5_BAD} → {FULL_H5_GOOD}")
    return new_content, True


def needs_in_channels_fix(content):
    for pat, _ in IN_CHANNELS_PATTERNS:
        if pat.search(content):
            return True
    return False


def apply_in_channels_fix(content, filepath, dry_run):
    changed = False
    new_content = content

    for pat, replacement in IN_CHANNELS_PATTERNS:
        if pat.search(new_content):
            new_content = pat.sub(replacement, new_content)
            changed = True

    if not changed:
        return content, False

    # Inject detection block only once, and only if not already present
    if INJECT_MARKER in new_content and INJECT_CODE.strip() not in new_content:
        lines = new_content.splitlines(keepends=True)
        insert_at = None
        for i, line in enumerate(lines):
            if INJECT_AFTER_PATTERN.search(line):
                insert_at = i + 1
                break
        if insert_at is None:
            # Fallback: insert before first model construction line
            for i, line in enumerate(lines):
                if 'Model initialized' in line or re.search(r'\bmodel\s*=', line):
                    insert_at = i
                    break
        if insert_at is not None:
            lines.insert(insert_at, INJECT_CODE)
            new_content = "".join(lines)
        else:
            print(f"  WARNING: could not find injection point in {filepath}."
                  " Manually add in_channels_auto detection before model init.")

    print(f"{'[DRY-RUN] ' if dry_run else ''}{filepath}")
    print("  [IN-CHANNELS] replaced hardcoded 34 with in_channels_auto")
    return new_content, True


def process_file(filepath, dry_run):
    with open(filepath, 'r', encoding='utf-8') as fh:
        original = fh.read()

    content = original
    any_change = False

    content, changed = fix_full_h5_path(content, filepath, dry_run)
    any_change = any_change or changed

    if needs_in_channels_fix(content):
        content, changed = apply_in_channels_fix(content, filepath, dry_run)
        any_change = any_change or changed

    if any_change and not dry_run:
        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(content)

    return any_change


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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
    print(f"Scanning for Training_script*.py under: {root}\n")
    scripts = find_training_scripts(root)

    if not scripts:
        print("No Training_script*.py files found. Check --root path.")
        sys.exit(1)

    print(f"Found {len(scripts)} script(s).\n")
    n_changed = 0
    for fp in scripts:
        if process_file(fp, args.dry_run):
            n_changed += 1

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Done. "
          f"{n_changed}/{len(scripts)} file(s) "
          f"{'would be' if args.dry_run else 'were'} modified.")

    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
