#!/usr/bin/env python3
"""
fix_ablations.py
Fixes two bugs in the Disc Lifing paper training scripts:

1. Full-variant scripts have EXPECTED_REPR = "mesh" but the H5 files
   (disc_dataset_full_zonal.h5 / disc_dataset_full_uniform.h5) store
   representation = "full". The H5 filenames are correct; only EXPECTED_REPR
   needs changing to "full".
   Affected: every Training_script*.py under */Full/*/

2. PointNetMLPJoint scripts hardcode in_channels=34, which crashes on
   edge-arc (51 feats), edge-arc-feat (119 feats), edge-zoneID, etc.
   Fix: replace the hardcoded value with a runtime lookup from the H5 data.
   Affected: every Training_script*.py under */PointNetMLPJoint/

Usage:
    python fix_ablations.py [--dry-run] [--root /path/to/Disc_lifing_paper]
"""

import argparse
import glob
import os
import re
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Fix 1: EXPECTED_REPR "mesh" -> "full" in Full-variant scripts
# ──────────────────────────────────────────────────────────────────────────────

EXPECTED_REPR_PATTERN = re.compile(
    r'(EXPECTED_REPR\s*:\s*str\s*=\s*)["\']mesh["\']'
)


def fix_expected_repr(content, filepath, dry_run):
    """Change EXPECTED_REPR = "mesh" -> "full" only in Full-variant scripts."""
    # Only touch scripts that live under a 'Full' ablation folder
    if not re.search(r'[\\/]Full[\\/]', filepath):
        return content, False
    if not EXPECTED_REPR_PATTERN.search(content):
        return content, False
    new_content = EXPECTED_REPR_PATTERN.sub(r'\1"full"', content)
    print(f"{'[DRY-RUN] ' if dry_run else ''}{filepath}")
    print('  [EXPECTED_REPR] "mesh" -> "full"')
    return new_content, True


# ──────────────────────────────────────────────────────────────────────────────
# Fix 2: dynamic in_channels for PointNetMLPJoint scripts
# ──────────────────────────────────────────────────────────────────────────────

IN_CHANNELS_PATTERNS = [
    (re.compile(r'\bin_channels\s*=\s*34\b'),        'in_channels=in_channels_auto'),
    (re.compile(r'"in_channels"\s*:\s*34\b'),         '"in_channels": in_channels_auto'),
    (re.compile(r"'in_channels'\s*:\s*34\b"),         "'in_channels': in_channels_auto"),
]

INJECT_MARKER = "in_channels_auto"

INJECT_CODE = '''
# --- AUTO-INJECTED by fix_ablations.py: derive in_channels from dataset ---
with __import__('h5py').File(h5_path, 'r') as _hf:
    _sample_key = list(_hf.keys())[0]
    _feats = _hf[_sample_key]['edge_features'][:]
    in_channels_auto = int(_feats.shape[-1])
print(f"[auto] in_channels detected from data: {in_channels_auto}")
# --- END AUTO-INJECT ---
'''

INJECT_AFTER_PATTERN = re.compile(r'Loaded\s+\d+\s+datasets', re.IGNORECASE)


def needs_in_channels_fix(content):
    return any(pat.search(content) for pat, _ in IN_CHANNELS_PATTERNS)


def apply_in_channels_fix(content, filepath, dry_run):
    changed = False
    new_content = content
    for pat, replacement in IN_CHANNELS_PATTERNS:
        if pat.search(new_content):
            new_content = pat.sub(replacement, new_content)
            changed = True
    if not changed:
        return content, False

    # Inject detection block once, after "Loaded N datasets" print
    if INJECT_MARKER in new_content and INJECT_CODE.strip() not in new_content:
        lines = new_content.splitlines(keepends=True)
        insert_at = None
        for i, line in enumerate(lines):
            if INJECT_AFTER_PATTERN.search(line):
                insert_at = i + 1
                break
        if insert_at is None:
            for i, line in enumerate(lines):
                if 'Model initialized' in line or re.search(r'\bmodel\s*=', line):
                    insert_at = i
                    break
        if insert_at is not None:
            lines.insert(insert_at, INJECT_CODE)
            new_content = ''.join(lines)
        else:
            print(f'  WARNING: could not find injection point in {filepath}.'
                  ' Add in_channels_auto detection manually before model init.')

    print(f"{'[DRY-RUN] ' if dry_run else ''}{filepath}")
    print('  [IN-CHANNELS] replaced hardcoded 34 with in_channels_auto')
    return new_content, True


# ──────────────────────────────────────────────────────────────────────────────
# File processing
# ──────────────────────────────────────────────────────────────────────────────

def find_training_scripts(root):
    pattern = os.path.join(root, '**', 'Training_script*.py')
    return sorted(glob.glob(pattern, recursive=True))


def process_file(filepath, dry_run):
    with open(filepath, 'r', encoding='utf-8') as fh:
        content = fh.read()

    any_change = False

    content, changed = fix_expected_repr(content, filepath, dry_run)
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
