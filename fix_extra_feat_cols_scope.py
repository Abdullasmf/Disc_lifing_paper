#!/usr/bin/env python3
"""
fix_extra_feat_cols_scope.py
============================
Run from the ROOT of the Disc_lifing_paper repository:

    python fix_extra_feat_cols_scope.py

What this script fixes
-----------------------
All Training_script*.py files under Zonal/ and Uniform/ suffer from one of
two variants of the same scoping bug:

  BUG A – No module-level EXTRA_FEAT_COLS declaration at all
      (present in variants that carry extra features, e.g. Edge_arc_feat).
      The name is only ever assigned inside main(), yet GeomLifeDataset,
      build_enc_norm, and compute_global_normalization all reference it as
      a free (global) variable.  In normal execution order this "works", but:
        • Any import-time inspection or test that instantiates the class before
          main() assigns the variable raises NameError.
        • The pattern is non-standard, fragile, and easy to break.

  BUG B – Stale INPUT_COLS declaration still present
      (e.g. `INPUT_COLS: List[int] = [0, 1, 3, 4, 5, 6, 7]`)
      This variable is dead code left over from the pre-patch state.
      Nothing uses it, but it misleads readers and is evidence of an
      incomplete migration.

Fixes applied
-------------
  1. If EXTRA_FEAT_COLS is NOT declared at module level → insert a safe
     module-level sentinel:
         EXTRA_FEAT_COLS: List[int] = []  # populated in main() from data width
     immediately after the PER-ABLATION CONFIG block closing comment,
     or after the last INPUT_COLS / TARGET_NAMES / NUM_TARGETS declaration,
     whichever comes first.

  2. If a stale INPUT_COLS line exists at module level AND EXTRA_FEAT_COLS is
     already wired through the code (i.e. INPUT_COLS is never referenced after
     the declaration), the line is commented out with a note.

  3. Every fix is idempotent – running the script twice produces no additional
     changes.

  4. A .bak file is written alongside each modified script so you can diff or
     revert.
"""

import os
import re
import sys
import shutil
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
SEARCH_ROOTS = [REPO_ROOT / "Zonal", REPO_ROOT / "Uniform"]
SCRIPT_GLOB = "Training_script*.py"

# Sentinel comment that marks the script as already patched by THIS tool
PATCH_MARKER = "# [fix_extra_feat_cols_scope] patched"

# The module-level sentinel line we inject
SENTINEL_LINE = "EXTRA_FEAT_COLS: List[int] = []  # populated in main() from data width\n"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_training_scripts(search_roots):
    scripts = []
    for root in search_roots:
        if not root.exists():
            print(f"  [WARN] Directory not found, skipping: {root}")
            continue
        for path in sorted(root.rglob(SCRIPT_GLOB)):
            scripts.append(path)
    return scripts


def has_module_level_extra_feat_cols(lines):
    """
    Returns True if EXTRA_FEAT_COLS is assigned at module level
    (i.e. NOT inside an indented block such as a function or class).
    We look for lines that:
      - Are not indented (no leading whitespace)
      - Match `EXTRA_FEAT_COLS` on the left-hand side of an assignment
    """
    for line in lines:
        # Not indented → module level
        if re.match(r'^EXTRA_FEAT_COLS\s*[:=]', line):
            return True
    return False


def has_patch_marker(lines):
    return any(PATCH_MARKER in line for line in lines)


def find_insert_position(lines):
    """
    Find the best line number (0-indexed) AFTER which to insert the sentinel.

    Priority order:
      1. After '# ==== END PER-ABLATION CONFIG ====' comment block
      2. After the last module-level INPUT_COLS / TARGET_NAMES / NUM_TARGETS /
         QUERY_COLS / H5_FILENAME / EXPECTED_REPR declaration (whichever is last)
      3. After the last `from ... import ...` or `import ...` line at module level
      4. Line 0 as absolute fallback

    Returns the index of the line AFTER which to insert (so insert at idx+1).
    """
    end_config_re = re.compile(r'^#\s*====\s*END PER-ABLATION CONFIG', re.IGNORECASE)
    module_decl_re = re.compile(
        r'^(INPUT_COLS|TARGET_NAMES|NUM_TARGETS|QUERY_COLS|H5_FILENAME|EXPECTED_REPR)\s*[:=]'
    )
    import_re = re.compile(r'^(import |from )')

    end_config_idx = -1
    last_decl_idx = -1
    last_import_idx = -1

    for i, line in enumerate(lines):
        if end_config_re.match(line):
            end_config_idx = i
        if module_decl_re.match(line):
            last_decl_idx = i
        if import_re.match(line):
            last_import_idx = i

    if end_config_idx >= 0:
        return end_config_idx  # insert AFTER this line
    if last_decl_idx >= 0:
        return last_decl_idx
    if last_import_idx >= 0:
        return last_import_idx
    return 0


def comment_out_input_cols(lines):
    """
    If INPUT_COLS is declared at module level AND is never referenced elsewhere
    in the file (other than its own declaration line), comment it out.

    Returns (new_lines, was_changed).
    """
    # Find module-level INPUT_COLS declaration lines
    decl_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^INPUT_COLS\s*[:=]', line) and "# [STALE]" not in line:
            decl_indices.append(i)

    if not decl_indices:
        return lines, False

    # Count references to INPUT_COLS outside those declaration lines
    ref_count = 0
    for i, line in enumerate(lines):
        if i in decl_indices:
            continue
        # Look for bare usage (not inside a comment)
        stripped = line.split('#')[0]  # drop inline comments
        if 'INPUT_COLS' in stripped:
            ref_count += 1

    if ref_count > 0:
        # INPUT_COLS is still used somewhere – leave it alone
        return lines, False

    # Safe to comment it out
    new_lines = list(lines)
    for i in decl_indices:
        original = new_lines[i].rstrip('\n')
        new_lines[i] = f"# [STALE – removed by fix_extra_feat_cols_scope] {original}\n"
    return new_lines, True


def patch_file(path: Path) -> str:
    """
    Apply all fixes to a single file.
    Returns a short status string.
    """
    original_text = path.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)

    if has_patch_marker(lines):
        return "SKIP (already patched)"

    changed = False
    report_parts = []

    # ── Fix 1: comment out stale INPUT_COLS ──────────────────────────────────
    lines, input_cols_fixed = comment_out_input_cols(lines)
    if input_cols_fixed:
        changed = True
        report_parts.append("stale INPUT_COLS commented out")

    # ── Fix 2: add module-level EXTRA_FEAT_COLS sentinel if missing ──────────
    if not has_module_level_extra_feat_cols(lines):
        insert_after = find_insert_position(lines)
        # Build the block to insert (blank line + sentinel)
        insert_block = ["\n", SENTINEL_LINE]
        lines = lines[:insert_after + 1] + insert_block + lines[insert_after + 1:]
        changed = True
        report_parts.append(f"EXTRA_FEAT_COLS sentinel inserted after line {insert_after + 1}")
    else:
        report_parts.append("EXTRA_FEAT_COLS already declared at module level")

    if not changed:
        return "OK (no changes needed)"

    # ── Add patch marker at top of file ──────────────────────────────────────
    # Insert after the first line (shebang / encoding declaration / existing header comment)
    # Find a good insertion point: after any leading comment/shebang block
    marker_inserted = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') or stripped == '':
            continue
        # First non-comment, non-blank line – insert marker just before it
        lines.insert(i, PATCH_MARKER + "\n")
        marker_inserted = True
        break
    if not marker_inserted:
        lines.insert(0, PATCH_MARKER + "\n")

    # ── Write backup + patched file ──────────────────────────────────────────
    backup_path = path.with_suffix(".py.bak")
    shutil.copy2(path, backup_path)
    path.write_text("".join(lines), encoding="utf-8")

    return "PATCHED: " + "; ".join(report_parts)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("fix_extra_feat_cols_scope.py")
    print(f"Repo root : {REPO_ROOT}")
    print("=" * 70)

    scripts = find_training_scripts(SEARCH_ROOTS)
    if not scripts:
        print("No Training_script*.py files found. Check that you are running")
        print("this from the repository root.")
        sys.exit(1)

    print(f"\nFound {len(scripts)} training script(s):\n")
    results = {}
    for script in scripts:
        rel = script.relative_to(REPO_ROOT)
        status = patch_file(script)
        results[str(rel)] = status
        indicator = "✓" if status.startswith("PATCHED") else ("–" if status.startswith("SKIP") else "·")
        print(f"  {indicator}  {rel}")
        print(f"       {status}")

    patched = sum(1 for s in results.values() if s.startswith("PATCHED"))
    skipped = sum(1 for s in results.values() if s.startswith("SKIP"))
    ok      = sum(1 for s in results.values() if s.startswith("OK"))

    print("\n" + "=" * 70)
    print(f"Summary: {patched} patched, {ok} already clean, {skipped} skipped (already patched)")
    print("=" * 70)

    if patched:
        print("\nBackup files (.bak) written alongside each patched script.")
        print("To verify a patch:  diff <script>.py.bak <script>.py")
        print("To revert:          cp <script>.py.bak <script>.py")


if __name__ == "__main__":
    main()
