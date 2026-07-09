#!/usr/bin/env python3
"""
fix_pointnet_features.py
Run from the repo root: python fix_pointnet_features.py

Fixes identified in audit of Disc_lifing_paper PointNetMLPJoint variants:

1. EXTRA_FEAT_COLS scoping bug (all Training_script*.py):
   Adds `global EXTRA_FEAT_COLS` as the first executable line inside main()
   so the module-level variable is updated and all dataset/normalization
   functions see the real column list instead of the empty [] placeholder.

2. Stale `INPUT_COLS = [0, 1, 3]` in Edge_arc/Training_script.py:
   Removes the unused line entirely.

3. Missing module-scope sentinels in Edge_arc/Training_script.py:
   Ensures `EXTRA_FEAT_COLS: List[int] = []` and
   `QUERY_COLS: List[int] = [0, 1]` exist at module scope,
   consistent with the Edge and Edge_Prox variants.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

TRAINING_SCRIPTS = [
    "Uniform/Edge/PointNetMLPJoint/Training_script.py",
    "Uniform/Edge/PointNetMLPJoint/Training_script_L1.py",
    "Uniform/Edge_Prox/PointNetMLPJoint/Training_script.py",
    "Uniform/Edge_Prox/PointNetMLPJoint/Training_script_L1.py",
    "Uniform/Edge_arc/PointNetMLPJoint/Training_script.py",
    "Uniform/Edge_arc/PointNetMLPJoint/Training_script_L1.py",
]

# ---------------------------------------------------------------------------
# Fix 1: Inject global EXTRA_FEAT_COLS as the first statement inside main()
# ---------------------------------------------------------------------------

def inject_global_extra_feat_cols(source):
    changes = []

    if "global EXTRA_FEAT_COLS" in source:
        return source, ["  [SKIP] global EXTRA_FEAT_COLS already present"]

    lines = source.splitlines(keepends=True)
    in_main = False
    insert_after = -1
    indent = "    "

    for i, line in enumerate(lines):
        if re.match(r"^def main\(", line):
            in_main = True
            continue
        if in_main:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            m = re.match(r"^(\s+)", line)
            indent = m.group(1) if m else "    "
            insert_after = i
            break

    if insert_after == -1:
        return source, ["  [ERROR] Could not locate main() body -- manual fix needed"]

    injection = "{}global EXTRA_FEAT_COLS\n".format(indent)
    lines.insert(insert_after, injection)
    changes.append("  + Injected `global EXTRA_FEAT_COLS` before line {} (inside main)".format(insert_after + 1))
    return "".join(lines), changes


# ---------------------------------------------------------------------------
# Fix 2: Remove stale INPUT_COLS = [0, 1, 3] at module scope (Edge_arc only)
# ---------------------------------------------------------------------------

STALE_INPUT_COLS_RE = re.compile(
    r"^INPUT_COLS:\s*List\[int\]\s*=\s*\[.*?\].*?\n",
    re.MULTILINE,
)

def remove_stale_input_cols(source):
    changes = []
    new, n = STALE_INPUT_COLS_RE.subn("", source)
    if n:
        changes.append("  - Removed {} stale INPUT_COLS = [...] line(s)".format(n))
    return new, changes


# ---------------------------------------------------------------------------
# Fix 3: Ensure module-scope sentinels exist (Edge_arc only)
# ---------------------------------------------------------------------------

EXTRA_FEAT_SENTINEL = "EXTRA_FEAT_COLS: List[int] = []"
QUERY_COLS_SENTINEL  = "QUERY_COLS: List[int] = [0, 1]  # head query always uses (x, r)"
END_ABLATION_MARKER  = "# ==== END PER-ABLATION CONFIG ===="

def ensure_module_scope_sentinels(source):
    changes = []

    if END_ABLATION_MARKER not in source:
        return source, ["  [SKIP] END PER-ABLATION CONFIG marker not found"]

    if "EXTRA_FEAT_COLS: List[int]" not in source:
        source = source.replace(
            END_ABLATION_MARKER,
            END_ABLATION_MARKER + "\n\n" + EXTRA_FEAT_SENTINEL,
        )
        changes.append("  + Added module-scope `EXTRA_FEAT_COLS: List[int] = []`")

    if "QUERY_COLS: List[int]" not in source:
        target = "EXTRA_FEAT_COLS: List[int] = []"
        if target in source:
            source = source.replace(target, target + "\n" + QUERY_COLS_SENTINEL)
        else:
            source = source.replace(
                END_ABLATION_MARKER,
                END_ABLATION_MARKER + "\n" + QUERY_COLS_SENTINEL,
            )
        changes.append("  + Added module-scope `QUERY_COLS: List[int] = [0, 1]`")

    return source, changes


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------

def process_file(rel_path):
    full_path = REPO_ROOT / rel_path
    if not full_path.exists():
        print("[MISSING] {} -- file not found, skipping".format(rel_path))
        return

    source = full_path.read_text(encoding="utf-8")
    original = source
    all_changes = []

    is_edge_arc = "Edge_arc" in rel_path

    source, changes = inject_global_extra_feat_cols(source)
    all_changes.extend(changes)

    if is_edge_arc:
        source, changes = remove_stale_input_cols(source)
        all_changes.extend(changes)

    if is_edge_arc:
        source, changes = ensure_module_scope_sentinels(source)
        all_changes.extend(changes)

    if source != original:
        full_path.write_text(source, encoding="utf-8")
        print("[FIXED]   {}".format(rel_path))
    else:
        print("[OK]      {}".format(rel_path))

    for c in all_changes:
        print(c)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Repo root: {}\n".format(REPO_ROOT))
    for script_path in TRAINING_SCRIPTS:
        process_file(script_path)
    print("\nDone. Run `git diff` to review all changes before committing.")


if __name__ == "__main__":
    main()
