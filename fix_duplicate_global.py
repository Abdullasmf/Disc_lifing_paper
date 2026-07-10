#!/usr/bin/env python3
"""
fix_duplicate_global.py
=======================
Removes duplicate `global EXTRA_FEAT_COLS` (and `global HEAD_FEAT_COLS`) declarations
that were injected twice by the patch agent into PointNetMLPJoint training scripts.

The rule applied is simple and surgical:
  - Walk every Training_script.py and Training_script_L1.py under
    Uniform/*/PointNetMLPJoint/ and Zonal/*/PointNetMLPJoint/
    (both standard and _headfeat variants).
  - Within the body of each function definition, if the same
    `global <NAME>` line appears more than once, keep the FIRST
    occurrence and delete every subsequent duplicate.
  - A line is considered a duplicate if its stripped content equals
    a `global <NAME>` line already seen in the current function scope.
  - Lines that are only whitespace-different from the canonical form
    are treated as duplicates too (strip() comparison).
  - No other changes are made to the file.

Usage (run from repo root):
    python fix_duplicate_global.py [--dry-run]

Options:
    --dry-run   Print what would change without writing any file.
"""

import argparse
import ast
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLOBAL_RE = re.compile(r"^([ \t]*)global\s+(.+)$")


def _strip_trailing_comment(line: str) -> str:
    """Return the code portion of a line (before any # comment), stripped."""
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line.rstrip()


def parse_global_names(line: str) -> frozenset:
    """Return frozenset of variable names declared in a `global` statement."""
    code = _strip_trailing_comment(line)
    m = GLOBAL_RE.match(code)
    if not m:
        return frozenset()
    names_part = m.group(2)
    names = frozenset(n.strip() for n in names_part.split(",") if n.strip())
    return names


def remove_duplicate_globals(source: str) -> tuple:
    """
    Return (new_source, removed_linenos).

    removed_linenos is 1-based.  The algorithm:
      - Track function depth via indentation.
      - Within each function scope, track which global names have already
        appeared on a `global` statement.
      - On seeing a second `global X` in the same function scope, drop that line.
    """
    lines = source.splitlines(keepends=True)
    removed = []
    result = []

    # Stack of sets: each entry = set of global names declared in that function.
    scope_stack = [set()]   # index 0 = module scope
    indent_stack = [-1]     # indent of the def/class line that opened the scope

    def current_indent(line):
        return len(line) - len(line.lstrip())

    for lineno, line in enumerate(lines, start=1):
        stripped = line.rstrip("\n\r")

        # Pop scopes that are now closed (skip blank lines for indent tracking)
        if stripped.strip():
            ci = current_indent(stripped)
            while len(indent_stack) > 1 and ci <= indent_stack[-1]:
                scope_stack.pop()
                indent_stack.pop()

        # Detect new function/class scope opening
        func_m = re.match(r"^([ \t]*)(?:async\s+)?def\s+\w+|^([ \t]*)class\s+\w+", stripped)
        if func_m:
            ci = current_indent(stripped)
            scope_stack.append(set())
            indent_stack.append(ci)
            result.append(line)
            continue

        # Check for global statement inside a function scope
        if stripped.strip().startswith("global ") and len(scope_stack) > 1:
            names = parse_global_names(stripped.strip())
            if names:
                current_scope = scope_stack[-1]
                new_names = names - current_scope
                already_seen = names & current_scope

                if already_seen and not new_names:
                    # Pure duplicate — every name already declared → drop line
                    removed.append(lineno)
                    continue
                elif already_seen and new_names:
                    # Partial duplicate — keep only new names, rewrite line
                    indent = re.match(r"^([ \t]*)", stripped).group(1)
                    new_line = indent + "global " + ", ".join(sorted(new_names)) + "\n"
                    result.append(new_line)
                    removed.append(lineno)  # log as modified
                    current_scope.update(new_names)
                    continue
                else:
                    current_scope.update(names)

        result.append(line)

    return "".join(result), removed


def find_target_scripts(repo_root):
    """Find all Training_script*.py under PointNetMLPJoint* folders."""
    targets = []
    for pattern in ("Uniform", "Zonal"):
        base = repo_root / pattern
        if not base.exists():
            continue
        for ablation_dir in sorted(base.iterdir()):
            if not ablation_dir.is_dir():
                continue
            for model_dir in sorted(ablation_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                if "PointNetMLPJoint" not in model_dir.name:
                    continue
                for script in sorted(model_dir.glob("Training_script*.py")):
                    targets.append(script)
    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate global declarations injected by patch agent."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print changes without writing files."
    )
    parser.add_argument(
        "--repo-root", default=".",
        help="Path to repo root (default: cwd)."
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    scripts = find_target_scripts(repo_root)

    if not scripts:
        print("No PointNetMLPJoint training scripts found. Check --repo-root.")
        sys.exit(1)

    total_fixed = 0
    total_lines_removed = 0

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        new_source, removed = remove_duplicate_globals(source)

        if not removed:
            print(f"  OK  (no duplicates)  {script.relative_to(repo_root)}")
            continue

        rel = script.relative_to(repo_root)
        print(
            f"  FIX {rel}  — removed {len(removed)} duplicate global line(s) "
            f"at line(s): {removed}"
        )
        total_fixed += 1
        total_lines_removed += len(removed)

        if not args.dry_run:
            script.write_text(new_source, encoding="utf-8")
            # Sanity-parse the patched file
            try:
                ast.parse(new_source)
            except SyntaxError as exc:
                print(f"    !! SYNTAX ERROR after patch: {exc} — reverting")
                script.write_text(source, encoding="utf-8")
                total_fixed -= 1
                total_lines_removed -= len(removed)

    print()
    print(
        f"Summary: {total_fixed} file(s) patched, "
        f"{total_lines_removed} duplicate line(s) removed."
    )
    if args.dry_run:
        print("(dry-run mode — no files written)")


if __name__ == "__main__":
    main()
