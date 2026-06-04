"""Backward-compatible wrapper around dataset_generator."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .dataset_generator import main
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.dataset_generator import main


if __name__ == "__main__":
    main()
