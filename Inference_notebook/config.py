"""Inference workspace configuration.

Edit the paths below to match your checkpoint locations.
All paths are resolved relative to the repo root by default.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root detection (one level up from this file)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Trained model checkpoints  (zonal + edge setup)
# ---------------------------------------------------------------------------
POINTNET_CHECKPOINT: Path = (
    REPO_ROOT
    / "Zonal"
    / "Edge_zoneID"
    / "PointNetMLPJoint"
    / "Trained_models"
    / "pn_s_full_ln_pos12_585ccb7c.pt"
)

ARGENT_CHECKPOINT: Path = (
    REPO_ROOT
    / "Zonal"
    / "Edge_zoneID"
    / "ArGEnT_self_att_noSDF"
    / "Trained_models"
    / "argent_self_nosdf_s_712f89d4.pt"
)

# ---------------------------------------------------------------------------
# Model source folders (added to sys.path so pn_models / benchmarks can be
# imported without installation)
# ---------------------------------------------------------------------------
POINTNET_MODEL_DIR: Path = (
    REPO_ROOT / "Zonal" / "Edge_zoneID" / "PointNetMLPJoint"
)

ARGENT_MODEL_DIR: Path = (
    REPO_ROOT / "Zonal" / "Edge_zoneID" / "ArGEnT_self_att_noSDF"
)

# ---------------------------------------------------------------------------
# Lifing mode — this workspace only supports zonal
# ---------------------------------------------------------------------------
LIFING_MODE: str = "zonal"

# ---------------------------------------------------------------------------
# Default device — "cpu" is safest for notebook use; set "cuda" to use GPU
# ---------------------------------------------------------------------------
DEFAULT_DEVICE: str = "cpu"
