"""Model loading and inference helpers.

Supports:
    PointNetMLPJoint  (zonal edge, Edge_zoneID ablation)
    ArGEnTDeepONet    (zonal edge, cross-attention, no SDF)

Public API
----------
load_pointnet_model(checkpoint_path, device)
load_argent_model(checkpoint_path, device)
predict_life_field(model, model_type, data, ckpt_stats, device)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent

_POINTNET_DIR = _REPO_ROOT / "Zonal" / "Edge_zoneID" / "PointNetMLPJoint"
_ARGENT_DIR   = _REPO_ROOT / "Zonal" / "Edge_zoneID" / "ArGEnT_self_att_noSDF"


# ---------------------------------------------------------------------------
# Isolated module loaders
#
# The two model folders both contain pn_models.py but their contents differ
# (ArGEnT's version lacks in_channels support in PointNet2Encoder2D).
# We use importlib to load each script from its exact path with a unique
# module name so they coexist on sys.modules without shadowing each other.
# ---------------------------------------------------------------------------

def _load_module_from_file(module_name: str, file_path: Path):
    """Load a Python source file as a module with a given name."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)

    # The module's own directory must be on sys.path so that its local imports
    # (e.g. `from pn_models import ...` inside benchmarks.py) resolve correctly.
    mod_dir = str(file_path.parent)
    inserted = False
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
        inserted = True

    sys.modules[module_name] = mod          # register before exec so circular imports work
    spec.loader.exec_module(mod)            # type: ignore[union-attr]

    if inserted:
        sys.path.remove(mod_dir)

    return mod


def _import_pointnet():
    """Import PointNetMLPJoint and its builder from the PointNet model folder."""
    pn_mod = _load_module_from_file(
        "_inf_pn_models", _POINTNET_DIR / "pn_models.py"
    )
    return pn_mod.PointNetMLPJoint, pn_mod.build_model_from_arch


def _import_argent():
    """Import ArGEnTDeepONet from the ArGEnT model folder.

    benchmarks.py does `from pn_models import ...` — we pre-load the ArGEnT
    pn_models.py under a distinct name so it doesn't conflict with the
    PointNet version already cached in sys.modules.
    """
    # Pre-load ArGEnT's pn_models with a unique module name so benchmarks.py
    # `from pn_models import …` finds the right version while ArGEnT dir is
    # temporarily on sys.path.
    _load_module_from_file("_inf_ag_pn_models", _ARGENT_DIR / "pn_models.py")
    # Register it under the bare name only during the benchmarks.py exec
    sys.modules.setdefault("pn_models", sys.modules["_inf_ag_pn_models"])

    bm_mod = _load_module_from_file(
        "_inf_ag_benchmarks", _ARGENT_DIR / "benchmarks.py"
    )
    return bm_mod.ArGEnTDeepONet


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: str | Path, device: str | torch.device) -> Dict:
    """Load a checkpoint and return its dict."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Check the paths in Inference_notebook/config.py."
        )
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    return ckpt


# ---------------------------------------------------------------------------
# PointNetMLPJoint
# ---------------------------------------------------------------------------

def load_pointnet_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple:
    """Load a PointNetMLPJoint checkpoint.

    Returns
    -------
    (model, ckpt_stats) where ckpt_stats contains the normalisation tensors
    (coord_center, coord_half_range, target_mean, target_std, extra_feat_stats).
    """
    _, build_model_from_arch = _import_pointnet()
    ckpt = _load_checkpoint(checkpoint_path, device)

    arch = ckpt.get("arch")
    if arch is None:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has no 'arch' key.  "
            "Cannot reconstruct model architecture."
        )

    model = build_model_from_arch(arch)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    ckpt_stats = _extract_norm_stats(ckpt)
    return model, ckpt_stats


# ---------------------------------------------------------------------------
# ArGEnT
# ---------------------------------------------------------------------------

def load_argent_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple:
    """Load an ArGEnTDeepONet checkpoint.

    Returns
    -------
    (model, ckpt_stats) where ckpt_stats contains the normalisation tensors.
    """
    ArGEnTDeepONet = _import_argent()
    ckpt = _load_checkpoint(checkpoint_path, device)

    arch = ckpt.get("arch")
    if arch is None:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has no 'arch' key."
        )

    model = ArGEnTDeepONet(**arch)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    ckpt_stats = _extract_norm_stats(ckpt)
    return model, ckpt_stats


# ---------------------------------------------------------------------------
# Normalisation stats extraction
# ---------------------------------------------------------------------------

def _extract_norm_stats(ckpt: Dict) -> Dict:
    """Pull normalisation tensors out of a checkpoint dict."""
    required = ("coord_center", "coord_half_range", "target_mean", "target_std")
    for key in required:
        if key not in ckpt:
            raise RuntimeError(f"Checkpoint is missing required key '{key}'.")

    return {
        "coord_center":     ckpt["coord_center"].cpu().float(),
        "coord_half_range": ckpt["coord_half_range"].cpu().float(),
        "target_mean":      ckpt["target_mean"].cpu().float(),
        "target_std":       ckpt["target_std"].cpu().float(),
        "extra_feat_stats": ckpt.get("extra_feat_stats", {2: {"mean": 0.0, "std": 4.0}}),
        "extra_feat_cols":  ckpt.get("extra_feat_cols", [2]),
    }


# ---------------------------------------------------------------------------
# Unified inference
# ---------------------------------------------------------------------------

def predict_life_field(
    model: nn.Module,
    model_type: Literal["pointnet", "argent"],
    data: Dict,
    ckpt_stats: Dict,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run model inference over the full disc mesh and return life values.

    The model predicts [Stress, log10(Life)] at each query (mesh) node.
    Only the life output is returned, converted from log10 to cycles.

    Parameters
    ----------
    model:
        Loaded model (output of :func:`load_pointnet_model` or
        :func:`load_argent_model`).
    model_type:
        ``"pointnet"`` or ``"argent"``.
    data:
        Output of :func:`generate_inference_data`.
    ckpt_stats:
        Normalisation stats from :func:`load_pointnet_model` /
        :func:`load_argent_model`.
    device:
        Torch device string or object.

    Returns
    -------
    life_field : np.ndarray, shape [M]
        Predicted life (cycles) at each mesh node.
    """
    # Import feature builders here to avoid circular imports at module level
    from inference_helpers import build_pointnet_features, build_argent_features  # type: ignore[import]

    device = torch.device(device)
    target_mean = ckpt_stats["target_mean"].to(device)
    target_std  = ckpt_stats["target_std"].to(device)

    model.eval()
    model.to(device)

    with torch.no_grad():
        if model_type == "pointnet":
            geom_xyz_norm, geom_feats_norm, query_norm = build_pointnet_features(
                data, ckpt_stats
            )
            pred_z = model(
                geom_xyz_norm.unsqueeze(0).to(device),   # [1, N, 2]
                query_norm.unsqueeze(0).to(device),      # [1, M, 2]
                geom_feats_norm.unsqueeze(0).to(device), # [1, N, 1]
            )  # [1, M, 2]  standardised (Stress, LogLife)

        elif model_type == "argent":
            geom_norm, query_norm = build_argent_features(data, ckpt_stats)
            pred_z = model(
                geom_norm.unsqueeze(0).to(device),   # [1, N, 3]
                query_norm.unsqueeze(0).to(device),  # [1, M, 2]
            )  # [1, M, 2]

        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'.  "
                "Use 'pointnet' or 'argent'."
            )

        # De-standardise: pred_raw = pred_z * std + mean
        pred_z = pred_z.squeeze(0)  # [M, 2]
        pred_raw = pred_z * target_std + target_mean  # [M, 2]  (Stress, LogLife)

        # Column 1 is log10(life); convert to cycles
        log_life = pred_raw[:, 1].cpu().numpy()
        life_field = np.power(10.0, log_life)

    return life_field.astype(np.float64)
