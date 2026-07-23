"""Geometry generation and feature-assembly helpers for inference.

All heavy logic is here; the notebook calls simple wrappers.

Public API
----------
generate_inference_data(offsets, lifing_mode="zonal")
    Generate a full-disc FEM sample plus the edge representation needed
    as model encoder input — single FEM solve, no redundant work.

build_pointnet_features(data, ckpt_stats)
    Assemble normalised tensors for PointNetMLPJoint inference.

build_argent_features(data, ckpt_stats)
    Assemble normalised tensors for ArGEnT inference.

get_key_life_points(mesh_nodes, radial_breaks_mm)
    Return (name, index) pairs for the five canonical disc locations.

summarize_key_life_points(life_field, mesh_nodes, radial_breaks_mm)
    Return a dict of scalar life values at each canonical location.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so Data_gen can be imported from anywhere
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Data_gen.config import (
    PUBLIC_GEOMETRY_PARAMETERS,
    NOMINAL_GEOMETRY_MM,
    MIN_OFFSET_MM,
    MAX_OFFSET_MM,
    radial_stations_from_params,
)
from Data_gen.sample_generator import generate_sample
from Data_gen.features import (
    resample_contour_uniform_arc_length,
    contour_derivative_features,
)
from Data_gen.mesh_ops import assign_zone_and_region_from_radius


# ---------------------------------------------------------------------------
# Geometry generation
# ---------------------------------------------------------------------------

def generate_inference_data(
    offsets: Optional[Dict[str, float]] = None,
    lifing_mode: str = "zonal",
    seed: int = 0,
) -> Dict:
    """Generate a complete inference dataset for one geometry.

    Runs a single FEM solve (using the "full" mesh representation) to obtain
    mesh node coordinates with FEM-computed life, and derives the edge
    representation from the stored contour without a second FEM solve.

    Parameters
    ----------
    offsets:
        Per-parameter offsets from nominal geometry (mm).  Omit or pass None
        to use the nominal geometry.  Unknown keys raise ValueError.
    lifing_mode:
        "zonal" (default) — use zone-specific S-N curves.
    seed:
        Random seed forwarded to the mesh generator.

    Returns
    -------
    dict with keys:

    Encoder inputs (edge representation):
        edge_points       [N, 2]  (x, r) in mm
        edge_zone_id      [N]     integer zone IDs
        edge_arc_mm       [N]     cumulative arc-length (mm)
        edge_features     [N, 4]  tangent_x, tangent_r, curvature, curv_grad

    Query set (full mesh):
        mesh_nodes        [M, 2]  (x, r) in mm
        mesh_zone_id      [M]     integer zone IDs
        mesh_fem_life     [M]     FEM-computed life (cycles, raw)
        triangles         [T, 3]  triangle connectivity for visualization

    Geometry metadata:
        contour_points    [C, 2]  disc contour boundary
        contour_zone_id   [C]     zone IDs for contour
        radial_breaks_mm  [6]     radial zone boundaries
        geometry_params   dict    actual geometry (nominal + offsets, mm)
        param_offsets     dict    clipped offsets actually used
        lifing_mode       str
    """
    if offsets is None:
        offsets = {}

    # Single FEM solve — "full" representation returns mesh nodes with FEM life
    full_sample = generate_sample(
        param_offsets=offsets,
        representation="full",
        seed=seed,
        include_derivatives=False,
        include_debug_fields=False,
        lifing_mode=lifing_mode,
    )

    contour_pts = full_sample["contour_points_mm"]          # [C, 2]
    contour_arc = full_sample["contour_arc_length_mm"]       # [C]
    radial_breaks = full_sample["radial_breaks_mm"]          # [6]
    actual_params = full_sample["geometry_parameters_actual"]

    # ------------------------------------------------------------------
    # Build edge representation from contour (no second FEM solve)
    # ------------------------------------------------------------------
    n_edge = contour_pts.shape[0]  # match training: same number of edge pts
    edge_points, edge_arc = resample_contour_uniform_arc_length(
        points=contour_pts,
        arc_length_mm=contour_arc,
        n_samples=n_edge,
    )

    edge_zone, _ = assign_zone_and_region_from_radius(edge_points, radial_breaks)

    dfeat = contour_derivative_features(edge_points, edge_arc)
    edge_features = np.column_stack([
        dfeat["tangent"][:, 0],
        dfeat["tangent"][:, 1],
        dfeat["curvature"],
        dfeat["curvature_gradient"],
    ]).astype(np.float64)

    return {
        # Encoder inputs
        "edge_points":    edge_points,
        "edge_zone_id":   edge_zone,
        "edge_arc_mm":    edge_arc,
        "edge_features":  edge_features,
        # Query set
        "mesh_nodes":     full_sample["node_coords_mm"],
        "mesh_zone_id":   full_sample["zone_id"],
        "mesh_fem_life":  full_sample["life_raw"],
        "triangles":      full_sample["triangles"],
        # Geometry metadata
        "contour_points": contour_pts,
        "contour_zone_id": full_sample["contour_zone_id"],
        "radial_breaks_mm": radial_breaks,
        "geometry_params": actual_params,
        "param_offsets":   full_sample["param_offsets"],
        "lifing_mode":     lifing_mode,
    }


def nominal_offsets() -> Dict[str, float]:
    """Return a zero-offset dict (nominal geometry)."""
    return {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS}


# ---------------------------------------------------------------------------
# Feature assembly for PointNetMLPJoint
# ---------------------------------------------------------------------------

def build_pointnet_features(
    data: Dict,
    ckpt_stats: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalised tensors for PointNetMLPJoint inference.

    The Edge_zoneID ablation uses EXTRA_FEAT_COLS = [2] (zone_id only).
    Normalisation mirrors the dataset pipeline: coordinates are min-max
    normalised; zone_id is scaled by 1/4 (fixed scheme).

    Parameters
    ----------
    data:
        Output of :func:`generate_inference_data`.
    ckpt_stats:
        Dict with keys ``coord_center``, ``coord_half_range``,
        ``target_mean``, ``target_std``, ``extra_feat_stats``.

    Returns
    -------
    geom_xyz_norm   [N, 2]  normalised encoder coordinates
    geom_feats_norm [N, 1]  normalised zone_id feature
    query_norm      [M, 2]  normalised query coordinates
    """
    coord_center     = ckpt_stats["coord_center"].float()
    coord_half_range = ckpt_stats["coord_half_range"].float().clamp(min=1e-8)
    extra_feat_stats = ckpt_stats["extra_feat_stats"]

    # Encoder geometry (x, r) — min-max normalised
    geom_xyz = torch.tensor(data["edge_points"], dtype=torch.float32)
    geom_xyz_norm = (geom_xyz - coord_center) / coord_half_range

    # Zone_id feature — fixed scaling: mean=0, std=4
    zone_id = torch.tensor(data["edge_zone_id"], dtype=torch.float32)
    z_mean = float(extra_feat_stats[2]["mean"])
    z_std  = max(float(extra_feat_stats[2]["std"]), 1e-8)
    geom_feats_norm = ((zone_id - z_mean) / z_std).unsqueeze(-1)  # [N, 1]

    # Query points (full mesh nodes) — same coordinate normalisation
    query_pts = torch.tensor(data["mesh_nodes"], dtype=torch.float32)
    query_norm = (query_pts - coord_center) / coord_half_range

    return geom_xyz_norm, geom_feats_norm, query_norm


# ---------------------------------------------------------------------------
# Feature assembly for ArGEnT
# ---------------------------------------------------------------------------

def build_argent_features(
    data: Dict,
    ckpt_stats: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build normalised tensors for ArGEnT (cross-attention, no SDF).

    INPUT_COLS = [0, 1, 2] → geom_points is (x, r, zone_id) stacked.
    Column normalisation:
        col 0 (x):       (v - coord_center[0]) / coord_half_range[0]
        col 1 (r):       (v - coord_center[1]) / coord_half_range[1]
        col 2 (zone_id): v / 4.0   (fixed mean=0, std=4 scheme)
    Query points are always (x, r) with the same coordinate normalisation.

    Parameters
    ----------
    data:
        Output of :func:`generate_inference_data`.
    ckpt_stats:
        Dict with keys ``coord_center``, ``coord_half_range``,
        ``extra_feat_stats``.

    Returns
    -------
    geom_norm   [N, 3]  normalised encoder features  (x, r, zone_id)
    query_norm  [M, 2]  normalised query coordinates  (x, r)
    """
    coord_center     = ckpt_stats["coord_center"].float()
    coord_half_range = ckpt_stats["coord_half_range"].float().clamp(min=1e-8)
    extra_feat_stats = ckpt_stats["extra_feat_stats"]

    z_mean = float(extra_feat_stats[2]["mean"])
    z_std  = max(float(extra_feat_stats[2]["std"]), 1e-8)

    enc_mean = torch.tensor(
        [float(coord_center[0]), float(coord_center[1]), z_mean],
        dtype=torch.float32,
    )
    enc_std = torch.tensor(
        [float(coord_half_range[0]), float(coord_half_range[1]), z_std],
        dtype=torch.float32,
    ).clamp(min=1e-8)

    # Encoder: (x, r, zone_id)
    xyz  = torch.tensor(data["edge_points"],  dtype=torch.float32)  # [N, 2]
    z_id = torch.tensor(data["edge_zone_id"], dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    geom = torch.cat([xyz, z_id], dim=-1)  # [N, 3]
    geom_norm = (geom - enc_mean) / enc_std

    # Query: (x, r)
    query_pts  = torch.tensor(data["mesh_nodes"], dtype=torch.float32)
    query_norm = (query_pts - coord_center) / coord_half_range

    return geom_norm, query_norm


# ---------------------------------------------------------------------------
# Key life-point extraction
# ---------------------------------------------------------------------------

_KEY_ZONE_MIDPOINTS = [
    ("bore",              0),   # zone_id 0
    ("lower_transition",  1),   # zone_id 1
    ("web",               2),   # zone_id 2
    ("upper_transition",  3),   # zone_id 3
    ("rim",               4),   # zone_id 4
]


def get_key_life_points(
    mesh_nodes: np.ndarray,
    radial_breaks_mm: np.ndarray,
) -> Dict[str, int]:
    """Return a dict mapping location name → mesh node index.

    Each representative point is defined as the node whose radial coordinate
    is closest to the midpoint of its zone's radial span AND whose axial (x)
    coordinate is smallest in absolute value (nearest to disc centre-plane).

    Includes:
        bore, lower_transition, web, upper_transition, rim  — zone midpoints
        global_min_life — not included here (requires life field, see below)

    Parameters
    ----------
    mesh_nodes:       [M, 2] node coordinates (x, r) in mm
    radial_breaks_mm: [6]   radial zone boundaries

    Returns
    -------
    dict mapping name → index into mesh_nodes
    """
    rb = radial_breaks_mm
    # Radial midpoints for each zone
    zone_mid_r = {
        "bore":              0.5 * (rb[0] + rb[1]),
        "lower_transition":  0.5 * (rb[1] + rb[2]),
        "web":               0.5 * (rb[2] + rb[3]),
        "upper_transition":  0.5 * (rb[3] + rb[4]),
        "rim":               0.5 * (rb[4] + rb[5]),
    }

    indices: Dict[str, int] = {}
    for name, r_mid in zone_mid_r.items():
        # Score: distance in r-space + small penalty for x deviation from 0
        dr = np.abs(mesh_nodes[:, 1] - r_mid)
        dx = np.abs(mesh_nodes[:, 0])
        # Weight x deviation lightly so we prefer near-centreline nodes
        score = dr + 0.05 * dx
        indices[name] = int(np.argmin(score))

    return indices


def summarize_key_life_points(
    life_field: np.ndarray,
    mesh_nodes: np.ndarray,
    radial_breaks_mm: np.ndarray,
) -> Dict[str, float]:
    """Compute scalar life figures at canonical disc locations.

    Parameters
    ----------
    life_field:       [M] predicted (or FEM) life in cycles
    mesh_nodes:       [M, 2]
    radial_breaks_mm: [6]

    Returns
    -------
    dict with keys bore, lower_transition, web, upper_transition, rim,
    global_min_life, global_min_r, global_min_x
    """
    key_indices = get_key_life_points(mesh_nodes, radial_breaks_mm)

    result: Dict[str, float] = {}
    for name, idx in key_indices.items():
        result[name] = float(life_field[idx])

    gmin_idx = int(np.argmin(life_field))
    result["global_min_life"] = float(life_field[gmin_idx])
    result["global_min_r"]    = float(mesh_nodes[gmin_idx, 1])
    result["global_min_x"]    = float(mesh_nodes[gmin_idx, 0])

    return result


# ---------------------------------------------------------------------------
# Parameter sweep helper
# ---------------------------------------------------------------------------

def sweep_single_parameter(
    param_name: str,
    sweep_values,
    predict_fn,
    fixed_offsets: Optional[Dict[str, float]] = None,
    lifing_mode: str = "zonal",
    verbose: bool = True,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Sweep one geometry offset parameter over a range.

    For each value in ``sweep_values`` the geometry is regenerated, model
    inference is run, and key-life figures are extracted.

    Parameters
    ----------
    param_name:
        Name of the offset to vary (must be in PUBLIC_GEOMETRY_PARAMETERS).
    sweep_values:
        Iterable of offset values (mm) to evaluate.
    predict_fn:
        Callable ``predict_fn(data) -> dict[model_name -> np.ndarray life_field]``.
        The function is called once per sweep step.
    fixed_offsets:
        Offsets for all parameters that are NOT being swept.  Defaults to
        all-zero (nominal).
    lifing_mode:
        Passed through to :func:`generate_inference_data`.
    verbose:
        Print progress if True.

    Returns
    -------
    pandas.DataFrame with columns:
        param_name, model, bore, lower_transition, web, upper_transition,
        rim, global_min_life, global_min_r, global_min_x
    """
    import pandas as pd  # local import to keep the module importable without pandas

    if param_name not in PUBLIC_GEOMETRY_PARAMETERS:
        raise ValueError(
            f"Unknown parameter '{param_name}'.  "
            f"Valid choices: {list(PUBLIC_GEOMETRY_PARAMETERS)}"
        )
    if fixed_offsets is None:
        fixed_offsets = {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS}

    records = []
    sweep_values = list(sweep_values)
    for i, v in enumerate(sweep_values):
        if verbose:
            print(f"  [{i+1}/{len(sweep_values)}]  {param_name} offset = {v:+.4f} mm")

        offsets = dict(fixed_offsets)
        offsets[param_name] = float(v)

        data = generate_inference_data(offsets, lifing_mode=lifing_mode)
        life_fields = predict_fn(data)

        for model_name, life_field in life_fields.items():
            key_lives = summarize_key_life_points(
                life_field,
                data["mesh_nodes"],
                data["radial_breaks_mm"],
            )
            record = {param_name: v, "model": model_name, **key_lives}
            records.append(record)

    return pd.DataFrame(records)
