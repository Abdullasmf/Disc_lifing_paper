"""Deterministic FEM meshing (scikit-fem P2) and radius-threshold zone/region assignment.

The mesh is a clean structured-Delaunay triangulation of the disc meridional
cross-section, built directly from the parametric front/rear thickness profile so
element edges align with the true boundary.  It is fully deterministic given the
geometry parameters (no random jitter, no seed dependence) and is locally refined
in the lower- and upper-transition radial bands where stress concentrates.  The
resulting :class:`MeshData` exposes a ``skfem.MeshTri`` used directly for the
axisymmetric FEA solve and for ML feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay, cKDTree
from skfem import MeshTri

from .config import ZONE_NAME_TO_ID, ZONE_TO_REGION, REGION_NAME_TO_ID

# Base number of radial sampling levels across the full disc span and the number
# of through-thickness nodes per level.  These set the nominal (coarse) density;
# transition bands receive a refinement multiplier on the radial spacing.
BASE_RADIAL_LEVELS = 110
THICKNESS_NODES = 11
TRANSITION_REFINE_FACTOR = 3  # transition bands get this many extra radial levels per base step


@dataclass
class MeshData:
    mesh: MeshTri
    nodes: np.ndarray
    triangles: np.ndarray
    boundary_node_ids: np.ndarray
    nearest_contour_index: np.ndarray
    distance_to_contour: np.ndarray


def _unique_rows(points: np.ndarray) -> np.ndarray:
    uniq, idx = np.unique(np.round(points, 9), axis=0, return_index=True)
    order = np.argsort(idx)
    return points[idx[order]]


def _thickness_at(r: np.ndarray, params: dict, radial_breaks: np.ndarray) -> np.ndarray:
    from .geometry import _thickness_profile
    return _thickness_profile(r, params, radial_breaks)


def _radial_levels(radial_breaks: np.ndarray) -> np.ndarray:
    """Deterministic, monotonically increasing radial sampling stations.

    Levels are spread roughly uniformly across the whole span, then the lower
    ([r1,r2]) and upper ([r3,r4]) transition bands are densified by inserting
    additional stations so those regions are finer than bore/web/rim.
    """
    r0 = float(radial_breaks[0])
    r5 = float(radial_breaks[5])
    r1, r2, r3, r4 = (float(radial_breaks[i]) for i in (1, 2, 3, 4))

    base = np.linspace(r0, r5, BASE_RADIAL_LEVELS)

    span = max(r5 - r0, 1e-9)
    extra = []
    for r_start, r_end in ((r1, r2), (r3, r4)):
        band_frac = max(r_end - r_start, 1e-9) / span
        n_band_base = max(int(round(band_frac * BASE_RADIAL_LEVELS)), 2)
        n_refined = n_band_base * TRANSITION_REFINE_FACTOR
        extra.append(np.linspace(r_start, r_end, n_refined))

    levels = np.concatenate([base] + extra)
    levels = np.unique(np.round(levels, 9))
    return levels


def generate_mesh(
    contour_points: np.ndarray,
    grid_x: int,
    grid_r: int,
    seed: int = 0,
    radial_breaks: Optional[np.ndarray] = None,
    geometry_params: Optional[dict] = None,
) -> MeshData:
    """Generate a deterministic P2-ready triangular FEM mesh of the cross-section.

    The mesh is built from the parametric front/rear surfaces of the disc (the
    profile is symmetric about x=0), so the boundary is represented exactly.  The
    ``seed``, ``grid_x`` and ``grid_r`` arguments are accepted for call-site
    compatibility but do not introduce any randomness — the mesh is a pure
    function of the geometry.  When ``geometry_params`` is not supplied the
    thickness profile is recovered from the contour envelope.
    """
    contour_points = np.asarray(contour_points, dtype=np.float64)
    poly = Path(contour_points)

    if radial_breaks is None:
        # Fallback: derive a span from the contour extent.
        r_min = float(contour_points[:, 1].min())
        r_max = float(contour_points[:, 1].max())
        radial_breaks = np.array([r_min, r_min, r_min, r_max, r_max, r_max], dtype=np.float64)

    levels = _radial_levels(radial_breaks)

    if geometry_params is not None:
        half_t = 0.5 * np.maximum(_thickness_at(levels, geometry_params, radial_breaks), 1e-9)
    else:
        # Recover thickness from contour envelope at each radial level.
        tree_r = contour_points[:, 1]
        half_t = np.empty_like(levels)
        for i, r in enumerate(levels):
            near = contour_points[np.abs(tree_r - r) <= (levels[-1] - levels[0]) / len(levels) + 1e-9]
            if near.shape[0] >= 2:
                half_t[i] = 0.5 * (near[:, 0].max() - near[:, 0].min())
            else:
                half_t[i] = 1e-9
        half_t = np.maximum(half_t, 1e-9)

    # Build structured node rows across the thickness at each radial level.
    eta = np.linspace(-1.0, 1.0, THICKNESS_NODES)
    node_list = []
    for r, ht in zip(levels, half_t):
        xs = eta * ht
        rs = np.full_like(xs, r)
        node_list.append(np.column_stack([xs, rs]))
    points = np.vstack(node_list)
    points = _unique_rows(points)

    tri = Delaunay(points)
    triangles = tri.simplices
    centroids = points[triangles].mean(axis=1)
    triangles = triangles[poly.contains_points(centroids)]

    # Drop nodes that ended up unreferenced after culling, then compact indices.
    used = np.unique(triangles)
    remap = -np.ones(points.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    points = points[used]
    triangles = remap[triangles]

    mesh = MeshTri(points.T, triangles.T.astype(np.int64))
    boundary_nodes = np.asarray(mesh.boundary_nodes(), dtype=np.int32)

    tree = cKDTree(contour_points)
    distance_to_contour, nearest_contour_index = tree.query(points, k=1)

    return MeshData(
        mesh=mesh,
        nodes=points.astype(np.float64),
        triangles=triangles.astype(np.int32),
        boundary_node_ids=boundary_nodes,
        nearest_contour_index=nearest_contour_index.astype(np.int32),
        distance_to_contour=distance_to_contour.astype(np.float64),
    )


def _region_from_zone(zone_ids: np.ndarray) -> np.ndarray:
    lookup = np.array([
        REGION_NAME_TO_ID[ZONE_TO_REGION[name]]
        for name, _ in sorted(ZONE_NAME_TO_ID.items(), key=lambda item: item[1])
    ], dtype=np.int32)
    return lookup[zone_ids]


def assign_zone_and_region_from_radius(
    nodes: np.ndarray,
    radial_breaks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign zone and region to every node directly from its radial coordinate.

    Every node's zone is determined solely by comparing its r-coordinate against
    the radial break thresholds [r0, r1, r2, r3, r4, r5].  No nearest-contour
    voting is performed.

    Parameters
    ----------
    nodes:
        (N, 2) array of node coordinates [x, r].
    radial_breaks:
        1-D array of 6 radial station values [r0, r1, r2, r3, r4, r5].

    Returns
    -------
    zone_ids, region_ids : (N,) int32 arrays
    """
    r = nodes[:, 1]
    rb = radial_breaks
    zone_ids = np.empty(r.shape[0], dtype=np.int32)
    zone_ids[r <= rb[1]] = ZONE_NAME_TO_ID["bore"]
    zone_ids[(r > rb[1]) & (r <= rb[2])] = ZONE_NAME_TO_ID["lower_transition"]
    zone_ids[(r > rb[2]) & (r <= rb[3])] = ZONE_NAME_TO_ID["web"]
    zone_ids[(r > rb[3]) & (r <= rb[4])] = ZONE_NAME_TO_ID["upper_transition"]
    zone_ids[r > rb[4]] = ZONE_NAME_TO_ID["rim"]
    region_ids = _region_from_zone(zone_ids)
    return zone_ids, region_ids
