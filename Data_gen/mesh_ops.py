"""Meshing and radius-threshold zone/region assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay, cKDTree
from skfem import MeshTri

from .config import ZONE_NAME_TO_ID, ZONE_TO_REGION, REGION_NAME_TO_ID


JITTER_FACTOR = 0.32
# 0.32 keeps the cloud randomized but avoids frequent edge leakage in thin-web cases.


@dataclass
class MeshData:
    mesh: MeshTri
    nodes: np.ndarray
    triangles: np.ndarray
    boundary_node_ids: np.ndarray
    nearest_contour_index: np.ndarray
    distance_to_contour: np.ndarray


def _unique_rows(points: np.ndarray) -> np.ndarray:
    uniq, idx = np.unique(points, axis=0, return_index=True)
    return uniq[np.argsort(idx)]


def _build_surface_interpolants(contour_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build radial interpolants of front and rear surfaces from contour points."""
    front = contour_points[contour_points[:, 0] <= 0.0]
    rear = contour_points[contour_points[:, 0] >= 0.0]

    r_front = front[:, 1]
    r_rear = rear[:, 1]
    x_front = front[:, 0]
    x_rear = rear[:, 0]

    idx_f = np.argsort(r_front)
    idx_r = np.argsort(r_rear)
    r_front_sorted = r_front[idx_f]
    r_rear_sorted = r_rear[idx_r]
    x_front_sorted = x_front[idx_f]
    x_rear_sorted = x_rear[idx_r]

    r_unique = np.unique(np.concatenate([r_front_sorted, r_rear_sorted]))
    xf = np.interp(r_unique, r_front_sorted, x_front_sorted)
    xr = np.interp(r_unique, r_rear_sorted, x_rear_sorted)
    return r_unique, xf, xr


def _deterministic_jitter(index: np.ndarray, seed: int) -> np.ndarray:
    """Return deterministic pseudo-random values in [-1, 1] from index and seed."""
    phase = np.sin((index + 1.0) * 12.9898 + (float(seed) + 1.0) * 78.233) * 43758.5453123
    frac = phase - np.floor(phase)
    return 2.0 * frac - 1.0


def generate_mesh(
    contour_points: np.ndarray,
    grid_x: int,
    grid_r: int,
    seed: int = 0,
    radial_breaks: Optional[np.ndarray] = None,
) -> MeshData:
    """Generate a Delaunay mesh from the contour with optional transition-band refinement.

    radial_breaks: array [r0, r1, r2, r3, r4, r5].  When provided, extra interior
    points are sampled uniformly within the lower- and upper-transition radial bands
    so that the mesh is locally denser near the stress-concentrating shoulder regions.
    """
    poly = Path(contour_points)
    x_min, r_min = contour_points.min(axis=0)
    x_max, r_max = contour_points.max(axis=0)

    dx = (x_max - x_min) / max(grid_x - 1, 1)
    dr = (r_max - r_min) / max(grid_r - 1, 1)

    gx = np.linspace(x_min, x_max, grid_x)
    gr = np.linspace(r_min, r_max, grid_r)
    xx, rr = np.meshgrid(gx, gr, indexing="xy")
    candidates = np.column_stack([xx.ravel(), rr.ravel()])

    # Mild deterministic de-regularization while reducing edge over-jitter in thin sections.
    idx = np.arange(candidates.shape[0], dtype=np.float64)
    jx = _deterministic_jitter(idx, seed)
    jr = _deterministic_jitter(idx + 0.5 * candidates.shape[0], seed + 7919)
    candidates[:, 0] += JITTER_FACTOR * dx * jx
    candidates[:, 1] += JITTER_FACTOR * dr * jr

    interior = candidates[poly.contains_points(candidates)]
    interior_list = [interior]

    # Targeted refinement in transition bands – denser sampling near shoulder regions.
    if radial_breaks is not None and len(radial_breaks) >= 5:
        r_interp, x_front_interp, x_rear_interp = _build_surface_interpolants(contour_points)
        n_extra_r = max(grid_r // 4, 16)
        n_extra_eta = max(grid_x // 3, 16)
        n_surface_extra_r = max(grid_r // 5, 14)
        shoulder_fracs = np.array([0.02, 0.05, 0.10, 0.16, 0.23], dtype=np.float64)
        for r_start, r_end in [
            (float(radial_breaks[1]), float(radial_breaks[2])),
            (float(radial_breaks[3]), float(radial_breaks[4])),
        ]:
            r_strip = np.linspace(r_start, r_end, n_extra_r, endpoint=False) + 0.5 * (r_end - r_start) / max(n_extra_r, 1)
            eta = np.linspace(0.0, 1.0, n_extra_eta, endpoint=False) + 0.5 / max(n_extra_eta, 1)
            rr_strip, eta_strip = np.meshgrid(r_strip, eta, indexing="xy")
            x_front = np.interp(rr_strip.ravel(), r_interp, x_front_interp)
            x_rear = np.interp(rr_strip.ravel(), r_interp, x_rear_interp)
            extra_x = x_front + eta_strip.ravel() * (x_rear - x_front)
            extra_cands = np.column_stack([extra_x, rr_strip.ravel()])
            extra_inside = extra_cands[poly.contains_points(extra_cands)]
            if extra_inside.shape[0] > 0:
                interior_list.append(extra_inside)

            # Optional shoulder-focused refinement near outer surfaces inside transition bands.
            r_surf = np.linspace(r_start, r_end, n_surface_extra_r, endpoint=False) + 0.5 * (r_end - r_start) / max(n_surface_extra_r, 1)
            rr_surf, ff = np.meshgrid(r_surf, shoulder_fracs, indexing="xy")
            x_front_s = np.interp(rr_surf.ravel(), r_interp, x_front_interp)
            x_rear_s = np.interp(rr_surf.ravel(), r_interp, x_rear_interp)
            thickness = np.maximum(x_rear_s - x_front_s, 1e-9)
            shoulder_front = np.column_stack([x_front_s + ff.ravel() * thickness, rr_surf.ravel()])
            shoulder_rear = np.column_stack([x_rear_s - ff.ravel() * thickness, rr_surf.ravel()])
            shoulder_cands = np.vstack([shoulder_front, shoulder_rear])
            shoulder_inside = shoulder_cands[poly.contains_points(shoulder_cands)]
            if shoulder_inside.shape[0] > 0:
                interior_list.append(shoulder_inside)

    combined = np.vstack(interior_list)
    points = _unique_rows(np.vstack([contour_points, combined]))

    tri = Delaunay(points)
    triangles = tri.simplices
    centroids = points[triangles].mean(axis=1)
    triangles = triangles[poly.contains_points(centroids)]

    mesh = MeshTri(points.T, triangles.T)
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
