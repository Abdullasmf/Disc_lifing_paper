"""Meshing and contour-provenance transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay, cKDTree
from skfem import MeshTri

from .config import ZONE_NAME_TO_ID, ZONE_TO_REGION, REGION_NAME_TO_ID


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


def generate_mesh(
    contour_points: np.ndarray,
    grid_x: int,
    grid_r: int,
    seed: int = 0,
) -> MeshData:
    poly = Path(contour_points)
    x_min, r_min = contour_points.min(axis=0)
    x_max, r_max = contour_points.max(axis=0)

    rng = np.random.default_rng(seed)
    dx = (x_max - x_min) / max(grid_x - 1, 1)
    dr = (r_max - r_min) / max(grid_r - 1, 1)

    gx = np.linspace(x_min, x_max, grid_x)
    gr = np.linspace(r_min, r_max, grid_r)
    xx, rr = np.meshgrid(gx, gr, indexing="xy")
    candidates = np.column_stack([xx.ravel(), rr.ravel()])

    candidates[:, 0] += rng.uniform(-0.32 * dx, 0.32 * dx, size=candidates.shape[0])
    candidates[:, 1] += rng.uniform(-0.32 * dr, 0.32 * dr, size=candidates.shape[0])

    interior = candidates[poly.contains_points(candidates)]
    points = _unique_rows(np.vstack([contour_points, interior]))

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


def _weighted_majority(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.int32)
    for i in range(values.shape[0]):
        uniq, inv = np.unique(values[i], return_inverse=True)
        score = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(score, inv, weights[i])
        out[i] = int(uniq[np.argmax(score)])
    return out


def _region_from_zone(zone_ids: np.ndarray) -> np.ndarray:
    region_ids = np.empty_like(zone_ids, dtype=np.int32)
    for zone_name, zone_id in ZONE_NAME_TO_ID.items():
        region_ids[zone_ids == zone_id] = REGION_NAME_TO_ID[ZONE_TO_REGION[zone_name]]
    return region_ids


def assign_zone_and_region_from_contour(
    nodes: np.ndarray,
    contour_points: np.ndarray,
    contour_zone_ids: np.ndarray,
    k_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(contour_points)
    k = min(int(k_neighbors), contour_points.shape[0])
    distances, indices = tree.query(nodes, k=k)

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    weights = 1.0 / (distances + 1e-6)
    zone_votes = contour_zone_ids[indices]
    zone_ids = _weighted_majority(zone_votes, weights)
    region_ids = _region_from_zone(zone_ids)

    nearest_contour_index = indices[:, 0].astype(np.int32)
    nearest_distance_mm = distances[:, 0].astype(np.float64)
    return zone_ids, region_ids, nearest_contour_index, nearest_distance_mm
