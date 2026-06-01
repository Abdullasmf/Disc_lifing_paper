"""Meshing and region transfer helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay, cKDTree
from skfem import MeshTri


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
    order = np.argsort(idx)
    return uniq[order]


def generate_mesh(
    contour_points: np.ndarray,
    grid_x: int,
    grid_r: int,
) -> MeshData:
    """Generate triangular mesh from contour and interior point cloud."""
    poly = Path(contour_points)
    x_min, r_min = contour_points.min(axis=0)
    x_max, r_max = contour_points.max(axis=0)

    gx = np.linspace(x_min, x_max, grid_x)
    gr = np.linspace(r_min, r_max, grid_r)
    xx, rr = np.meshgrid(gx, gr, indexing="xy")
    interior = np.column_stack([xx.ravel(), rr.ravel()])
    inside = poly.contains_points(interior)
    interior = interior[inside]

    points = np.vstack([contour_points, interior])
    points = _unique_rows(points)

    tri = Delaunay(points)
    triangles = tri.simplices
    centroids = points[triangles].mean(axis=1)
    keep = poly.contains_points(centroids)
    triangles = triangles[keep]

    mesh = MeshTri(points.T, triangles.T)
    boundary_nodes = mesh.boundary_nodes()

    tree = cKDTree(contour_points)
    d2c, i2c = tree.query(points, k=1)

    return MeshData(
        mesh=mesh,
        nodes=points,
        triangles=triangles,
        boundary_node_ids=np.asarray(boundary_nodes, dtype=np.int32),
        nearest_contour_index=i2c.astype(np.int32),
        distance_to_contour=d2c.astype(float),
    )


def assign_regions_from_contour(
    nearest_contour_index: np.ndarray,
    contour_region_ids: np.ndarray,
) -> np.ndarray:
    """Assign node region from nearest contour sample point."""
    return contour_region_ids[nearest_contour_index].astype(np.int32)

