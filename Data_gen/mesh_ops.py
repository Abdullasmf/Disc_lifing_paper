"""Unstructured FEM meshing via gmsh and radius-threshold zone/region assignment.

The mesh is a boundary-conforming unstructured triangulation of the disc meridional
cross-section, built with the gmsh Python API.  Element size is graded: fine near
the fillet/transition radii where stress concentrates, coarser in the bore, web and
rim bulks.  The resulting node count varies with geometry (as in real FEM practice).
The :class:`MeshData` exposes a ``skfem.MeshTri`` used directly for the axisymmetric
FEA solve and for ML feature extraction.
"""

from __future__ import annotations

import tempfile
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from skfem import MeshTri

from .config import ZONE_NAME_TO_ID, ZONE_TO_REGION, REGION_NAME_TO_ID

# ---------------------------------------------------------------------------
# Mesh size parameters (mm)
# ---------------------------------------------------------------------------
LC_BULK = 2.5       # element size in bore / web / rim bulk regions
LC_FILLET = 0.5     # element size near fillet / transition zones
FILLET_INFLUENCE_MM = 4.0  # radius around fillet points that gets fine mesh


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


def _fillet_points(contour_points: np.ndarray, radial_breaks: np.ndarray) -> np.ndarray:
    """Return contour points that lie close to the fillet transition radii."""
    r1, r2, r3, r4 = float(radial_breaks[1]), float(radial_breaks[2]), \
                     float(radial_breaks[3]), float(radial_breaks[4])
    fillet_radii = np.array([r1, r2, r3, r4])
    rs = contour_points[:, 1]
    mask = np.zeros(len(rs), dtype=bool)
    for rf in fillet_radii:
        mask |= np.abs(rs - rf) < FILLET_INFLUENCE_MM
    return contour_points[mask]


def generate_mesh(
    contour_points: np.ndarray,
    grid_x: int,
    grid_r: int,
    seed: int = 0,
    radial_breaks: Optional[np.ndarray] = None,
    geometry_params: Optional[dict] = None,
) -> MeshData:
    """Generate an unstructured boundary-conforming triangular mesh via gmsh.

    ``grid_x``, ``grid_r`` and ``seed`` are accepted for call-site compatibility
    but are unused — mesh density is controlled by ``LC_BULK`` and ``LC_FILLET``.
    """
    import gmsh

    contour_points = np.asarray(contour_points, dtype=np.float64)

    if radial_breaks is None:
        r_min = float(contour_points[:, 1].min())
        r_max = float(contour_points[:, 1].max())
        radial_breaks = np.array([r_min, r_min, r_min, r_max, r_max, r_max])

    # Identify which contour points are near fillets (get fine mesh size)
    fillet_pts = _fillet_points(contour_points, radial_breaks)
    fillet_set = set(map(tuple, np.round(fillet_pts, 6)))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)   # suppress output
    gmsh.option.setNumber("Mesh.Algorithm", 6)      # Frontal-Delaunay (6) — best for curved boundaries
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC_FILLET * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC_BULK)
    gmsh.model.add("disc_meridional")

    try:
        # --- Add all contour points -------------------------------------------
        point_tags = []
        for x, r in contour_points:
            lc = LC_FILLET if (round(x, 6), round(r, 6)) in fillet_set else LC_BULK
            tag = gmsh.model.geo.addPoint(x, r, 0.0, lc)
            point_tags.append(tag)

        # --- Connect into a closed loop (spline segments per zone boundary) ---
        n = len(point_tags)
        line_tags = []
        for i in range(n):
            tag = gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n])
            line_tags.append(tag)

        loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
        surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])

        gmsh.model.geo.synchronize()

        # --- Refine around fillet zones with a distance field -----------------
        r1, r2, r3, r4 = (float(radial_breaks[i]) for i in (1, 2, 3, 4))
        fillet_radii = [r1, r2, r3, r4]

        field_ids = []
        for rf in fillet_radii:
            # Collect point tags at this radius
            pts_at_r = [
                point_tags[i]
                for i, (_, r) in enumerate(contour_points)
                if abs(r - rf) < FILLET_INFLUENCE_MM
            ]
            if not pts_at_r:
                continue
            fid = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fid, "PointsList", pts_at_r)
            field_ids.append(fid)

        if field_ids:
            threshold_ids = []
            for fid in field_ids:
                tid = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(tid, "InField", fid)
                gmsh.model.mesh.field.setNumber(tid, "SizeMin", LC_FILLET)
                gmsh.model.mesh.field.setNumber(tid, "SizeMax", LC_BULK)
                gmsh.model.mesh.field.setNumber(tid, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(tid, "DistMax", FILLET_INFLUENCE_MM * 2.0)
                threshold_ids.append(tid)

            if len(threshold_ids) > 1:
                min_fid = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(min_fid, "FieldsList", threshold_ids)
                gmsh.model.mesh.field.setAsBackgroundMesh(min_fid)
            else:
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold_ids[0])

        # --- Generate 2-D mesh ------------------------------------------------
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")   # smooth for better element quality

        # --- Extract nodes and triangles --------------------------------------
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)
        points = coords[:, :2].copy()   # x, r (drop z=0)

        # gmsh node tags are 1-based and not necessarily contiguous
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_list = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:   # 3-node triangle
                tri_arr = enodes.reshape(-1, 3)
                tri_list.append(tri_arr)
            elif etype == 9:  # 6-node quadratic triangle — take corner nodes only
                tri_arr = enodes.reshape(-1, 6)[:, :3]
                tri_list.append(tri_arr)

        if not tri_list:
            raise RuntimeError("gmsh returned no triangular elements")

        triangles_raw = np.vstack(tri_list).astype(np.int64)
        triangles = np.vectorize(tag_to_idx.__getitem__)(triangles_raw)

    finally:
        gmsh.finalize()

    # --- Compact: drop unreferenced nodes ------------------------------------
    used = np.unique(triangles)
    remap = -np.ones(points.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    points = points[used]
    triangles = remap[triangles]

    mesh = MeshTri(points.T.copy(), triangles.T.astype(np.int64).copy())
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
    """Assign zone and region to every node directly from its radial coordinate."""
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
