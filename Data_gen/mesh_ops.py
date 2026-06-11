"""Unstructured FEM meshing via gmsh and radius-threshold zone/region assignment.

The mesh is a boundary-conforming unstructured triangulation of the disc meridional
cross-section, built with the gmsh Python API.  Element size is graded:
  - fine at fillet / transition zone boundaries  (LC_FILLET)
  - fine at the bore inner face                  (LC_BORE)
  - fine at the rim outer face                   (LC_RIM)
  - coarser in bulk web / interior regions       (LC_BULK)
The resulting node count varies with geometry (as in real FEM practice).
The :class:`MeshData` exposes a ``skfem.MeshTri`` used directly for the
axisymmetric FEA solve and for ML feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from skfem import MeshTri

from .config import ZONE_NAME_TO_ID, ZONE_TO_REGION, REGION_NAME_TO_ID

# ---------------------------------------------------------------------------
# Mesh size parameters (mm)
# ---------------------------------------------------------------------------
LC_BULK   = 2.0   # general interior / web bulk
LC_FILLET = 0.5   # fillet / transition zone boundaries
LC_BORE   = 1.0   # bore inner face (high hoop stress surface)
LC_RIM    = 1.2   # rim outer face

FILLET_INFLUENCE_MM = 4.0   # distance field extent around each fillet radius
BORE_INFLUENCE_MM   = 3.0   # distance field extent inward from bore inner radius
RIM_INFLUENCE_MM    = 3.0   # distance field extent outward from rim outer radius


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
    but are unused — mesh density is controlled by the LC_* constants.
    """
    import gmsh

    contour_points = np.asarray(contour_points, dtype=np.float64)

    if radial_breaks is None:
        r_min = float(contour_points[:, 1].min())
        r_max = float(contour_points[:, 1].max())
        radial_breaks = np.array([r_min, r_min, r_min, r_max, r_max, r_max])

    r0 = float(radial_breaks[0])   # bore inner radius
    r1 = float(radial_breaks[1])   # bore / lower_transition boundary
    r2 = float(radial_breaks[2])   # lower_transition / web boundary
    r3 = float(radial_breaks[3])   # web / upper_transition boundary
    r4 = float(radial_breaks[4])   # upper_transition / rim boundary
    r5 = float(radial_breaks[5])   # rim outer radius

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC_FILLET * 0.4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC_BULK)
    gmsh.model.add("disc_meridional")

    try:
        # ------------------------------------------------------------------
        # 1. Add contour points with a first-pass size hint
        # ------------------------------------------------------------------
        point_tags = []
        for x, r in contour_points:
            # Assign initial lc based on proximity to key radii
            if abs(r - r0) < BORE_INFLUENCE_MM:
                lc = LC_BORE
            elif abs(r - r5) < RIM_INFLUENCE_MM:
                lc = LC_RIM
            elif any(abs(r - rf) < FILLET_INFLUENCE_MM for rf in (r1, r2, r3, r4)):
                lc = LC_FILLET
            else:
                lc = LC_BULK
            tag = gmsh.model.geo.addPoint(x, r, 0.0, lc)
            point_tags.append(tag)

        # ------------------------------------------------------------------
        # 2. Closed boundary loop
        # ------------------------------------------------------------------
        n = len(point_tags)
        line_tags = []
        for i in range(n):
            tag = gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n])
            line_tags.append(tag)

        loop_tag    = gmsh.model.geo.addCurveLoop(line_tags)
        surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])
        gmsh.model.geo.synchronize()

        # ------------------------------------------------------------------
        # 3. Distance / Threshold fields for smooth size grading
        # ------------------------------------------------------------------
        all_threshold_ids = []

        def _add_threshold(pt_list, lc_min, dist_max):
            if not pt_list:
                return
            fid = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fid, "PointsList", pt_list)
            tid = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tid, "InField",  fid)
            gmsh.model.mesh.field.setNumber(tid, "SizeMin",  lc_min)
            gmsh.model.mesh.field.setNumber(tid, "SizeMax",  LC_BULK)
            gmsh.model.mesh.field.setNumber(tid, "DistMin",  0.0)
            gmsh.model.mesh.field.setNumber(tid, "DistMax",  dist_max)
            all_threshold_ids.append(tid)

        # Fillet zone boundaries (r1, r2, r3, r4)
        for rf in (r1, r2, r3, r4):
            pts = [
                point_tags[i]
                for i, (_, r) in enumerate(contour_points)
                if abs(r - rf) < FILLET_INFLUENCE_MM
            ]
            _add_threshold(pts, LC_FILLET, FILLET_INFLUENCE_MM * 2.0)

        # Bore inner face (r ~ r0)
        bore_pts = [
            point_tags[i]
            for i, (_, r) in enumerate(contour_points)
            if abs(r - r0) < BORE_INFLUENCE_MM
        ]
        _add_threshold(bore_pts, LC_BORE, BORE_INFLUENCE_MM * 2.0)

        # Rim outer face (r ~ r5)
        rim_pts = [
            point_tags[i]
            for i, (_, r) in enumerate(contour_points)
            if abs(r - r5) < RIM_INFLUENCE_MM
        ]
        _add_threshold(rim_pts, LC_RIM, RIM_INFLUENCE_MM * 2.0)

        if all_threshold_ids:
            if len(all_threshold_ids) > 1:
                min_fid = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(min_fid, "FieldsList", all_threshold_ids)
                gmsh.model.mesh.field.setAsBackgroundMesh(min_fid)
            else:
                gmsh.model.mesh.field.setAsBackgroundMesh(all_threshold_ids[0])

        # ------------------------------------------------------------------
        # 4. Generate and smooth
        # ------------------------------------------------------------------
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")

        # ------------------------------------------------------------------
        # 5. Extract nodes and triangles
        # ------------------------------------------------------------------
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords   = coords.reshape(-1, 3)
        points   = coords[:, :2].copy()   # [x, r]

        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_list = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:    # 3-node linear triangle
                tri_list.append(enodes.reshape(-1, 3))
            elif etype == 9:  # 6-node quadratic triangle — corners only
                tri_list.append(enodes.reshape(-1, 6)[:, :3])

        if not tri_list:
            raise RuntimeError("gmsh returned no triangular elements")

        triangles_raw = np.vstack(tri_list).astype(np.int64)
        triangles     = np.vectorize(tag_to_idx.__getitem__)(triangles_raw)

    finally:
        gmsh.finalize()

    # ------------------------------------------------------------------
    # 6. Compact: drop unreferenced nodes
    # ------------------------------------------------------------------
    used  = np.unique(triangles)
    remap = -np.ones(points.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    points    = points[used]
    triangles = remap[triangles]

    mesh           = MeshTri(points.T.copy(), triangles.T.astype(np.int64).copy())
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
    r  = nodes[:, 1]
    rb = radial_breaks
    zone_ids = np.empty(r.shape[0], dtype=np.int32)
    zone_ids[r <= rb[1]]                          = ZONE_NAME_TO_ID["bore"]
    zone_ids[(r > rb[1]) & (r <= rb[2])]          = ZONE_NAME_TO_ID["lower_transition"]
    zone_ids[(r > rb[2]) & (r <= rb[3])]          = ZONE_NAME_TO_ID["web"]
    zone_ids[(r > rb[3]) & (r <= rb[4])]          = ZONE_NAME_TO_ID["upper_transition"]
    zone_ids[r > rb[4]]                           = ZONE_NAME_TO_ID["rim"]
    region_ids = _region_from_zone(zone_ids)
    return zone_ids, region_ids
