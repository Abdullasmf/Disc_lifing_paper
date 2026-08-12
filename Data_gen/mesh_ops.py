"""Unstructured FEM meshing via gmsh and radius-threshold zone/region assignment.

The mesh is a boundary-conforming unstructured triangulation of the disc meridional
cross-section, built with the gmsh Python API.  Element size is graded:
  - fine at fillet / transition zone boundaries  (LC_FILLET)
  - fine at the bore inner face                  (LC_BORE)
  - fine at the rim outer face                   (LC_RIM)
  - uniform shell around entire contour           (LC_EDGE)
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
LC_EDGE   = 0.50  # uniform shell around the entire contour boundary (medium production)
LC_FILLET = 0.30  # fillet / transition zone boundaries (medium production)
LC_BORE   = 1.0   # bore inner face (high hoop stress surface)
LC_RIM    = 1.2   # rim outer face

EDGE_INFLUENCE_MM   = 2.5   # depth of uniform boundary shell (just above edge_proximity_distance_mm)
FILLET_INFLUENCE_MM = 4.0   # distance field extent around each fillet radius
BORE_INFLUENCE_MM   = 3.0   # distance field extent inward from bore inner radius
RIM_INFLUENCE_MM    = 3.0   # distance field extent outward from rim outer radius

# Feature-specific refinement influence radii (mm)
FEAT_INFLUENCE_MM = 1.5    # neighbourhood for C-groove, ligament, arm features


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
    rim_feature_params: Optional[dict] = None,
) -> MeshData:
    """Generate an unstructured boundary-conforming triangular mesh via gmsh.

    ``grid_x``, ``grid_r`` and ``seed`` are accepted for call-site compatibility
    but are unused — mesh density is controlled by the LC_* constants.
    ``rim_feature_params``: sanitised rim-feature parameters (RIM_FEATURE_PARAMETERS
    keys).  When provided, named feature landmarks get individual Distance/Threshold
    refinement fields for the C-groove, visible ligament, rear arm root, arm outer
    corner, and arm end face.
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
    r5 = float(radial_breaks[5])   # rim outer radius (main cap)

    # r_max from the actual contour may exceed r5 when flanges are present.
    r_max_contour = float(contour_points[:, 1].max())

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
                lc = LC_EDGE  # default to edge size for all boundary points
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

        def _add_threshold(pt_list, lc_min, dist_max, curve_list=None):
            if not pt_list and not curve_list:
                return
            fid = gmsh.model.mesh.field.add("Distance")
            if pt_list:
                gmsh.model.mesh.field.setNumbers(fid, "PointsList", pt_list)
            if curve_list:
                gmsh.model.mesh.field.setNumbers(fid, "CurvesList", curve_list)
                gmsh.model.mesh.field.setNumber(fid, "Sampling", 20)
            tid = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tid, "InField",  fid)
            gmsh.model.mesh.field.setNumber(tid, "SizeMin",  lc_min)
            gmsh.model.mesh.field.setNumber(tid, "SizeMax",  LC_BULK)
            gmsh.model.mesh.field.setNumber(tid, "DistMin",  0.0)
            gmsh.model.mesh.field.setNumber(tid, "DistMax",  dist_max)
            all_threshold_ids.append(tid)

        # ------------------------------------------------------------------
        # 3a. Uniform boundary shell: LC_EDGE across ALL contour lines.
        #     This ensures bore/rim flat faces are as dense as the web walls.
        #     Influence depth = EDGE_INFLUENCE_MM (just above edge_proximity_distance_mm).
        # ------------------------------------------------------------------
        _add_threshold(
            pt_list=[],
            lc_min=LC_EDGE,
            dist_max=EDGE_INFLUENCE_MM,
            curve_list=line_tags,
        )

        # ------------------------------------------------------------------
        # 3b. Fillet zone boundaries (r1, r2, r3, r4) — kept finer than LC_EDGE
        #     for stress concentration resolution.
        # ------------------------------------------------------------------
        for rf in (r1, r2, r3, r4):
            pts = [
                point_tags[i]
                for i, (_, r) in enumerate(contour_points)
                if abs(r - rf) < FILLET_INFLUENCE_MM
            ]
            _add_threshold(pts, LC_FILLET, FILLET_INFLUENCE_MM * 2.0)

        # ------------------------------------------------------------------
        # 3c. Bore inner face (r ~ r0) — high hoop stress region.
        # ------------------------------------------------------------------
        bore_pts = [
            point_tags[i]
            for i, (_, r) in enumerate(contour_points)
            if abs(r - r0) < BORE_INFLUENCE_MM
        ]
        _add_threshold(bore_pts, LC_BORE, BORE_INFLUENCE_MM * 2.0)

        # ------------------------------------------------------------------
        # 3d. Rim outer face (r ~ r5) AND flange top (r ~ r_max_contour if flanges present).
        # ------------------------------------------------------------------
        rim_pts = [
            point_tags[i]
            for i, (_, r) in enumerate(contour_points)
            if abs(r - r5) < RIM_INFLUENCE_MM
        ]
        _add_threshold(rim_pts, LC_RIM, RIM_INFLUENCE_MM * 2.0)

        # 3e. Rim outer top — all arm/groove features reside above r5.
        #     Refine broadly around the entire outer feature region.
        if r_max_contour > r5 + 0.5:
            rim_feature_pts = [
                point_tags[i]
                for i, (x, r) in enumerate(contour_points)
                if r > r5 + 0.10
            ]
            _add_threshold(rim_feature_pts, LC_FILLET, FILLET_INFLUENCE_MM)

        # ------------------------------------------------------------------
        # 3f. Named rim feature landmarks: C-groove, ligament, arm.
        #     Each is a specific coordinate-based neighbourhood that gets
        #     LC_FILLET refinement within FEAT_INFLUENCE_MM.
        # ------------------------------------------------------------------
        if rim_feature_params is not None and geometry_params is not None:
            t_rim = float(geometry_params.get("rim_thickness", 20.0))
            x_front = -0.5 * t_rim
            x_rear = 0.5 * t_rim
            h_arm = float(rim_feature_params.get("rear_arm_radial_height", 5.0))
            cg_depth = float(rim_feature_params.get("front_cgroove_axial_depth", 4.0))
            cg_pos = float(rim_feature_params.get("front_cgroove_radial_pos", 0.8))
            cg_span = float(rim_feature_params.get("front_cgroove_radial_span", 3.0))
            arm_proj = float(rim_feature_params.get("rear_arm_axial_projection", 4.0))
            rf_root = float(rim_feature_params.get("rear_arm_root_radius", 0.6))
            rf_corner = float(rim_feature_params.get("rear_arm_outer_corner_radius", 0.6))

            r_arm_top = r5 + h_arm
            x_arm_end = x_rear + arm_proj

            # Named feature landmarks: [x, r] coordinates
            feature_landmarks = {
                "cgroove_entry":    (x_front, r5 + cg_pos),
                "cgroove_floor":    (x_front + cg_depth, r5 + cg_pos + 0.5 * cg_span),
                "cgroove_exit":     (x_front, r5 + cg_pos + cg_span),
                "ligament":         (0.5 * (x_front + x_rear), r_arm_top),
                "arm_root":         (x_rear, r_arm_top - 0.5 * rf_root),
                "arm_neck":         (x_rear, r5 + 0.5 * float(rim_feature_params.get("rear_arm_neck_thickness", 2.0))),
                "arm_outer_corner": (x_arm_end - 0.5 * rf_corner, r_arm_top - 0.5 * rf_corner),
                "arm_end_face":     (x_arm_end, 0.5 * (r5 + r_arm_top)),
            }

            for feat_name, (fx, fr) in feature_landmarks.items():
                feat_pts = [
                    point_tags[i]
                    for i, (px, pr) in enumerate(contour_points)
                    if ((px - fx)**2 + (pr - fr)**2) < FEAT_INFLUENCE_MM**2
                ]
                if feat_pts:
                    _add_threshold(feat_pts, LC_FILLET, FEAT_INFLUENCE_MM * 2.0)

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
