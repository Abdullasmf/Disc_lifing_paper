"""Single-sample deterministic generator layer."""

from __future__ import annotations


import numpy as np
from scipy.spatial import cKDTree

from .config import (
    CYCLE_SPEED_FACTORS,
    FLANGE_GEOMETRY_PARAMETERS,
    REPRESENTATIONS,
    SUBZONE_NAME_TO_ID,
    SampleGenerationConfig,
    clip_flange_offsets_to_bounds,
    clip_offsets_to_bounds,
    resolve_flange_parameters,
    resolve_geometry_parameters,
)
from .features import contour_derivative_features, empty_features, resample_contour_uniform_arc_length
from .geometry import build_disc_contour, sanitize_flange_parameters, sanitize_geometry_parameters
from .mesh_ops import assign_zone_and_region_from_radius, generate_mesh
from .physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max


EDGE_DUPLICATE_EPS_MM = 1e-8


def _phase_stress_from_base_vm(base_vm: np.ndarray) -> np.ndarray:
    """Scale a base (takeoff) von Mises field to all 7 flight phases.

    Mirrors the scaling used inside the FEM solver so that points sampled off the
    FEM mesh (contour / edge) carry a consistent phase-stress matrix.
    """
    phase_scale = CYCLE_SPEED_FACTORS ** 2
    return (base_vm[:, None] * phase_scale[None, :]).astype(np.float64)


def _targets_from_fem_field(
    base_vm_mesh: np.ndarray,
    mesh_nodes: np.ndarray,
    query_nodes: np.ndarray,
    zone_ids: np.ndarray,
    geometry_params: dict[str, float],
    radial_breaks: np.ndarray,
    lifing_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the FEM von Mises field to ``query_nodes`` and derive targets.

    The axisymmetric FEM stress is solved once on the mesh; off-mesh sample points
    (contour vertices, arc-length-resampled edge points) take the nearest mesh
    node's base von Mises, then phase stress / life are recomputed consistently.
    """
    if query_nodes.shape[0] == 0:
        empty = np.empty((0, CYCLE_SPEED_FACTORS.shape[0]), dtype=np.float64)
        return empty, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if not np.any(base_vm_mesh):
        n = query_nodes.shape[0]
        n_phases = CYCLE_SPEED_FACTORS.shape[0]
        empty_stress = np.zeros((n, n_phases), dtype=np.float64)
        empty_scalar = np.zeros(n, dtype=np.float64)
        return empty_stress, empty_scalar, empty_scalar

    tree = cKDTree(mesh_nodes)
    _, nearest = tree.query(query_nodes, k=1)
    base_vm = base_vm_mesh[nearest]

    phase_stress = _phase_stress_from_base_vm(base_vm)
    stress_max_vm = compute_stress_max(phase_stress)
    life_raw = compute_life_raw(
        phase_stress=phase_stress,
        zone_ids=zone_ids,
        nodes=query_nodes,
        geometry_params=geometry_params,
        radial_breaks=radial_breaks,
        lifing_mode=lifing_mode,
    )
    return phase_stress, stress_max_vm, life_raw


def generate_sample(
    param_offsets: dict[str, float],
    representation: str,
    seed: int = 0,
    include_derivatives: bool = True,
    include_debug_fields: bool = False,
    lifing_mode: str = "zonal",
    flange_param_offsets: dict[str, float] | None = None,
    use_flanges: bool = True,
) -> dict:
    """Generate one complete deterministic sample from one offset vector.

    Parameters
    ----------
    param_offsets : dict
        Offsets from nominal for the 11 core geometry parameters.
    representation : str
        One of ``"edge"``, ``"edge_proximity"``, or ``"full"``.
    seed : int
        Random seed (used for mesh generation reproducibility).
    include_derivatives : bool
        Whether to compute tangent/curvature edge features.
    include_debug_fields : bool
        Whether to include distance-to-contour debug arrays.
    lifing_mode : str
        ``"zonal"`` or ``"uniform"`` S-N law selection.
    flange_param_offsets : dict or None
        Offsets from nominal for the 10 flange geometry parameters.
        If *None*, flanges are generated at their nominal values.
        Pass an empty dict ``{}`` to use all-nominal flange geometry.
        Only used when ``use_flanges=True`` (the default).
    use_flanges : bool
        If *True* (default), generate front/rear flanges on the outer rim.
        If *False*, use the legacy flat outer cap (no flanges).
        This flag is primarily intended for validation/comparison runs.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    cfg = SampleGenerationConfig()
    clipped_offsets = clip_offsets_to_bounds(param_offsets)
    actual_params = sanitize_geometry_parameters(resolve_geometry_parameters(clipped_offsets))

    # Flange parameters: resolve only when use_flanges=True.
    if use_flanges:
        clipped_flange_offsets = clip_flange_offsets_to_bounds(flange_param_offsets or {})
        raw_flange_params = resolve_flange_parameters(clipped_flange_offsets)
        actual_flange_params = sanitize_flange_parameters(raw_flange_params, t_rim=actual_params["rim_thickness"])
        _build_flange_params: dict | None = actual_flange_params
    else:
        clipped_flange_offsets = {k: 0.0 for k in FLANGE_GEOMETRY_PARAMETERS}
        actual_flange_params = {k: 0.0 for k in FLANGE_GEOMETRY_PARAMETERS}
        _build_flange_params = None   # legacy flat outer cap

    contour = build_disc_contour(
        actual_params,
        points_per_side=cfg.contour_points_per_side,
        flange_params=_build_flange_params,
    )
    radial_breaks = contour.metadata["radial_breaks_mm"]

    mesh = generate_mesh(
        contour_points=contour.points,
        grid_x=cfg.mesh_grid_points_x,
        grid_r=cfg.mesh_grid_points_r,
        seed=int(seed),
        radial_breaks=radial_breaks,
        geometry_params=actual_params,
    )

    # Zone and region assigned purely from radius thresholds – no nearest-contour voting.
    mesh_zone_id, mesh_region_id = assign_zone_and_region_from_radius(
        nodes=mesh.nodes,
        radial_breaks=radial_breaks,
    )
    nearest_idx = mesh.nearest_contour_index
    distance_to_contour = mesh.distance_to_contour

    # Single axisymmetric FEM solve on the mesh; phases scaled inside the solver.
    mesh_phase_stress = compute_phase_equivalent_stresses(
        nodes=mesh.nodes,
        zone_ids=mesh_zone_id,
        region_ids=mesh_region_id,
        geometry_params=actual_params,
        radial_breaks=radial_breaks,
        mesh_obj=mesh.mesh,
        triangles=mesh.triangles,
    )
    fem_failed = not np.any(mesh_phase_stress)  # True if all-zero (FEM failure)
    mesh_stress_max_vm = compute_stress_max(mesh_phase_stress)
    if fem_failed:
        mesh_life_raw = np.zeros(mesh_phase_stress.shape[0], dtype=np.float64)
    else:
        mesh_life_raw = compute_life_raw(
            phase_stress=mesh_phase_stress,
            zone_ids=mesh_zone_id,
            nodes=mesh.nodes,
            geometry_params=actual_params,
            radial_breaks=radial_breaks,
            lifing_mode=lifing_mode,
        )
    # Base (takeoff, speed_factor=1) von Mises field used to interpolate to any
    # off-mesh sample points (contour / edge representations).
    takeoff_idx = int(np.argmax(CYCLE_SPEED_FACTORS))
    base_vm_mesh = mesh_phase_stress[:, takeoff_idx]

    contour_zone_id, contour_region_id = assign_zone_and_region_from_radius(
        nodes=contour.points,
        radial_breaks=radial_breaks,
    )
    contour_phase_stress, contour_stress_max_vm, contour_life_raw = _targets_from_fem_field(
        base_vm_mesh=base_vm_mesh,
        mesh_nodes=mesh.nodes,
        query_nodes=contour.points,
        zone_ids=contour_zone_id,
        geometry_params=actual_params,
        radial_breaks=radial_breaks,
        lifing_mode=lifing_mode,
    )

    if representation == "edge":
        edge_points, edge_arc = resample_contour_uniform_arc_length(
            points=contour.points,
            arc_length_mm=contour.arc_length_mm,
            n_samples=contour.points.shape[0],
        )
        edge_zone, edge_region = assign_zone_and_region_from_radius(
            nodes=edge_points,
            radial_breaks=radial_breaks,
        )
        # Assign subzone for edge points via nearest-contour lookup.
        from scipy.spatial import cKDTree as _KD
        _tree = _KD(contour.points)
        _, _nn = _tree.query(edge_points, k=1)
        edge_subzone = contour.subzone_ids[_nn].astype(np.int32)

        edge_phase_stress, edge_stress_max_vm, edge_life_raw = _targets_from_fem_field(
            base_vm_mesh=base_vm_mesh,
            mesh_nodes=mesh.nodes,
            query_nodes=edge_points,
            zone_ids=edge_zone,
            geometry_params=actual_params,
            radial_breaks=radial_breaks,
            lifing_mode=lifing_mode,
        )

        if include_derivatives:
            dfeat = contour_derivative_features(edge_points, edge_arc)
            node_features = np.column_stack(
                [
                    dfeat["tangent"][:, 0],
                    dfeat["tangent"][:, 1],
                    dfeat["curvature"],
                    dfeat["curvature_gradient"],
                ]
            ).astype(np.float64)
            node_feature_names = np.array(
                ["tangent_x", "tangent_r", "curvature", "curvature_gradient"],
                dtype="S64",
            )
        else:
            node_features, node_feature_names = empty_features(edge_points.shape[0])

        out = {
            "param_offsets": {k: float(v) for k, v in clipped_offsets.items()},
            "geometry_parameters_actual": {k: float(v) for k, v in actual_params.items()},
            "flange_param_offsets": {k: float(v) for k, v in clipped_flange_offsets.items()},
            "flange_parameters_actual": {k: float(v) for k, v in actual_flange_params.items()},
            "representation": representation,
            "node_coords_mm": edge_points,
            "zone_id": edge_zone,
            "region_id": edge_region,
            "subzone_id": edge_subzone,
            "stress_max_vm": edge_stress_max_vm,
            "life_raw": edge_life_raw,
            "phase_stress_eq": edge_phase_stress,
            "node_features": node_features,
            "node_feature_names": node_feature_names,
            "arc_length_mm": edge_arc,
            "radial_breaks_mm": radial_breaks.astype(np.float64),
        }
    elif representation == "edge_proximity":
        keep = (distance_to_contour <= cfg.edge_proximity_distance_mm) & (distance_to_contour > EDGE_DUPLICATE_EPS_MM)

        interior_nodes = mesh.nodes[keep]
        interior_zone = mesh_zone_id[keep]
        interior_region = mesh_region_id[keep]
        interior_stress = mesh_stress_max_vm[keep]
        interior_life = mesh_life_raw[keep]
        interior_phase = mesh_phase_stress[keep]

        node_coords = np.vstack([contour.points, interior_nodes])
        zone_id = np.concatenate([contour_zone_id, interior_zone])
        region_id = np.concatenate([contour_region_id, interior_region])
        # Subzone: contour uses contour.subzone_ids; interior nodes use nearest-contour lookup.
        from scipy.spatial import cKDTree as _KD2
        _tree2 = _KD2(contour.points)
        _, _nn2 = _tree2.query(interior_nodes, k=1)
        interior_subzone = contour.subzone_ids[_nn2].astype(np.int32)
        subzone_id = np.concatenate([contour.subzone_ids, interior_subzone])
        stress = np.concatenate([contour_stress_max_vm, interior_stress])
        life = np.concatenate([contour_life_raw, interior_life])
        phase = np.vstack([contour_phase_stress, interior_phase])
        arc = np.concatenate([
            contour.arc_length_mm,
            np.full(interior_nodes.shape[0], np.nan, dtype=np.float64),
        ])
        node_features, node_feature_names = empty_features(node_coords.shape[0])

        out = {
            "param_offsets": {k: float(v) for k, v in clipped_offsets.items()},
            "geometry_parameters_actual": {k: float(v) for k, v in actual_params.items()},
            "flange_param_offsets": {k: float(v) for k, v in clipped_flange_offsets.items()},
            "flange_parameters_actual": {k: float(v) for k, v in actual_flange_params.items()},
            "representation": representation,
            "node_coords_mm": node_coords,
            "zone_id": zone_id,
            "region_id": region_id,
            "subzone_id": subzone_id,
            "stress_max_vm": stress,
            "life_raw": life,
            "phase_stress_eq": phase,
            "node_features": node_features,
            "node_feature_names": node_feature_names,
            "arc_length_mm": arc,
            "radial_breaks_mm": radial_breaks.astype(np.float64),
        }
    else:
        node_features, node_feature_names = empty_features(mesh.nodes.shape[0])
        # For full-mesh representation, subzone via nearest-contour.
        from scipy.spatial import cKDTree as _KD3
        _tree3 = _KD3(contour.points)
        _, _nn3 = _tree3.query(mesh.nodes, k=1)
        mesh_subzone_id = contour.subzone_ids[_nn3].astype(np.int32)
        out = {
            "param_offsets": {k: float(v) for k, v in clipped_offsets.items()},
            "geometry_parameters_actual": {k: float(v) for k, v in actual_params.items()},
            "flange_param_offsets": {k: float(v) for k, v in clipped_flange_offsets.items()},
            "flange_parameters_actual": {k: float(v) for k, v in actual_flange_params.items()},
            "representation": representation,
            "node_coords_mm": mesh.nodes,
            "zone_id": mesh_zone_id,
            "region_id": mesh_region_id,
            "subzone_id": mesh_subzone_id,
            "stress_max_vm": mesh_stress_max_vm,
            "life_raw": mesh_life_raw,
            "phase_stress_eq": mesh_phase_stress,
            "node_features": node_features,
            "node_feature_names": node_feature_names,
            "radial_breaks_mm": radial_breaks.astype(np.float64),
        }
        if include_debug_fields:
            out["distance_to_contour_mm"] = distance_to_contour.astype(np.float64)
            out["nearest_contour_index"] = nearest_idx.astype(np.int32)

    out["seed"] = int(seed)
    out["lifing_mode"] = lifing_mode
    out["triangles"] = mesh.triangles.astype(np.int32)
    out["contour_points_mm"] = contour.points.astype(np.float64)
    out["contour_zone_id"] = contour_zone_id.astype(np.int32)
    out["contour_subzone_id"] = contour.subzone_ids.astype(np.int32)
    out["contour_region_id"] = contour_region_id.astype(np.int32)
    out["contour_arc_length_mm"] = contour.arc_length_mm.astype(np.float64)
    out["zone_names"] = np.array(contour.zone_names, dtype="S32")
    out["subzone_names"] = np.array(contour.subzone_names, dtype="S32")
    for key, value in contour.landmarks_mm.items():
        if key in {"lower_transition_start", "lower_transition_end", "upper_transition_start", "upper_transition_end", "r_inner", "r_outer", "r_flange_outer"}:
            continue
        out.setdefault("feature_landmarks_mm", {})[key] = np.asarray(value, dtype=np.float64)
    return out
