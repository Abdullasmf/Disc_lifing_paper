"""Single-sample deterministic generator layer."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import REPRESENTATIONS, SampleGenerationConfig, clip_offsets_to_bounds, resolve_geometry_parameters
from .features import contour_derivative_features, empty_features, resample_contour_uniform_arc_length
from .geometry import build_disc_contour
from .mesh_ops import assign_zone_and_region_from_contour, generate_mesh
from .physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max


def _compute_targets(
    nodes: np.ndarray,
    zone_ids: np.ndarray,
    region_ids: np.ndarray,
    geometry_params: Dict[str, float],
    landmarks_mm: Dict[str, np.ndarray],
    contour_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_stress = compute_phase_equivalent_stresses(
        nodes=nodes,
        zone_ids=zone_ids,
        region_ids=region_ids,
        geometry_params=geometry_params,
        landmarks_mm=landmarks_mm,
        contour_points=contour_points,
    )
    stress_max_vm = compute_stress_max(phase_stress)
    life_raw = compute_life_raw(phase_stress, region_ids)
    return phase_stress, stress_max_vm, life_raw


def generate_sample(
    param_offsets: dict[str, float],
    representation: str,
    seed: int = 0,
    include_derivatives: bool = True,
    include_debug_fields: bool = False,
) -> dict:
    """Generate one complete deterministic sample from one offset vector."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")

    cfg = SampleGenerationConfig()
    clipped_offsets = clip_offsets_to_bounds(param_offsets)
    actual_params = resolve_geometry_parameters(clipped_offsets)

    contour = build_disc_contour(actual_params, points_per_side=cfg.contour_points_per_side)
    mesh = generate_mesh(
        contour_points=contour.points,
        grid_x=cfg.mesh_grid_points_x,
        grid_r=cfg.mesh_grid_points_r,
        seed=int(seed),
    )

    mesh_zone_id, mesh_region_id, nearest_idx, distance_to_contour = assign_zone_and_region_from_contour(
        nodes=mesh.nodes,
        contour_points=contour.points,
        contour_zone_ids=contour.zone_ids,
    )

    mesh_phase_stress, mesh_stress_max_vm, mesh_life_raw = _compute_targets(
        nodes=mesh.nodes,
        zone_ids=mesh_zone_id,
        region_ids=mesh_region_id,
        geometry_params=actual_params,
        landmarks_mm=contour.landmarks_mm,
        contour_points=contour.points,
    )

    contour_phase_stress, contour_stress_max_vm, contour_life_raw = _compute_targets(
        nodes=contour.points,
        zone_ids=contour.zone_ids,
        region_ids=contour.region_ids,
        geometry_params=actual_params,
        landmarks_mm=contour.landmarks_mm,
        contour_points=contour.points,
    )

    if representation == "edge":
        edge_points, edge_arc, edge_zone, edge_region = resample_contour_uniform_arc_length(
            points=contour.points,
            arc_length_mm=contour.arc_length_mm,
            zone_ids=contour.zone_ids,
            region_ids=contour.region_ids,
            n_samples=contour.points.shape[0],
        )

        edge_phase_stress, edge_stress_max_vm, edge_life_raw = _compute_targets(
            nodes=edge_points,
            zone_ids=edge_zone,
            region_ids=edge_region,
            geometry_params=actual_params,
            landmarks_mm=contour.landmarks_mm,
            contour_points=contour.points,
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
            "representation": representation,
            "node_coords_mm": edge_points,
            "zone_id": edge_zone,
            "region_id": edge_region,
            "stress_max_vm": edge_stress_max_vm,
            "life_raw": edge_life_raw,
            "phase_stress_eq": edge_phase_stress,
            "node_features": node_features,
            "node_feature_names": node_feature_names,
            "arc_length_mm": edge_arc,
        }
    elif representation == "edge_proximity":
        keep = (distance_to_contour <= cfg.edge_proximity_distance_mm) & (distance_to_contour > 1e-8)

        interior_nodes = mesh.nodes[keep]
        interior_zone = mesh_zone_id[keep]
        interior_region = mesh_region_id[keep]
        interior_stress = mesh_stress_max_vm[keep]
        interior_life = mesh_life_raw[keep]
        interior_phase = mesh_phase_stress[keep]

        node_coords = np.vstack([contour.points, interior_nodes])
        zone_id = np.concatenate([contour.zone_ids, interior_zone])
        region_id = np.concatenate([contour.region_ids, interior_region])
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
            "representation": representation,
            "node_coords_mm": node_coords,
            "zone_id": zone_id,
            "region_id": region_id,
            "stress_max_vm": stress,
            "life_raw": life,
            "phase_stress_eq": phase,
            "node_features": node_features,
            "node_feature_names": node_feature_names,
            "arc_length_mm": arc,
        }
    else:
        node_features, node_feature_names = empty_features(mesh.nodes.shape[0])
        out = {
            "param_offsets": {k: float(v) for k, v in clipped_offsets.items()},
            "geometry_parameters_actual": {k: float(v) for k, v in actual_params.items()},
            "representation": representation,
            "node_coords_mm": mesh.nodes,
            "zone_id": mesh_zone_id,
            "region_id": mesh_region_id,
            "stress_max_vm": mesh_stress_max_vm,
            "life_raw": mesh_life_raw,
            "phase_stress_eq": mesh_phase_stress,
            "node_features": node_features,
            "node_feature_names": node_feature_names,
        }
        if include_debug_fields:
            out["distance_to_contour_mm"] = distance_to_contour.astype(np.float64)
            out["nearest_contour_index"] = nearest_idx.astype(np.int32)

    out["seed"] = int(seed)
    out["triangles"] = mesh.triangles.astype(np.int32)
    out["contour_points_mm"] = contour.points.astype(np.float64)
    out["contour_zone_id"] = contour.zone_ids.astype(np.int32)
    out["contour_region_id"] = contour.region_ids.astype(np.int32)
    out["contour_arc_length_mm"] = contour.arc_length_mm.astype(np.float64)
    out["zone_names"] = np.array(contour.zone_names, dtype="S32")
    return out
