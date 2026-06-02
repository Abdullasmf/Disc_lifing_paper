"""Node-configuration extraction and contour derivative features."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _circular_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.concatenate([values[-(window // 2) :], values, values[: window // 2]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.shape[0]]


def contour_derivative_features(contour_points: np.ndarray, arc_length_mm: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute tangent, curvature, and curvature-gradient on ordered cyclic contour."""
    p_prev = np.roll(contour_points, 1, axis=0)
    p_next = np.roll(contour_points, -1, axis=0)

    ds_prev = np.maximum(arc_length_mm - np.roll(arc_length_mm, 1), 1e-9)
    ds_next = np.maximum(np.roll(arc_length_mm, -1) - arc_length_mm, 1e-9)

    d1 = (p_next - p_prev) / (ds_prev[:, None] + ds_next[:, None])
    speed = np.linalg.norm(d1, axis=1) + 1e-12
    tangent = d1 / speed[:, None]

    d2 = 2.0 * ((p_next - contour_points) / ds_next[:, None] - (contour_points - p_prev) / ds_prev[:, None])
    curvature = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed**3, 1e-12)

    ds_total = 0.5 * (ds_prev + ds_next)
    curvature_grad = (np.roll(curvature, -1) - np.roll(curvature, 1)) / np.maximum(2.0 * ds_total, 1e-9)

    curvature = _circular_smooth(curvature, window=5)
    curvature_grad = _circular_smooth(curvature_grad, window=5)

    return {
        "tangent": tangent,
        "curvature": curvature,
        "curvature_gradient": curvature_grad,
    }


def _payload(
    node_coords_mm: np.ndarray,
    region_id: np.ndarray,
    segment_id: np.ndarray,
    stress_max_vm: np.ndarray,
    life_raw: np.ndarray,
    phase_stress: np.ndarray,
    arc_length_mm: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "node_coords_mm": node_coords_mm.astype(np.float64),
        "region_id": region_id.astype(np.int32),
        "segment_id": segment_id.astype(np.int32),
        "stress_max_vm": stress_max_vm.astype(np.float64),
        "life_raw": life_raw.astype(np.float64),
        "phase_stress_eq": phase_stress.astype(np.float64),
        "arc_length_mm": arc_length_mm.astype(np.float64),
    }


def extract_config_nodes(
    config_name: str,
    mesh_nodes: np.ndarray,
    mesh_region_ids: np.ndarray,
    mesh_segment_ids: np.ndarray,
    mesh_distance_to_contour: np.ndarray,
    mesh_stress_max_vm: np.ndarray,
    mesh_life_raw: np.ndarray,
    mesh_phase_stress: np.ndarray,
    contour_points: np.ndarray,
    contour_region_ids: np.ndarray,
    contour_segment_ids: np.ndarray,
    contour_arc_length_mm: np.ndarray,
    contour_stress_max_vm: np.ndarray,
    contour_life_raw: np.ndarray,
    contour_phase_stress: np.ndarray,
    edge_proximity_distance_mm: float,
) -> Dict[str, np.ndarray]:
    """Extract one node configuration payload.

    edge / edge_derivatives use ordered contour samples as canonical edge representation.
    """
    if config_name == "edge":
        payload = _payload(
            node_coords_mm=contour_points,
            region_id=contour_region_ids,
            segment_id=contour_segment_ids,
            stress_max_vm=contour_stress_max_vm,
            life_raw=contour_life_raw,
            phase_stress=contour_phase_stress,
            arc_length_mm=contour_arc_length_mm,
        )
        payload["node_features"] = np.empty((contour_points.shape[0], 0), dtype=np.float64)
        payload["node_feature_names"] = np.array([], dtype="S64")
        return payload

    if config_name == "edge_derivatives":
        dfeat = contour_derivative_features(contour_points, contour_arc_length_mm)
        payload = _payload(
            node_coords_mm=contour_points,
            region_id=contour_region_ids,
            segment_id=contour_segment_ids,
            stress_max_vm=contour_stress_max_vm,
            life_raw=contour_life_raw,
            phase_stress=contour_phase_stress,
            arc_length_mm=contour_arc_length_mm,
        )
        payload["node_features"] = np.column_stack(
            [
                dfeat["tangent"][:, 0],
                dfeat["tangent"][:, 1],
                dfeat["curvature"],
                dfeat["curvature_gradient"],
            ]
        ).astype(np.float64)
        payload["node_feature_names"] = np.array(
            ["tangent_x", "tangent_r", "curvature", "curvature_gradient"], dtype="S64"
        )
        return payload

    if config_name == "edge_proximity":
        near_edge = mesh_distance_to_contour <= edge_proximity_distance_mm
        strictly_interior = mesh_distance_to_contour > 1e-8
        keep = near_edge & strictly_interior

        interior_nodes = mesh_nodes[keep]
        interior_region = mesh_region_ids[keep]
        interior_segment = mesh_segment_ids[keep]
        interior_stress_max = mesh_stress_max_vm[keep]
        interior_life = mesh_life_raw[keep]
        interior_phase = mesh_phase_stress[keep]
        interior_arc = np.full(interior_nodes.shape[0], np.nan, dtype=np.float64)

        payload = _payload(
            node_coords_mm=np.vstack([contour_points, interior_nodes]),
            region_id=np.concatenate([contour_region_ids, interior_region]),
            segment_id=np.concatenate([contour_segment_ids, interior_segment]),
            stress_max_vm=np.concatenate([contour_stress_max_vm, interior_stress_max]),
            life_raw=np.concatenate([contour_life_raw, interior_life]),
            phase_stress=np.vstack([contour_phase_stress, interior_phase]),
            arc_length_mm=np.concatenate([contour_arc_length_mm, interior_arc]),
        )
        payload["node_features"] = np.empty((payload["node_coords_mm"].shape[0], 0), dtype=np.float64)
        payload["node_feature_names"] = np.array([], dtype="S64")
        return payload

    if config_name == "full":
        payload = _payload(
            node_coords_mm=mesh_nodes,
            region_id=mesh_region_ids,
            segment_id=mesh_segment_ids,
            stress_max_vm=mesh_stress_max_vm,
            life_raw=mesh_life_raw,
            phase_stress=mesh_phase_stress,
            arc_length_mm=np.full(mesh_nodes.shape[0], np.nan, dtype=np.float64),
        )
        payload["node_features"] = np.empty((mesh_nodes.shape[0], 0), dtype=np.float64)
        payload["node_feature_names"] = np.array([], dtype="S64")
        return payload

    raise ValueError(f"Unknown config: {config_name}")
