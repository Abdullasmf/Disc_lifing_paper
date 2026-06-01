"""Node-configuration extraction and derivative feature computation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def contour_derivative_features(contour_points: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute ordered-contour tangent, curvature and second-derivative-like feature."""
    p_prev = np.roll(contour_points, 1, axis=0)
    p_next = np.roll(contour_points, -1, axis=0)
    d1 = 0.5 * (p_next - p_prev)
    speed = np.linalg.norm(d1, axis=1) + 1e-12
    tangent = d1 / speed[:, None]

    d2 = p_next - 2.0 * contour_points + p_prev
    curvature = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed**3, 1e-12)
    second_like = np.linalg.norm(d2, axis=1)
    return {"tangent": tangent, "curvature": curvature, "second_like": second_like}


def extract_config_nodes(
    config_name: str,
    nodes: np.ndarray,
    boundary_node_ids: np.ndarray,
    distance_to_contour: np.ndarray,
    nearest_contour_index: np.ndarray,
    contour_points: np.ndarray,
    contour_region_ids: np.ndarray,
    stress_max_vm: np.ndarray,
    life_raw: np.ndarray,
    phase_stress: np.ndarray,
    edge_proximity_distance_mm: float,
) -> Dict[str, np.ndarray]:
    """Extract one node configuration payload."""
    n_nodes = nodes.shape[0]
    is_boundary = np.zeros(n_nodes, dtype=bool)
    is_boundary[boundary_node_ids] = True

    if config_name in {"edge", "edge_derivatives"}:
        keep = is_boundary
    elif config_name == "edge_proximity":
        keep = is_boundary | (distance_to_contour <= edge_proximity_distance_mm)
    elif config_name == "full":
        keep = np.ones(n_nodes, dtype=bool)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    idx = np.flatnonzero(keep)
    idx = np.unique(idx)

    payload = {
        "node_coords_mm": nodes[idx].astype(np.float64),
        "region_id": contour_region_ids[nearest_contour_index[idx]].astype(np.int32),
        "stress_max_vm": stress_max_vm[idx].astype(np.float64),
        "life_raw": life_raw[idx].astype(np.float64),
        "phase_stress_eq": phase_stress[idx].astype(np.float64),
    }

    if config_name == "edge_derivatives":
        dfeat = contour_derivative_features(contour_points)
        cidx = nearest_contour_index[idx]
        payload["node_features"] = np.column_stack(
            [
                dfeat["tangent"][cidx, 0],
                dfeat["tangent"][cidx, 1],
                dfeat["curvature"][cidx],
                dfeat["second_like"][cidx],
            ]
        ).astype(np.float64)
        payload["node_feature_names"] = np.array(
            ["tangent_x", "tangent_r", "curvature", "second_derivative_like"], dtype="S64"
        )
    else:
        payload["node_features"] = np.empty((idx.shape[0], 0), dtype=np.float64)
        payload["node_feature_names"] = np.array([], dtype="S64")

    return payload

