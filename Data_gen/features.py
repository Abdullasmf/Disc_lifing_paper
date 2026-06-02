"""Node-configuration extraction and contour derivative features."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _circular_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.concatenate([values[-(window // 2) :], values, values[: window // 2]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.shape[0]]


def resample_contour_uniform_arc_length(
    points: np.ndarray,
    arc_length_mm: np.ndarray,
    region_ids: np.ndarray,
    segment_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a closed contour to uniform arc-length spacing.

    The node count is preserved.  Region and segment labels are assigned from
    the nearest original contour sample measured in 1-D arc-length space (not
    loose spatial nearest-neighbour), which correctly respects segment
    boundaries along the contour.

    Returns
    -------
    new_points : (n, 2) float64
    new_arc_length_mm : (n,) float64  – uniform, starts at 0
    new_region_ids : (n,) int32
    new_segment_ids : (n,) int32
    """
    n = points.shape[0]
    closing_gap = float(np.linalg.norm(points[0] - points[-1]))
    total_arc = float(arc_length_mm[-1]) + closing_gap

    # Uniform arc-length sample positions for n points.
    s_new = np.linspace(0.0, total_arc, n, endpoint=False)

    # Extend with the wrap-around point for linear interpolation.
    s_ext = np.append(arc_length_mm, total_arc)
    pts_ext = np.vstack([points, points[0]])

    x_new = np.interp(s_new, s_ext, pts_ext[:, 0])
    r_new = np.interp(s_new, s_ext, pts_ext[:, 1])
    new_points = np.column_stack([x_new, r_new])

    # Nearest-original-sample assignment in 1-D arc-length space with cyclic wrap.
    diff = np.abs(arc_length_mm[None, :] - s_new[:, None])          # (n, n)
    diff_wrapped = np.minimum(diff, total_arc - diff)
    nearest = np.argmin(diff_wrapped, axis=1)

    new_region_ids = region_ids[nearest].astype(np.int32)
    new_segment_ids = segment_ids[nearest].astype(np.int32)

    return new_points, s_new.astype(np.float64), new_region_ids, new_segment_ids


def contour_derivative_features(contour_points: np.ndarray, arc_length_mm: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute tangent, curvature, and curvature-gradient on ordered cyclic contour."""
    p_prev = np.roll(contour_points, 1, axis=0)
    p_next = np.roll(contour_points, -1, axis=0)

    # Compute cyclic arc-length spacing, correctly handling the wrap-around
    # boundary so that the first and last points are treated symmetrically.
    s = arc_length_mm
    closing_gap = float(np.linalg.norm(contour_points[0] - contour_points[-1]))
    total_arc = float(s[-1]) + closing_gap

    s_prev_vals = np.roll(s, 1)
    s_next_vals = np.roll(s, -1)
    # Boundary corrections: point-0's previous is the last point (at -closing_gap
    # from point-0) and the last point's next is point-0 (at +closing_gap).
    s_prev_vals[0] = s[-1] - total_arc      # equivalent negative offset
    s_next_vals[-1] = total_arc             # wrap-around position of point-0

    ds_prev = np.maximum(s - s_prev_vals, 1e-9)
    ds_next = np.maximum(s_next_vals - s, 1e-9)

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
    mesh_nearest_contour_index: Optional[np.ndarray] = None,
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
        # Reparameterize to uniform arc-length spacing before computing derivatives.
        resampled_pts, resampled_arc, resampled_region_ids, resampled_segment_ids = \
            resample_contour_uniform_arc_length(
                contour_points, contour_arc_length_mm, contour_region_ids, contour_segment_ids
            )

        # Interpolate scalar and phase-wise fields to the resampled arc positions.
        closing_gap = float(np.linalg.norm(contour_points[0] - contour_points[-1]))
        total_arc = float(contour_arc_length_mm[-1]) + closing_gap
        s_ext = np.append(contour_arc_length_mm, total_arc)

        resampled_stress_max_vm = np.interp(
            resampled_arc, s_ext, np.append(contour_stress_max_vm, contour_stress_max_vm[0])
        )
        resampled_life_raw = np.interp(
            resampled_arc, s_ext, np.append(contour_life_raw, contour_life_raw[0])
        )
        n_phases = contour_phase_stress.shape[1]
        resampled_phase_stress = np.column_stack([
            np.interp(
                resampled_arc, s_ext,
                np.append(contour_phase_stress[:, i], contour_phase_stress[0, i])
            )
            for i in range(n_phases)
        ])

        dfeat = contour_derivative_features(resampled_pts, resampled_arc)
        payload = _payload(
            node_coords_mm=resampled_pts,
            region_id=resampled_region_ids,
            segment_id=resampled_segment_ids,
            stress_max_vm=resampled_stress_max_vm,
            life_raw=resampled_life_raw,
            phase_stress=resampled_phase_stress,
            arc_length_mm=resampled_arc,
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
        # Optional debug geometry fields – stored in full only.
        payload["distance_to_contour_mm"] = mesh_distance_to_contour.astype(np.float64)
        if mesh_nearest_contour_index is not None:
            payload["nearest_contour_index"] = mesh_nearest_contour_index.astype(np.int32)
        return payload

    raise ValueError(f"Unknown config: {config_name}")
