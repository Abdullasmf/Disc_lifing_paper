"""Representation extraction and edge derivative features."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _circular_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: values.shape[0]]


def resample_contour_uniform_arc_length(
    points: np.ndarray,
    arc_length_mm: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    closing_gap = float(np.linalg.norm(points[0] - points[-1]))
    total_arc = float(arc_length_mm[-1]) + closing_gap

    s_new = np.linspace(0.0, total_arc, n_samples, endpoint=False)
    s_ext = np.append(arc_length_mm, total_arc)
    pts_ext = np.vstack([points, points[0]])

    x_new = np.interp(s_new, s_ext, pts_ext[:, 0])
    r_new = np.interp(s_new, s_ext, pts_ext[:, 1])
    new_points = np.column_stack([x_new, r_new])

    return (
        new_points.astype(np.float64),
        s_new.astype(np.float64),
    )


def contour_derivative_features(contour_points: np.ndarray, arc_length_mm: np.ndarray) -> Dict[str, np.ndarray]:
    p_prev = np.roll(contour_points, 1, axis=0)
    p_next = np.roll(contour_points, -1, axis=0)

    s = arc_length_mm
    closing_gap = float(np.linalg.norm(contour_points[0] - contour_points[-1]))
    total_arc = float(s[-1]) + closing_gap

    s_prev_vals = np.roll(s, 1)
    s_next_vals = np.roll(s, -1)
    s_prev_vals[0] = -closing_gap
    s_next_vals[-1] = total_arc

    ds_prev = np.maximum(s - s_prev_vals, 1e-9)
    ds_next = np.maximum(s_next_vals - s, 1e-9)

    d1 = (p_next - p_prev) / (ds_prev[:, None] + ds_next[:, None])
    speed = np.linalg.norm(d1, axis=1) + 1e-12
    tangent = d1 / speed[:, None]

    d2 = 2.0 * ((p_next - contour_points) / ds_next[:, None] - (contour_points - p_prev) / ds_prev[:, None])
    curvature = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed**3, 1e-12)

    ds_total = 0.5 * (ds_prev + ds_next)
    curvature_grad = (np.roll(curvature, -1) - np.roll(curvature, 1)) / np.maximum(2.0 * ds_total, 1e-9)

    return {
        "tangent": tangent.astype(np.float64),
        "curvature": _circular_smooth(curvature, window=5).astype(np.float64),
        "curvature_gradient": _circular_smooth(curvature_grad, window=5).astype(np.float64),
    }


def empty_features(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.empty((n_nodes, 0), dtype=np.float64), np.array([], dtype="S64")
