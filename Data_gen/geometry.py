"""Parametric meridional disc geometry with explicit segment provenance.

v2 notes:
- Keeps the original practical 3-region disc profile (bore/web/rim).
- Every contour sample carries region_id + segment_id provenance.
- Landmark points used by synthetic stress concentration logic are exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import REGION_NAME_TO_ID


@dataclass
class ContourData:
    points: np.ndarray
    region_ids: np.ndarray
    segment_ids: np.ndarray
    arc_length_mm: np.ndarray
    segment_names: List[str]
    segment_regions: List[str]
    landmarks_mm: Dict[str, np.ndarray]


def _line(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]


def _quad_bezier(p0: np.ndarray, pc: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return (
        ((1.0 - t) ** 2)[:, None] * p0[None, :]
        + (2.0 * (1.0 - t) * t)[:, None] * pc[None, :]
        + (t**2)[:, None] * p1[None, :]
    )


def _arc_length(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d[:-1])])
    return s.astype(np.float64)


def build_disc_contour(params: Dict[str, float], points_per_segment: int = 40) -> ContourData:
    """Build a closed contour and semantic metadata in mm."""
    x0 = 0.0
    x1 = params["bore_thickness"]
    x2 = x1 + params["web_length"]
    x3 = x2 + params["rim_length"]

    r_inner = params["bore_radius"]
    bore_wall = max(3.0, 0.7 * params["web_thickness"])
    r_bore_outer = r_inner + bore_wall

    slope = params["web_slope"]
    dr_web = slope * (x2 - x1)
    web_center_start = r_bore_outer + 0.5 * params["web_thickness"]
    web_center_end = web_center_start + dr_web

    r_web_low_start = max(r_inner + 1.0, web_center_start - 0.5 * params["web_thickness"])
    r_web_high_start = r_web_low_start + params["web_thickness"]
    r_web_low_end = max(r_inner + 1.2, web_center_end - 0.5 * params["web_thickness"])
    r_web_high_end = r_web_low_end + params["web_thickness"]

    r_rim_inner = r_web_low_end
    r_rim_outer = r_web_high_end + params["rim_thickness"]

    p0 = np.array([x0, r_inner])
    p1 = np.array([x1, r_inner])
    p2 = np.array([x1, r_web_low_start])
    p3 = np.array([x2, r_web_low_end])
    p4 = np.array([x3, r_rim_inner])
    p5 = np.array([x3, r_rim_outer])
    p6 = np.array([x2, r_web_high_end])
    p7 = np.array([x1, r_web_high_start])
    p8 = np.array([x0, r_bore_outer])

    rbw = params["bore_web_fillet_radius"]
    rwr = params["web_rim_fillet_radius"]

    c23 = np.array([x1 + 0.6 * rbw, r_web_low_start + 0.3 * dr_web])
    c67 = np.array([x2 - 0.6 * rwr, r_web_high_end - 0.2 * dr_web])
    c78 = np.array([x1 - 0.6 * rbw, r_web_high_start - 0.1 * dr_web])
    c34 = np.array([x2 + 0.6 * rwr, r_web_low_end + 0.1 * dr_web])

    segments: List[Tuple[str, str, np.ndarray]] = [
        ("bore_inner_line", "bore", _line(p0, p1, points_per_segment)),
        ("bore_rear_face", "bore", _line(p1, p2, points_per_segment)),
        ("bore_web_lower_fillet", "web", _quad_bezier(p2, c23, p3, points_per_segment)),
        ("web_rim_lower_fillet", "rim", _quad_bezier(p3, c34, p4, points_per_segment)),
        ("rim_rear_face", "rim", _line(p4, p5, points_per_segment)),
        ("rim_outer_line", "rim", _line(p5, p6, points_per_segment)),
        ("web_rim_upper_fillet", "rim", _quad_bezier(p6, c67, p7, points_per_segment)),
        ("web_bore_upper_fillet", "web", _quad_bezier(p7, c78, p8, points_per_segment)),
        ("bore_front_face", "bore", _line(p8, p0, points_per_segment)),
    ]

    points = []
    region_ids = []
    segment_ids = []
    segment_names: List[str] = []
    segment_regions: List[str] = []

    for seg_id, (seg_name, region_name, seg_points) in enumerate(segments):
        points.append(seg_points)
        region_ids.append(np.full(seg_points.shape[0], REGION_NAME_TO_ID[region_name], dtype=np.int32))
        segment_ids.append(np.full(seg_points.shape[0], seg_id, dtype=np.int32))
        segment_names.append(seg_name)
        segment_regions.append(region_name)

    contour_points = np.vstack(points)
    contour_region_ids = np.concatenate(region_ids)
    contour_segment_ids = np.concatenate(segment_ids)
    arc_length_mm = _arc_length(contour_points)

    landmarks_mm = {
        "bore_web_lower": p2,
        "bore_web_upper": p7,
        "web_rim_lower": p3,
        "web_rim_upper": p6,
        "bore_web_lower_ctrl": c23,
        "web_bore_upper_ctrl": c78,
        "web_rim_lower_ctrl": c34,
        "web_rim_upper_ctrl": c67,
        "r_inner": np.array([r_inner], dtype=float),
        "r_outer": np.array([r_rim_outer], dtype=float),
    }

    return ContourData(
        points=contour_points,
        region_ids=contour_region_ids,
        segment_ids=contour_segment_ids,
        arc_length_mm=arc_length_mm,
        segment_names=segment_names,
        segment_regions=segment_regions,
        landmarks_mm=landmarks_mm,
    )
