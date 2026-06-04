"""Disc meridional geometry for the required 5-zone family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import ZONE_NAME_TO_ID, ZONE_TO_REGION, REGION_NAME_TO_ID


@dataclass
class ContourData:
    points: np.ndarray
    zone_ids: np.ndarray
    region_ids: np.ndarray
    arc_length_mm: np.ndarray
    zone_names: List[str]
    landmarks_mm: Dict[str, np.ndarray]
    metadata: Dict[str, np.ndarray]


def _zone_by_radius(r: np.ndarray, rb: np.ndarray) -> np.ndarray:
    zone = np.empty(r.shape[0], dtype=np.int32)
    zone[r <= rb[1]] = ZONE_NAME_TO_ID["bore"]
    zone[(r > rb[1]) & (r <= rb[2])] = ZONE_NAME_TO_ID["lower_transition"]
    zone[(r > rb[2]) & (r <= rb[3])] = ZONE_NAME_TO_ID["web"]
    zone[(r > rb[3]) & (r <= rb[4])] = ZONE_NAME_TO_ID["upper_transition"]
    zone[r > rb[4]] = ZONE_NAME_TO_ID["rim"]
    return zone


def _region_from_zone(zone_ids: np.ndarray) -> np.ndarray:
    regions = np.empty_like(zone_ids)
    for zone_name, zid in ZONE_NAME_TO_ID.items():
        region_name = ZONE_TO_REGION[zone_name]
        regions[zone_ids == zid] = REGION_NAME_TO_ID[region_name]
    return regions.astype(np.int32)


def _fillet_blend(u: np.ndarray, delta_t: float, fillet_radius: float) -> np.ndarray:
    ratio = abs(delta_t) / max(fillet_radius, 1e-6)
    power = np.clip(1.2 + 0.45 * ratio, 1.2, 4.0)
    up = np.power(np.clip(u, 0.0, 1.0), power)
    down = np.power(np.clip(1.0 - u, 0.0, 1.0), power)
    return up / np.maximum(up + down, 1e-12)


def _thickness_profile(r: np.ndarray, params: Dict[str, float], rb: np.ndarray) -> np.ndarray:
    tb = params["bore_thickness"]
    tw = params["web_thickness"]
    tr = params["rim_thickness"]

    t = np.empty_like(r)

    bore_mask = r <= rb[1]
    lower_mask = (r > rb[1]) & (r <= rb[2])
    web_mask = (r > rb[2]) & (r <= rb[3])
    upper_mask = (r > rb[3]) & (r <= rb[4])
    rim_mask = r > rb[4]

    t[bore_mask] = tb
    t[web_mask] = tw
    t[rim_mask] = tr

    if np.any(lower_mask):
        u = (r[lower_mask] - rb[1]) / max(rb[2] - rb[1], 1e-9)
        s = _fillet_blend(u, tw - tb, params["lower_fillet_radius"])
        t[lower_mask] = tb + (tw - tb) * s

    if np.any(upper_mask):
        u = (r[upper_mask] - rb[3]) / max(rb[4] - rb[3], 1e-9)
        s = _fillet_blend(u, tr - tw, params["upper_fillet_radius"])
        t[upper_mask] = tw + (tr - tw) * s

    return t


def _polyline_arc_length(points: np.ndarray) -> np.ndarray:
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return np.concatenate([[0.0], np.cumsum(ds[:-1])]).astype(np.float64)


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = _ccw(a, b, c)
    o2 = _ccw(a, b, d)
    o3 = _ccw(c, d, a)
    o4 = _ccw(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _validate_simple_closed_contour(points: np.ndarray) -> None:
    n = points.shape[0]
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == n - 1:
                continue
            c = points[j]
            d = points[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                raise ValueError("Generated contour is self-intersecting")


def validate_geometry_parameters(params: Dict[str, float]) -> None:
    positive_keys = [
        "bore_radius_inner",
        "bore_height",
        "bore_thickness",
        "lower_transition_height",
        "web_height",
        "web_thickness",
        "upper_transition_height",
        "rim_height",
        "rim_thickness",
        "lower_fillet_radius",
        "upper_fillet_radius",
    ]
    for key in positive_keys:
        if params[key] <= 0.0:
            raise ValueError(f"Invalid geometry: {key} must be positive")

    lower_dt = abs(params["bore_thickness"] - params["web_thickness"])
    upper_dt = abs(params["rim_thickness"] - params["web_thickness"])
    lower_limit = 0.5 * min(params["lower_transition_height"], max(lower_dt, 1e-6))
    upper_limit = 0.5 * min(params["upper_transition_height"], max(upper_dt, 1e-6))

    if params["lower_fillet_radius"] > lower_limit + 1e-9:
        raise ValueError("Invalid geometry: lower_fillet_radius too large for lower transition")
    if params["upper_fillet_radius"] > upper_limit + 1e-9:
        raise ValueError("Invalid geometry: upper_fillet_radius too large for upper transition")

    if params["rim_thickness"] < 0.65 * params["web_thickness"]:
        raise ValueError("Invalid geometry: rim thickness too thin for stable section construction")


def build_disc_contour(params: Dict[str, float], points_per_side: int = 220) -> ContourData:
    """Build required bore/lower-transition/web/upper-transition/rim contour."""
    validate_geometry_parameters(params)

    r0 = params["bore_radius_inner"]
    r1 = r0 + params["bore_height"]
    r2 = r1 + params["lower_transition_height"]
    r3 = r2 + params["web_height"]
    r4 = r3 + params["upper_transition_height"]
    r5 = r4 + params["rim_height"]
    radial_breaks = np.array([r0, r1, r2, r3, r4, r5], dtype=np.float64)

    front_r = np.linspace(r0, r5, points_per_side, endpoint=False)
    rear_r = np.linspace(r5, r0, points_per_side, endpoint=False)

    front_t = _thickness_profile(front_r, params, radial_breaks)
    rear_t = _thickness_profile(rear_r, params, radial_breaks)

    front_x = -0.5 * front_t
    rear_x = +0.5 * rear_t

    inner_cap = np.column_stack([
        np.linspace(-0.5 * params["bore_thickness"], +0.5 * params["bore_thickness"], 20, endpoint=False),
        np.full(20, r0, dtype=np.float64),
    ])
    outer_cap = np.column_stack([
        np.linspace(+0.5 * params["rim_thickness"], -0.5 * params["rim_thickness"], 20, endpoint=False),
        np.full(20, r5, dtype=np.float64),
    ])

    front_points = np.column_stack([front_x, front_r])
    rear_points = np.column_stack([rear_x, rear_r])

    contour_points = np.vstack([inner_cap, rear_points, outer_cap, front_points])
    zone_ids = np.concatenate([
        np.full(inner_cap.shape[0], ZONE_NAME_TO_ID["bore"], dtype=np.int32),
        _zone_by_radius(rear_r, radial_breaks),
        np.full(outer_cap.shape[0], ZONE_NAME_TO_ID["rim"], dtype=np.int32),
        _zone_by_radius(front_r, radial_breaks),
    ])
    region_ids = _region_from_zone(zone_ids)

    _validate_simple_closed_contour(contour_points)
    arc_length_mm = _polyline_arc_length(contour_points)

    landmarks_mm = {
        "lower_transition_start": np.array([0.0, r1], dtype=np.float64),
        "lower_transition_end": np.array([0.0, r2], dtype=np.float64),
        "upper_transition_start": np.array([0.0, r3], dtype=np.float64),
        "upper_transition_end": np.array([0.0, r4], dtype=np.float64),
        "r_inner": np.array([r0], dtype=np.float64),
        "r_outer": np.array([r5], dtype=np.float64),
    }

    metadata = {
        "radial_breaks_mm": radial_breaks,
        "zone_ids_by_break": np.array([
            ZONE_NAME_TO_ID["bore"],
            ZONE_NAME_TO_ID["lower_transition"],
            ZONE_NAME_TO_ID["web"],
            ZONE_NAME_TO_ID["upper_transition"],
            ZONE_NAME_TO_ID["rim"],
        ], dtype=np.int32),
    }

    return ContourData(
        points=contour_points.astype(np.float64),
        zone_ids=zone_ids,
        region_ids=region_ids,
        arc_length_mm=arc_length_mm,
        zone_names=["bore", "lower_transition", "web", "upper_transition", "rim"],
        landmarks_mm=landmarks_mm,
        metadata=metadata,
    )
