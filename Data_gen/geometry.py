"""Disc meridional geometry for the required 5-zone family.

Outer-contour structure (v2):
  The disc contour is a closed polygon in [x, r] (axial × radial) coordinates.
  Going clockwise from the bore inner face:
    1. inner_cap     – bore inner face at r = r0
    2. front_face    – bore/lower_transition/web/upper_transition/rim front face
    3. outer_cap     – outer rim face, now composed of named segments:
         front_flange_face  : vertical at x=-t_rim/2, from r5 to r5+h_fl
         front_flange_top   : horizontal at r=r5+h_fl with top-corner fillet
         front_shoulder     : cosine-blend descent from r5+h_fl to r5
         rim_main           : flat cap at r=r5
         rear_shoulder      : cosine-blend ascent from r5 to r5+h_rl
         rear_flange_top    : horizontal at r=r5+h_rl with top-corner fillet
         rear_flange_face   : vertical at x=+t_rim/2, from r5+h_rl to r5
    4. rear_face     – rim/upper_transition/web/lower_transition/bore rear face

  Subzone labels (SUBZONE_NAME_TO_ID) are assigned per point during construction
  and stored in ContourData.subzone_ids alongside the existing zone_ids.
  zone_ids (0-4) are unchanged; S-N curve selection is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import (
    REGION_NAME_TO_ID,
    SUBZONE_NAME_TO_ID,
    THICKNESS_ORDERING_GAP_MM,
    ZONE_NAME_TO_ID,
    ZONE_TO_REGION,
    ZONE_TO_SUBZONE,
    radial_stations_from_params,
)


@dataclass
class ContourData:
    points: np.ndarray
    zone_ids: np.ndarray
    region_ids: np.ndarray
    subzone_ids: np.ndarray       # new: fine-grained subzone labels
    arc_length_mm: np.ndarray
    zone_names: List[str]
    subzone_names: List[str]      # new: ordered subzone name list
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
    ratio = fillet_radius / max(abs(delta_t), 1e-6)
    power = np.clip(2.2 - 0.5 * ratio, 1.6, 2.4)
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




def sanitize_flange_parameters(fp: Dict[str, float], t_rim: float) -> Dict[str, float]:
    """Clip stepped-rim parameter values to meshable, non-overlapping limits."""
    out = {k: max(float(v), 1e-3) for k, v in fp.items()}

    min_land = 0.8
    min_core = 2.0
    max_total = 0.88 * float(t_rim)

    out["front_flange_radial_height"] = min(out["front_flange_radial_height"], 0.22 * t_rim)
    out["rear_flange_radial_height"] = min(out["rear_flange_radial_height"], 0.22 * t_rim)

    out["front_fillet_radius"] = min(
        out["front_fillet_radius"],
        0.45 * out["front_flange_radial_height"],
        0.45 * out["front_flange_axial_length"],
    )
    out["rear_fillet_radius"] = min(
        out["rear_fillet_radius"],
        0.45 * out["rear_flange_radial_height"],
        0.45 * out["rear_flange_axial_length"],
    )

    out["rim_to_flange_fillet_radius_front"] = min(
        out["rim_to_flange_fillet_radius_front"],
        0.45 * out["front_flange_radial_height"],
        0.45 * out["front_shoulder_offset"],
    )
    out["rim_to_flange_fillet_radius_rear"] = min(
        out["rim_to_flange_fillet_radius_rear"],
        0.45 * out["rear_flange_radial_height"],
        0.45 * out["rear_shoulder_offset"],
    )

    out["front_flange_axial_length"] = max(
        out["front_flange_axial_length"], out["front_fillet_radius"] + min_land
    )
    out["rear_flange_axial_length"] = max(
        out["rear_flange_axial_length"], out["rear_fillet_radius"] + min_land
    )
    out["front_shoulder_offset"] = max(
        out["front_shoulder_offset"], out["rim_to_flange_fillet_radius_front"] + 0.6
    )
    out["rear_shoulder_offset"] = max(
        out["rear_shoulder_offset"], out["rim_to_flange_fillet_radius_rear"] + 0.6
    )

    total_ax = (
        out["front_flange_axial_length"] + out["front_shoulder_offset"]
        + out["rear_flange_axial_length"] + out["rear_shoulder_offset"]
    )
    if total_ax > max_total:
        scale = max_total / total_ax
        for key in (
            "front_flange_axial_length", "front_shoulder_offset",
            "rear_flange_axial_length", "rear_shoulder_offset",
        ):
            out[key] *= scale

    def _enforce_side(prefix: str, shoulder_key: str, land_key: str, corner_key: str, height_key: str) -> None:
        out[land_key] = max(out[land_key], out[corner_key] + min_land)
        out[shoulder_key] = max(out[shoulder_key], out[prefix] + 0.6)
        out[prefix] = min(out[prefix], 0.45 * out[shoulder_key], 0.45 * out[land_key])
        out[corner_key] = min(out[corner_key], 0.45 * out[land_key], 0.45 * out[height_key])

    _enforce_side("rim_to_flange_fillet_radius_front", "front_shoulder_offset", "front_flange_axial_length", "front_fillet_radius", "front_flange_radial_height")
    _enforce_side("rim_to_flange_fillet_radius_rear", "rear_shoulder_offset", "rear_flange_axial_length", "rear_fillet_radius", "rear_flange_radial_height")

    total_ax = (
        out["front_flange_axial_length"] + out["front_shoulder_offset"]
        + out["rear_flange_axial_length"] + out["rear_shoulder_offset"]
    )
    if total_ax > max_total:
        scale = max_total / total_ax
        for key in (
            "front_flange_axial_length", "front_shoulder_offset",
            "rear_flange_axial_length", "rear_shoulder_offset",
        ):
            out[key] *= scale
        out["front_flange_axial_length"] = max(out["front_flange_axial_length"], out["front_fillet_radius"] + min_land)
        out["rear_flange_axial_length"] = max(out["rear_flange_axial_length"], out["rear_fillet_radius"] + min_land)

    front_total = out["front_flange_axial_length"] + out["front_shoulder_offset"]
    rear_total = out["rear_flange_axial_length"] + out["rear_shoulder_offset"]
    core_land = float(t_rim) - front_total - rear_total
    if core_land < min_core:
        deficit = min_core - core_land
        reducible_front = max(out["front_flange_axial_length"] - (out["front_fillet_radius"] + min_land), 0.0)
        reducible_rear = max(out["rear_flange_axial_length"] - (out["rear_fillet_radius"] + min_land), 0.0)
        reducible_sh_f = max(out["front_shoulder_offset"] - (out["rim_to_flange_fillet_radius_front"] + 0.6), 0.0)
        reducible_sh_r = max(out["rear_shoulder_offset"] - (out["rim_to_flange_fillet_radius_rear"] + 0.6), 0.0)
        reducible_total = reducible_front + reducible_rear + reducible_sh_f + reducible_sh_r
        if reducible_total > 1e-9:
            for key, reducible in (
                ("front_flange_axial_length", reducible_front),
                ("rear_flange_axial_length", reducible_rear),
                ("front_shoulder_offset", reducible_sh_f),
                ("rear_shoulder_offset", reducible_sh_r),
            ):
                out[key] -= deficit * (reducible / reducible_total)

    return out


def _arc_points(center_x: float, center_r: float, radius: float, angle_start_deg: float, angle_end_deg: float, n: int) -> np.ndarray:
    angles = np.linspace(np.deg2rad(angle_start_deg), np.deg2rad(angle_end_deg), n, endpoint=False)
    return np.column_stack([center_x + radius * np.cos(angles), center_r + radius * np.sin(angles)])


def _line_points(x0: float, r0: float, x1: float, r1: float, n: int) -> np.ndarray:
    x = np.linspace(x0, x1, n, endpoint=False)
    r = np.linspace(r0, r1, n, endpoint=False)
    return np.column_stack([x, r])


def _build_outer_cap_with_flanges(
    t_rim: float,
    r5: float,
    fp: Dict[str, float],
    n_per_seg: int = 15,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build the stepped rim outer boundary with one stepped feature per side."""
    x_front = -0.5 * t_rim
    x_rear = 0.5 * t_rim

    h_f = fp["front_flange_radial_height"]
    h_r = fp["rear_flange_radial_height"]
    land_f = fp["front_flange_axial_length"]
    land_r = fp["rear_flange_axial_length"]
    sh_f = fp["front_shoulder_offset"]
    sh_r = fp["rear_shoulder_offset"]
    rf_f = fp["front_fillet_radius"]
    rf_r = fp["rear_fillet_radius"]
    root_f = fp["rim_to_flange_fillet_radius_front"]
    root_r = fp["rim_to_flange_fillet_radius_rear"]

    x_step_front_end = x_front + land_f
    x_root_front_end = x_step_front_end + sh_f
    x_step_rear_start = x_rear - land_r
    x_root_rear_start = x_step_rear_start - sh_r
    x_core_start = x_root_front_end
    x_core_end = x_root_rear_start

    if x_core_end <= x_core_start:
        raise ValueError("Invalid stepped-rim geometry: front and rear features overlap")

    sz = SUBZONE_NAME_TO_ID
    segs: List[Tuple[np.ndarray, int]] = []

    segs.append((_line_points(x_front, r5, x_front, r5 + h_f - rf_f, max(5, n_per_seg // 2)), sz["front_step"]))
    segs.append((_arc_points(x_front + rf_f, r5 + h_f - rf_f, rf_f, 180.0, 90.0, max(8, n_per_seg // 2)), sz["front_step"]))
    segs.append((_line_points(x_front + rf_f, r5 + h_f, x_step_front_end, r5 + h_f, max(10, n_per_seg)), sz["front_step"]))
    segs.append((_arc_points(x_step_front_end, r5 + h_f - root_f, root_f, 90.0, 0.0, max(8, n_per_seg // 2)), sz["front_shoulder"]))
    segs.append((_line_points(x_step_front_end + root_f, r5 + h_f - root_f, x_root_front_end, r5, max(12, n_per_seg)), sz["front_shoulder"]))

    segs.append((_line_points(x_core_start, r5, x_core_end, r5, max(18, 2 * n_per_seg)), sz["rim_main"]))

    segs.append((_line_points(x_root_rear_start, r5, x_step_rear_start - root_r, r5 + h_r - root_r, max(12, n_per_seg)), sz["rear_shoulder"]))
    segs.append((_arc_points(x_step_rear_start, r5 + h_r - root_r, root_r, 180.0, 90.0, max(8, n_per_seg // 2)), sz["rear_shoulder"]))
    segs.append((_line_points(x_step_rear_start, r5 + h_r, x_rear - rf_r, r5 + h_r, max(10, n_per_seg)), sz["rear_step"]))
    segs.append((_arc_points(x_rear - rf_r, r5 + h_r - rf_r, rf_r, 90.0, 0.0, max(8, n_per_seg // 2)), sz["rear_step"]))
    segs.append((_line_points(x_rear, r5 + h_r - rf_r, x_rear, r5, max(5, n_per_seg // 2)), sz["rear_step"]))

    points = np.vstack([s[0] for s in segs]).astype(np.float64)
    subzone = np.concatenate([np.full(s[0].shape[0], s[1], dtype=np.int32) for s in segs])
    feature_meta = {
        "front_root": np.array([x_step_front_end + 0.6 * root_f, r5 + h_f - 0.6 * root_f], dtype=np.float64),
        "front_outer_corner": np.array([x_front + 0.6 * rf_f, r5 + h_f - 0.6 * rf_f], dtype=np.float64),
        "rear_root": np.array([x_step_rear_start - 0.6 * root_r, r5 + h_r - 0.6 * root_r], dtype=np.float64),
        "rear_outer_corner": np.array([x_rear - 0.6 * rf_r, r5 + h_r - 0.6 * rf_r], dtype=np.float64),
        "rim_core_reference": np.array([0.5 * (x_core_start + x_core_end), r5], dtype=np.float64),
        "front_land_end": np.array([x_step_front_end, r5 + h_f], dtype=np.float64),
        "rear_land_start": np.array([x_step_rear_start, r5 + h_r], dtype=np.float64),
    }
    return points, subzone, feature_meta


def sanitize_geometry_parameters(params: Dict[str, float]) -> Dict[str, float]:
    """Clip geometry values to physically constructible limits."""
    out = {k: float(v) for k, v in params.items()}

    for key in [
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
    ]:
        out[key] = max(out[key], 1e-3)

    # Mandatory section-thickness ordering for every generated sample.
    # The benchmark semantics require bore > rim > web, not only in nominal.
    out["rim_thickness"] = max(out["rim_thickness"], out["web_thickness"] + THICKNESS_ORDERING_GAP_MM)
    out["bore_thickness"] = max(out["bore_thickness"], out["rim_thickness"] + THICKNESS_ORDERING_GAP_MM)
    # Keep this ordering block after all thickness edits so bore > rim > web is preserved.

    lower_dt = abs(out["bore_thickness"] - out["web_thickness"])
    upper_dt = abs(out["rim_thickness"] - out["web_thickness"])

    lower_limit = 0.5 * min(out["lower_transition_height"], max(lower_dt, 1e-6))
    upper_limit = 0.5 * min(out["upper_transition_height"], max(upper_dt, 1e-6))

    out["lower_fillet_radius"] = min(out["lower_fillet_radius"], lower_limit)
    out["upper_fillet_radius"] = min(out["upper_fillet_radius"], upper_limit)
    return out


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

    if not (params["bore_thickness"] > params["rim_thickness"] > params["web_thickness"]):
        raise ValueError("Invalid geometry: thickness ordering must satisfy bore_thickness > rim_thickness > web_thickness")


def _subzone_by_zone(zone_ids: np.ndarray) -> np.ndarray:
    """Map zone_ids to subzone_ids for points whose subzone equals their zone mapping."""
    subzone = np.empty_like(zone_ids)
    for zname, zid in ZONE_NAME_TO_ID.items():
        szname = ZONE_TO_SUBZONE.get(zname, "rim_main")
        subzone[zone_ids == zid] = SUBZONE_NAME_TO_ID[szname]
    return subzone.astype(np.int32)


def build_disc_contour(
    params: Dict[str, float],
    points_per_side: int = 220,
    flange_params: Dict[str, float] | None = None,
) -> ContourData:
    """Build bore/lower-transition/web/upper-transition/rim contour with optional flanges.

    Parameters
    ----------
    params : dict
        Core disc geometry parameters (PUBLIC_GEOMETRY_PARAMETERS keys).
    points_per_side : int
        Number of points for the front and rear faces.
    flange_params : dict or None
        Sanitised flange geometry parameters (FLANGE_GEOMETRY_PARAMETERS keys).
        If *None* or all-zero, the classic flat outer cap is used (no flanges).
        Pass the result of ``sanitize_flange_parameters`` to ensure validity.
    """
    validate_geometry_parameters(params)

    radial_breaks = radial_stations_from_params(params)
    r0, r1, r2, r3, r4, r5 = [float(v) for v in radial_breaks]

    front_r = np.linspace(r0, r5, points_per_side, endpoint=False)
    rear_r = np.linspace(r5, r0, points_per_side, endpoint=False)

    front_t = _thickness_profile(front_r, params, radial_breaks)
    rear_t = _thickness_profile(rear_r, params, radial_breaks)

    front_x = -0.5 * front_t
    rear_x = +0.5 * rear_t

    front_points = np.column_stack([front_x, front_r])
    rear_points  = np.column_stack([rear_x,  rear_r])

    front_zone = _zone_by_radius(front_r, radial_breaks)
    rear_zone  = _zone_by_radius(rear_r,  radial_breaks)

    # ------------------------------------------------------------------
    # Outer cap (with or without flanges)
    # ------------------------------------------------------------------
    t_rim = float(params["rim_thickness"])

    if flange_params is not None:
        outer_cap_pts, outer_cap_subzone, rim_feature_points = _build_outer_cap_with_flanges(
            t_rim=t_rim, r5=r5, fp=flange_params
        )
    else:
        # Legacy flat outer cap (no flanges)
        outer_cap_pts = np.column_stack([
            np.linspace(-0.5 * t_rim, +0.5 * t_rim, 20, endpoint=False),
            np.full(20, r5, dtype=np.float64),
        ])
        outer_cap_subzone = np.full(outer_cap_pts.shape[0], SUBZONE_NAME_TO_ID["rim_main"], dtype=np.int32)
        rim_feature_points = {}

    inner_cap = np.column_stack([
        np.linspace(+0.5 * params["bore_thickness"], -0.5 * params["bore_thickness"], 20, endpoint=False),
        np.full(20, r0, dtype=np.float64),
    ])

    contour_points = np.vstack([front_points, outer_cap_pts, rear_points, inner_cap])

    zone_ids = np.concatenate([
        front_zone,
        np.full(outer_cap_pts.shape[0], ZONE_NAME_TO_ID["rim"], dtype=np.int32),
        rear_zone,
        np.full(inner_cap.shape[0], ZONE_NAME_TO_ID["bore"], dtype=np.int32),
    ])
    region_ids  = _region_from_zone(zone_ids)

    # Subzone IDs: use outer_cap_subzone for outer cap, zone→subzone mapping elsewhere.
    front_subzone = _subzone_by_zone(front_zone)
    rear_subzone  = _subzone_by_zone(rear_zone)
    inner_cap_subzone = np.full(inner_cap.shape[0], SUBZONE_NAME_TO_ID["bore"], dtype=np.int32)
    subzone_ids = np.concatenate([
        front_subzone, outer_cap_subzone, rear_subzone, inner_cap_subzone,
    ]).astype(np.int32)

    _validate_simple_closed_contour(contour_points)
    arc_length_mm = _polyline_arc_length(contour_points)

    r_flange_outer = r5
    if flange_params is not None:
        r_flange_outer = r5 + max(
            float(flange_params.get("front_flange_radial_height", 0.0)),
            float(flange_params.get("rear_flange_radial_height", 0.0)),
        )

    landmarks_mm = {
        "lower_transition_start": np.array([0.0, r1], dtype=np.float64),
        "lower_transition_end":   np.array([0.0, r2], dtype=np.float64),
        "upper_transition_start": np.array([0.0, r3], dtype=np.float64),
        "upper_transition_end":   np.array([0.0, r4], dtype=np.float64),
        "r_inner":                np.array([r0], dtype=np.float64),
        "r_outer":                np.array([r5], dtype=np.float64),
        "r_flange_outer":         np.array([r_flange_outer], dtype=np.float64),
        "r_step_outer":           np.array([r_flange_outer], dtype=np.float64),
    }
    landmarks_mm.update(rim_feature_points)

    metadata = {
        "radial_breaks_mm": radial_breaks,
        "zone_ids_by_break": np.array([
            ZONE_NAME_TO_ID["bore"],
            ZONE_NAME_TO_ID["lower_transition"],
            ZONE_NAME_TO_ID["web"],
            ZONE_NAME_TO_ID["upper_transition"],
            ZONE_NAME_TO_ID["rim"],
        ], dtype=np.int32),
        "has_flanges": np.array([flange_params is not None], dtype=bool),
    }

    subzone_name_list = list({
        "bore", "lower_transition", "web", "upper_transition",
        "rim_main", "front_step", "rear_step", "front_shoulder", "rear_shoulder",
    })

    return ContourData(
        points=contour_points.astype(np.float64),
        zone_ids=zone_ids,
        region_ids=region_ids,
        subzone_ids=subzone_ids,
        arc_length_mm=arc_length_mm,
        zone_names=["bore", "lower_transition", "web", "upper_transition", "rim"],
        subzone_names=sorted(subzone_name_list, key=lambda n: SUBZONE_NAME_TO_ID[n]),
        landmarks_mm=landmarks_mm,
        metadata=metadata,
    )
