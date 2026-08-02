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
    """Clip flange parameter values to physically constructible limits.

    Ensures:
    - All values are positive.
    - Fillet radii do not exceed the flange height / axial length.
    - Combined axial extent of both flanges + shoulders fits within t_rim.
    """
    out = {k: max(float(v), 1e-3) for k, v in fp.items()}

    # Top-corner fillet must not exceed flange radial height.
    out["front_fillet_radius"] = min(out["front_fillet_radius"],
                                     0.45 * out["front_flange_radial_height"])
    out["rear_fillet_radius"]  = min(out["rear_fillet_radius"],
                                     0.45 * out["rear_flange_radial_height"])

    # Shoulder fillet must not exceed half the shoulder offset.
    out["rim_to_flange_fillet_radius_front"] = min(
        out["rim_to_flange_fillet_radius_front"], 0.45 * out["front_shoulder_offset"]
    )
    out["rim_to_flange_fillet_radius_rear"] = min(
        out["rim_to_flange_fillet_radius_rear"], 0.45 * out["rear_shoulder_offset"]
    )

    # Axial length must leave room for a flat top (> fillet + shoulder_fillet width).
    min_ax = out["front_fillet_radius"] + out["rim_to_flange_fillet_radius_front"] + 1e-3
    out["front_flange_axial_length"] = max(out["front_flange_axial_length"], min_ax)
    min_ax_r = out["rear_fillet_radius"] + out["rim_to_flange_fillet_radius_rear"] + 1e-3
    out["rear_flange_axial_length"] = max(out["rear_flange_axial_length"], min_ax_r)

    # Combined axial extent must not exceed rim thickness (leave ≥5% clearance).
    total_ax = (
        out["front_flange_axial_length"] + out["front_shoulder_offset"]
        + out["rear_flange_axial_length"]  + out["rear_shoulder_offset"]
    )
    max_total = 0.90 * float(t_rim)
    if total_ax > max_total:
        scale = max_total / total_ax
        out["front_flange_axial_length"] *= scale
        out["front_shoulder_offset"]     *= scale
        out["rear_flange_axial_length"]  *= scale
        out["rear_shoulder_offset"]      *= scale
        # Re-clip fillets after scaling.
        out["rim_to_flange_fillet_radius_front"] = min(
            out["rim_to_flange_fillet_radius_front"], 0.45 * out["front_shoulder_offset"]
        )
        out["rim_to_flange_fillet_radius_rear"] = min(
            out["rim_to_flange_fillet_radius_rear"], 0.45 * out["rear_shoulder_offset"]
        )

    return out


# ---------------------------------------------------------------------------
# Outer-cap construction with front/rear flanges
# ---------------------------------------------------------------------------

def _quarter_arc_points(
    center_x: float, center_r: float, radius: float,
    angle_start_deg: float, angle_end_deg: float, n: int = 8,
) -> np.ndarray:
    """Return *n* points on a circular arc (exclusive of endpoint)."""
    angles = np.linspace(np.deg2rad(angle_start_deg), np.deg2rad(angle_end_deg), n, endpoint=False)
    x = center_x + radius * np.cos(angles)
    r = center_r + radius * np.sin(angles)
    return np.column_stack([x, r])


def _cosine_descent(
    x_start: float, x_end: float, r_start: float, r_end: float, n: int = 20,
) -> np.ndarray:
    """Smooth cosine blend from (x_start, r_start) to (x_end, r_end) (exclusive of endpoint)."""
    x = np.linspace(x_start, x_end, n, endpoint=False)
    u = (x - x_start) / max(x_end - x_start, 1e-9)
    r = r_start + (r_end - r_start) * 0.5 * (1.0 - np.cos(np.pi * u))
    return np.column_stack([x, r])


def _build_outer_cap_with_flanges(
    t_rim: float,
    r5: float,
    fp: Dict[str, float],
    n_per_seg: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the outer-cap path with front and rear flanges.

    Parameters
    ----------
    t_rim : float
        Rim thickness [mm].  Front face at x = -t_rim/2, rear at x = +t_rim/2.
    r5 : float
        Nominal rim outer radius [mm] (top of the main cap).
    fp : dict
        Sanitised flange parameter dictionary.
    n_per_seg : int
        Approximate number of points per straight or smooth segment.

    Returns
    -------
    points : (N, 2) float64 array of [x, r] outer-cap points.
    subzone_ids : (N,) int32 array of SUBZONE_NAME_TO_ID labels.
    """
    x_front = -0.5 * t_rim    # x-position of front (−) face
    x_rear  = +0.5 * t_rim    # x-position of rear (+) face

    h_fl = fp["front_flange_radial_height"]
    h_rl = fp["rear_flange_radial_height"]
    fl_ax = fp["front_flange_axial_length"]     # front flange axial length (inward from x_front)
    rl_ax = fp["rear_flange_axial_length"]       # rear  flange axial length (inward from x_rear)
    sh_f  = fp["front_shoulder_offset"]          # front shoulder transition width
    sh_r  = fp["rear_shoulder_offset"]           # rear  shoulder transition width
    rf_f  = fp["front_fillet_radius"]            # front flange top-corner fillet
    rf_r  = fp["rear_fillet_radius"]             # rear  flange top-corner fillet
    rsf_f = fp["rim_to_flange_fillet_radius_front"]  # front shoulder fillet
    rsf_r = fp["rim_to_flange_fillet_radius_rear"]   # rear  shoulder fillet

    # Key x-positions
    x_fl_inner = x_front + fl_ax          # inner axial edge of front flange
    x_rl_inner = x_rear  - rl_ax          # inner axial edge of rear flange

    SZ = SUBZONE_NAME_TO_ID

    segs: List[Tuple[np.ndarray, int]] = []   # (points, subzone_id)

    # ------------------------------------------------------------------
    # 1. Front flange face: vertical at x=x_front, from r5 to r5+h_fl
    #    No corner at the base (continuation of front face).
    #    The corner at the top needs a fillet.
    # ------------------------------------------------------------------
    n_ff_face = max(4, int(n_per_seg * h_fl / max(h_fl + fl_ax, 1e-9)))
    r_face_fl = np.linspace(r5, r5 + h_fl - rf_f, n_ff_face, endpoint=False)
    segs.append((np.column_stack([np.full(n_ff_face, x_front), r_face_fl]), SZ["front_flange"]))

    # 2. Front flange top-corner fillet (90°: going-up turns to going-right)
    #    Arc center at (x_front + rf_f, r5+h_fl - rf_f)
    #    Arc from 180° to 90° (sweeping counterclockwise → points go left→top)
    arc_ff = _quarter_arc_points(
        x_front + rf_f, r5 + h_fl - rf_f, rf_f,
        angle_start_deg=180.0, angle_end_deg=90.0, n=max(6, n_per_seg // 3),
    )
    segs.append((arc_ff, SZ["front_flange"]))

    # 3. Front flange top: horizontal at r=r5+h_fl
    x_fl_top_start = x_front + rf_f
    x_fl_top_end   = x_fl_inner - rsf_f
    if x_fl_top_end > x_fl_top_start + 1e-6:
        n_top_f = max(4, n_per_seg)
        x_top_f = np.linspace(x_fl_top_start, x_fl_top_end, n_top_f, endpoint=False)
        segs.append((np.column_stack([x_top_f, np.full(n_top_f, r5 + h_fl)]), SZ["front_flange"]))

    # 4. Front shoulder: smooth cosine descent from r5+h_fl to r5
    x_sh_f_start = x_fl_inner - rsf_f
    x_sh_f_end   = x_fl_inner + sh_f
    seg_sh_f = _cosine_descent(x_sh_f_start, x_sh_f_end, r5 + h_fl, r5, n=max(16, n_per_seg * 2))
    segs.append((seg_sh_f, SZ["front_shoulder"]))

    # 5. Main outer cap: flat at r=r5
    x_cap_start = x_fl_inner + sh_f
    x_cap_end   = x_rl_inner - sh_r
    if x_cap_end > x_cap_start + 1e-6:
        n_cap = max(4, n_per_seg * 2)
        x_cap = np.linspace(x_cap_start, x_cap_end, n_cap, endpoint=False)
        segs.append((np.column_stack([x_cap, np.full(n_cap, r5)]), SZ["rim_main"]))

    # 6. Rear shoulder: smooth cosine ascent from r5 to r5+h_rl
    x_sh_r_start = x_rl_inner - sh_r
    x_sh_r_end   = x_rl_inner + rsf_r
    seg_sh_r = _cosine_descent(x_sh_r_start, x_sh_r_end, r5, r5 + h_rl, n=max(16, n_per_seg * 2))
    segs.append((seg_sh_r, SZ["rear_shoulder"]))

    # 7. Rear flange top: horizontal at r=r5+h_rl
    x_rl_top_start = x_rl_inner + rsf_r
    x_rl_top_end   = x_rear - rf_r
    if x_rl_top_end > x_rl_top_start + 1e-6:
        n_top_r = max(4, n_per_seg)
        x_top_r = np.linspace(x_rl_top_start, x_rl_top_end, n_top_r, endpoint=False)
        segs.append((np.column_stack([x_top_r, np.full(n_top_r, r5 + h_rl)]), SZ["rear_flange"]))

    # 8. Rear flange top-corner fillet (90°: going-right turns to going-down)
    #    Arc center at (x_rear - rf_r, r5+h_rl - rf_r)
    #    Arc from 90° to 0°
    arc_rl = _quarter_arc_points(
        x_rear - rf_r, r5 + h_rl - rf_r, rf_r,
        angle_start_deg=90.0, angle_end_deg=0.0, n=max(6, n_per_seg // 3),
    )
    segs.append((arc_rl, SZ["rear_flange"]))

    # 9. Rear flange face: vertical at x=x_rear, from r5+h_rl to r5
    n_rl_face = max(4, int(n_per_seg * h_rl / max(h_rl + rl_ax, 1e-9)))
    r_face_rl = np.linspace(r5 + h_rl - rf_r, r5, n_rl_face, endpoint=False)
    segs.append((np.column_stack([np.full(n_rl_face, x_rear), r_face_rl]), SZ["rear_flange"]))

    points_list   = [s[0] for s in segs]
    subzone_list  = [np.full(s[0].shape[0], s[1], dtype=np.int32) for s in segs]

    return (
        np.vstack(points_list).astype(np.float64),
        np.concatenate(subzone_list).astype(np.int32),
    )


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
        outer_cap_pts, outer_cap_subzone = _build_outer_cap_with_flanges(
            t_rim=t_rim, r5=r5, fp=flange_params
        )
    else:
        # Legacy flat outer cap (no flanges)
        outer_cap_pts = np.column_stack([
            np.linspace(-0.5 * t_rim, +0.5 * t_rim, 20, endpoint=False),
            np.full(20, r5, dtype=np.float64),
        ])
        outer_cap_subzone = np.full(outer_cap_pts.shape[0], SUBZONE_NAME_TO_ID["rim_main"], dtype=np.int32)

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
        "has_flanges": np.array([flange_params is not None], dtype=bool),
    }

    subzone_name_list = list({
        "bore", "lower_transition", "web", "upper_transition",
        "rim_main", "front_flange", "rear_flange", "front_shoulder", "rear_shoulder",
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
