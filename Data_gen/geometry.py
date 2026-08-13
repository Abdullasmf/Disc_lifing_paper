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
    RIM_FEATURE_PARAMETERS,
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




def sanitize_rim_feature_parameters(
    fp: Dict[str, float], t_rim: float, bore_thickness: float
) -> Dict[str, float]:
    """Clip rim-feature parameter values to meshable, non-overlapping limits.

    Returns a dict with the same keys as RIM_FEATURE_PARAMETERS, with values
    clipped to physically valid ranges.  Also records whether sanitization was
    applied.
    """
    out = {k: max(float(v), 1e-3) for k, v in fp.items()}

    x_front = -0.5 * t_rim
    x_rear = 0.5 * t_rim

    h_arm = out["rear_arm_radial_height"]
    neck_t = out["rear_arm_neck_thickness"]
    rf_root = out["rear_arm_root_radius"]
    rf_corner = out["rear_arm_outer_corner_radius"]
    arm_proj = out["rear_arm_axial_projection"]

    cg_depth = out["front_cgroove_axial_depth"]
    cg_span = out["front_cgroove_radial_span"]
    cg_pos = out["front_cgroove_radial_pos"]
    rf_entry = out["front_cgroove_entry_radius"]
    rf_floor = out["front_cgroove_floor_radius"]
    rf_exit = out["front_cgroove_exit_radius"]

    # --- Arm radial height: keep modest relative to rim ---
    h_arm = min(h_arm, 0.45 * t_rim)
    h_arm = max(h_arm, 2.0)

    # --- Arm neck must be visibly narrower than arm body ---
    neck_t = min(neck_t, h_arm - 1.0)
    neck_t = max(neck_t, 0.8)

    # --- Root radius: must fit in neck face and body face ---
    rf_root_max = min(0.45 * neck_t, 0.45 * (h_arm - neck_t), 0.45 * rf_root + 0.45)
    rf_root = min(rf_root, rf_root_max)
    rf_root = max(rf_root, 0.2)

    # --- Corner radius: must fit within arm body height above neck ---
    rf_corner = min(rf_corner, 0.45 * (h_arm - neck_t))
    rf_corner = max(rf_corner, 0.2)

    # --- Arm axial projection: must be large enough for all arm features ---
    # Minimum: rf_root (neck-bot) + 0.4 (shelf) + rf_root (body-bot) + rf_root (body-top) + 0.3 (land) + rf_corner
    proj_min = 3.0 * rf_root + 0.4 + 0.3 + rf_corner
    arm_proj = max(arm_proj, proj_min)
    # Maximum: bore axial half-width minus t_rim/2 minus 0.5mm clearance
    proj_max = max(0.5 * bore_thickness - 0.5 * t_rim - 0.5, proj_min + 0.1)
    arm_proj = min(arm_proj, proj_max)

    # --- C-groove position and span ---
    cg_pos = max(cg_pos, 0.3)
    cg_span = max(cg_span, 1.5)

    # Groove must fit within front face (total height h_arm):
    # cg_pos + rf_entry (entry) + cg_span + rf_exit (exit) + rf_root (top corner) + 0.3 <= h_arm
    required_h = cg_pos + rf_entry + cg_span + rf_exit + rf_root + 0.3
    if required_h > h_arm:
        # Scale down cg_span to fit
        available = h_arm - cg_pos - rf_entry - rf_exit - rf_root - 0.3
        cg_span = max(available, 1.0)

    # --- C-groove radial features ---
    rf_floor = min(rf_floor, 0.45 * cg_span / 2.0)
    rf_floor = max(rf_floor, 0.15)

    rf_entry = min(rf_entry, 0.45 * cg_pos)
    rf_entry = max(rf_entry, 0.15)

    rf_exit_max = min(h_arm - cg_pos - cg_span, cg_span) * 0.45
    rf_exit = min(rf_exit, max(rf_exit_max, 0.15))
    rf_exit = max(rf_exit, 0.15)

    # --- C-groove depth: must leave at least 2mm ligament to arm neck ---
    # Groove floor at x_front + cg_depth; arm neck face at x_rear.
    # Ligament axial distance = (x_rear) - (x_front + cg_depth) = t_rim - cg_depth
    cg_depth = min(cg_depth, t_rim - 2.0)
    cg_depth = max(cg_depth, 1.5)

    out["rear_arm_radial_height"] = h_arm
    out["rear_arm_neck_thickness"] = neck_t
    out["rear_arm_root_radius"] = rf_root
    out["rear_arm_outer_corner_radius"] = rf_corner
    out["rear_arm_axial_projection"] = arm_proj
    out["front_cgroove_axial_depth"] = cg_depth
    out["front_cgroove_radial_span"] = cg_span
    out["front_cgroove_radial_pos"] = cg_pos
    out["front_cgroove_entry_radius"] = rf_entry
    out["front_cgroove_floor_radius"] = rf_floor
    out["front_cgroove_exit_radius"] = rf_exit
    return out


def _arc_points(center_x: float, center_r: float, radius: float, angle_start_deg: float, angle_end_deg: float, n: int) -> np.ndarray:
    angles = np.linspace(np.deg2rad(angle_start_deg), np.deg2rad(angle_end_deg), n, endpoint=False)
    return np.column_stack([center_x + radius * np.cos(angles), center_r + radius * np.sin(angles)])


def _line_points(x0: float, r0: float, x1: float, r1: float, n: int) -> np.ndarray:
    x = np.linspace(x0, x1, n, endpoint=False)
    r = np.linspace(r0, r1, n, endpoint=False)
    return np.column_stack([x, r])


def _build_outer_cap_cgroove_arm(
    t_rim: float,
    r5: float,
    fp: Dict[str, float],
    n_per_seg: int = 15,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build the C-groove + annular drive-arm outer boundary.

    Front side: externally open C-groove cut into the front axial face.
    Rear side: annular drive arm with visible neck, shelf, body, land, corner,
               and end face (the blade-equivalent load face).

    Arc convention (in x-r plane):
      _arc_points(cx, cr, rf, a_start, a_end, n) uses np.linspace(a_start, a_end)
      CW traversal (solid to the right): decreasing angles.
      CCW traversal (re-entrant): increasing angles.

    Right-turn (convex external) fillets — CW (decreasing):
      UP→RIGHT:   cx=x+rf, cr=r-rf, 180→90
      RIGHT→DOWN: cx=x-rf, cr=r-rf,  90→0
      LEFT→UP:    cx=x+rf, cr=r+rf, 270→180

    Left-turn (concave re-entrant) fillets — CCW (increasing):
      RIGHT→UP:   cx=x-rf, cr=r+rf, 270→360
      UP→LEFT:    cx=x-rf, cr=r-rf,   0→90
      DOWN→RIGHT: cx=x+rf, cr=r+rf, 180→270  (concave from the arm body's perspective)
    """
    x_front = -0.5 * t_rim
    x_rear = 0.5 * t_rim

    # C-groove parameters
    cg_depth = fp["front_cgroove_axial_depth"]
    cg_span = fp["front_cgroove_radial_span"]
    cg_pos = fp["front_cgroove_radial_pos"]
    rf_entry = fp["front_cgroove_entry_radius"]
    rf_floor = fp["front_cgroove_floor_radius"]
    rf_exit = fp["front_cgroove_exit_radius"]

    # Rear drive-arm parameters
    h_arm = fp["rear_arm_radial_height"]
    neck_t = fp["rear_arm_neck_thickness"]
    rf_root = fp["rear_arm_root_radius"]     # used for root corners, body corners, front top
    rf_corner = fp["rear_arm_outer_corner_radius"]
    arm_proj = fp["rear_arm_axial_projection"]

    # Key x positions
    x_arm_end = x_rear + arm_proj
    # x_body_start: where arm body face begins (leaves a visible shelf at neck level)
    x_body_start = x_rear + rf_root + 0.4 + rf_root  # shelf = rf_root + 0.4 + rf_root
    # Clamp: body_start must be < arm_end - rf_corner - 0.3 (minimum land)
    x_body_start = min(x_body_start, x_arm_end - rf_corner - rf_root - 0.3)
    x_body_start = max(x_body_start, x_rear + 2 * rf_root + 0.2)

    # Key r positions
    r_cg_bot = r5 + cg_pos               # bottom of C-groove opening
    r_cg_top = r5 + cg_pos + cg_span     # top of C-groove opening
    r_arm_top = r5 + h_arm               # arm body top (= ligament level)
    r_neck_top = r5 + neck_t             # arm neck top (shelf level)

    sz = SUBZONE_NAME_TO_ID
    segs: List[Tuple[np.ndarray, int]] = []
    n_arc = max(8, n_per_seg // 2)
    n_line = max(5, n_per_seg // 2)
    n_feat = max(6, n_per_seg // 2)

    # ================================================================
    # FRONT FACE — SECTION A (below C-groove mouth)
    # ================================================================
    if r_cg_bot - rf_entry > r5 + 1e-6:
        segs.append((_line_points(x_front, r5, x_front, r_cg_bot - rf_entry, n_line), sz["front_face"]))

    # ================================================================
    # C-GROOVE ENTRY FILLET  (UP → RIGHT, convex)
    # At corner (x_front, r_cg_bot): UP→RIGHT
    # cx = x_front + rf_entry, cr = r_cg_bot - rf_entry, 180→90
    # ================================================================
    segs.append((_arc_points(x_front + rf_entry, r_cg_bot - rf_entry,
                             rf_entry, 180.0, 90.0, n_feat), sz["front_cgroove"]))

    # ================================================================
    # C-GROOVE LOWER WALL  (going RIGHT at r = r_cg_bot)
    # ================================================================
    x_floor_start = x_front + rf_entry
    x_floor_end = x_front + cg_depth - rf_floor
    if x_floor_end > x_floor_start + 1e-6:
        segs.append((_line_points(x_floor_start, r_cg_bot, x_floor_end, r_cg_bot, n_feat), sz["front_cgroove"]))

    # ================================================================
    # C-GROOVE FLOOR LOWER CORNER  (RIGHT → UP, concave)
    # At corner (x_front+cg_depth, r_cg_bot): RIGHT→UP
    # cx = (x_front+cg_depth) - rf_floor, cr = r_cg_bot + rf_floor, 270→360
    # ================================================================
    segs.append((_arc_points(x_front + cg_depth - rf_floor, r_cg_bot + rf_floor,
                             rf_floor, 270.0, 360.0, n_feat), sz["front_cgroove"]))

    # ================================================================
    # C-GROOVE FLOOR FACE  (going UP at x = x_front + cg_depth)
    # ================================================================
    r_floor_bot = r_cg_bot + rf_floor
    r_floor_top = r_cg_top - rf_floor
    if r_floor_top > r_floor_bot + 1e-6:
        segs.append((_line_points(x_front + cg_depth, r_floor_bot,
                                  x_front + cg_depth, r_floor_top, n_feat), sz["front_cgroove"]))

    # ================================================================
    # C-GROOVE FLOOR UPPER CORNER  (UP → LEFT, concave)
    # At corner (x_front+cg_depth, r_cg_top): UP→LEFT
    # cx = (x_front+cg_depth) - rf_floor, cr = r_cg_top - rf_floor, 0→90
    # ================================================================
    segs.append((_arc_points(x_front + cg_depth - rf_floor, r_cg_top - rf_floor,
                             rf_floor, 0.0, 90.0, n_feat), sz["front_cgroove"]))

    # ================================================================
    # C-GROOVE UPPER WALL  (going LEFT at r = r_cg_top)
    # ================================================================
    x_upper_wall_end = x_front + rf_exit
    x_upper_wall_start = x_front + cg_depth - rf_floor
    if x_upper_wall_start > x_upper_wall_end + 1e-6:
        segs.append((_line_points(x_upper_wall_start, r_cg_top,
                                  x_upper_wall_end, r_cg_top, n_feat), sz["front_cgroove"]))

    # ================================================================
    # C-GROOVE EXIT FILLET  (LEFT → UP, convex)
    # At corner (x_front, r_cg_top): LEFT→UP
    # cx = x_front + rf_exit, cr = r_cg_top + rf_exit, 270→180
    # ================================================================
    segs.append((_arc_points(x_front + rf_exit, r_cg_top + rf_exit,
                             rf_exit, 270.0, 180.0, n_feat), sz["front_face"]))

    # ================================================================
    # FRONT FACE — SECTION B (above C-groove exit)
    # Goes from (x_front, r_cg_top + rf_exit) to (x_front, r_arm_top - rf_root)
    # then the top-corner fillet (UP→RIGHT, convex)
    # ================================================================
    r_front_b_bot = r_cg_top + rf_exit
    r_front_b_top = r_arm_top - rf_root   # before top-corner fillet
    if r_front_b_top > r_front_b_bot + 1e-6:
        segs.append((_line_points(x_front, r_front_b_bot, x_front, r_front_b_top, n_line), sz["front_face"]))

    # Front face top corner (UP → RIGHT, convex)
    # At corner (x_front, r_arm_top): UP→RIGHT
    # cx = x_front + rf_root, cr = r_arm_top - rf_root, 180→90
    segs.append((_arc_points(x_front + rf_root, r_arm_top - rf_root,
                             rf_root, 180.0, 90.0, n_arc), sz["rear_arm_neck"]))

    # ================================================================
    # LIGAMENT  (going RIGHT at r = r_arm_top)
    # From (x_front + rf_root, r_arm_top) to (x_rear - rf_root, r_arm_top)
    # ================================================================
    x_lig_start = x_front + rf_root
    x_lig_end = x_rear - rf_root      # before arm root top corner
    if x_lig_end > x_lig_start + 1e-6:
        segs.append((_line_points(x_lig_start, r_arm_top, x_lig_end, r_arm_top,
                                  max(12, n_per_seg)), sz["rear_arm_neck"]))

    # ================================================================
    # ARM ROOT TOP CORNER  (RIGHT → DOWN, convex)
    # At corner (x_rear, r_arm_top): RIGHT→DOWN
    # cx = x_rear - rf_root, cr = r_arm_top - rf_root, 90→0
    # ================================================================
    segs.append((_arc_points(x_rear - rf_root, r_arm_top - rf_root,
                             rf_root, 90.0, 0.0, n_arc), sz["rear_arm_neck"]))

    # ================================================================
    # ARM NECK FACE  (going DOWN at x = x_rear)
    # From (x_rear, r_arm_top - rf_root) to (x_rear, r_neck_top + rf_root)
    # ================================================================
    r_neck_face_top = r_arm_top - rf_root
    r_neck_face_bot = r_neck_top + rf_root   # before neck-bottom fillet
    if r_neck_face_top > r_neck_face_bot + 1e-6:
        segs.append((_line_points(x_rear, r_neck_face_top, x_rear, r_neck_face_bot,
                                  n_feat), sz["rear_arm_neck"]))

    # ================================================================
    # ARM NECK BOTTOM CORNER  (DOWN → RIGHT)
    # Traverses from going-DOWN at x_rear to going-RIGHT at r_neck_top.
    # Arc formula: cx = x_rear + rf_root, cr = r_neck_top + rf_root, 180→270 (CCW)
    # At 180°: (x_rear, r_neck_top + rf_root) — on neck face going DOWN ✓
    # At 270°: (x_rear + rf_root, r_neck_top) — on shelf going RIGHT ✓
    # ================================================================
    segs.append((_arc_points(x_rear + rf_root, r_neck_top + rf_root,
                             rf_root, 180.0, 270.0, n_feat), sz["rear_arm_neck"]))

    # ================================================================
    # ARM NECK SHELF  (going RIGHT at r = r_neck_top)
    # From (x_rear + rf_root, r_neck_top) to (x_body_start - rf_root, r_neck_top)
    # ================================================================
    x_shelf_start = x_rear + rf_root
    x_shelf_end = x_body_start - rf_root
    if x_shelf_end > x_shelf_start + 1e-6:
        segs.append((_line_points(x_shelf_start, r_neck_top, x_shelf_end, r_neck_top,
                                  n_feat), sz["rear_arm_neck"]))

    # ================================================================
    # ARM BODY LOWER CORNER  (RIGHT → UP, concave re-entrant)
    # At corner (x_body_start, r_neck_top): RIGHT→UP
    # cx = x_body_start - rf_root, cr = r_neck_top + rf_root, 270→360
    # ================================================================
    segs.append((_arc_points(x_body_start - rf_root, r_neck_top + rf_root,
                             rf_root, 270.0, 360.0, n_arc), sz["rear_arm_land"]))

    # ================================================================
    # ARM BODY FACE  (going UP at x = x_body_start)
    # From (x_body_start, r_neck_top + rf_root) to (x_body_start, r_arm_top - rf_root)
    # ================================================================
    r_body_bot = r_neck_top + rf_root
    r_body_top = r_arm_top - rf_root
    if r_body_top > r_body_bot + 1e-6:
        segs.append((_line_points(x_body_start, r_body_bot, x_body_start, r_body_top,
                                  n_feat), sz["rear_arm_land"]))

    # ================================================================
    # ARM BODY TOP CORNER  (UP → RIGHT, convex)
    # At corner (x_body_start, r_arm_top): UP→RIGHT
    # cx = x_body_start + rf_root, cr = r_arm_top - rf_root, 180→90
    # ================================================================
    segs.append((_arc_points(x_body_start + rf_root, r_arm_top - rf_root,
                             rf_root, 180.0, 90.0, n_arc), sz["rear_arm_land"]))

    # ================================================================
    # ARM LAND  (going RIGHT at r = r_arm_top)
    # From (x_body_start + rf_root, r_arm_top) to (x_arm_end - rf_corner, r_arm_top)
    # ================================================================
    x_land_start = x_body_start + rf_root
    x_land_end = x_arm_end - rf_corner
    if x_land_end > x_land_start + 1e-6:
        segs.append((_line_points(x_land_start, r_arm_top, x_land_end, r_arm_top,
                                  max(8, n_per_seg)), sz["rear_arm_land"]))

    # ================================================================
    # ARM OUTER CORNER  (RIGHT → DOWN, convex)
    # At corner (x_arm_end, r_arm_top): RIGHT→DOWN
    # cx = x_arm_end - rf_corner, cr = r_arm_top - rf_corner, 90→0
    # ================================================================
    segs.append((_arc_points(x_arm_end - rf_corner, r_arm_top - rf_corner,
                             rf_corner, 90.0, 0.0, n_arc), sz["rear_arm_corner"]))

    # ================================================================
    # ARM END FACE  (going DOWN at x = x_arm_end) — LOAD FACE
    # From (x_arm_end, r_arm_top - rf_corner) to (x_arm_end, r5)
    # ================================================================
    r_end_top = r_arm_top - rf_corner
    if r_end_top > r5 + 1e-6:
        segs.append((_line_points(x_arm_end, r_end_top, x_arm_end, r5,
                                  n_feat), sz["rear_arm_end_face"]))

    # ================================================================
    # ARM BOTTOM  (going LEFT at r = r5, back to x_rear)
    # ================================================================
    if x_arm_end > x_rear + 1e-6:
        segs.append((_line_points(x_arm_end, r5, x_rear, r5,
                                  max(8, n_per_seg)), sz["rim_main"]))

    points = np.vstack([s[0] for s in segs]).astype(np.float64)
    subzone = np.concatenate([np.full(s[0].shape[0], s[1], dtype=np.int32) for s in segs])

    # --- Feature landmark coordinates ---
    x_groove_floor = x_front + cg_depth
    r_groove_floor_mid = 0.5 * (r_cg_bot + r_cg_top)
    r_load_face_mid = 0.5 * (r5 + r_arm_top - rf_corner)
    feature_meta = {
        "front_cgroove_entry":        np.array([x_front + rf_entry, r_cg_bot], dtype=np.float64),
        "front_cgroove_floor":        np.array([x_groove_floor, r_groove_floor_mid], dtype=np.float64),
        "front_cgroove_exit":         np.array([x_front + rf_exit, r_cg_top], dtype=np.float64),
        "ligament_reference":         np.array([0.5*(x_front + rf_root + x_rear - rf_root), r_arm_top], dtype=np.float64),
        "rear_arm_root":              np.array([x_rear, r_arm_top - 0.5*rf_root], dtype=np.float64),
        "rear_arm_neck":              np.array([x_rear, 0.5*(r5 + r_neck_top)], dtype=np.float64),
        "rear_arm_outer_corner":      np.array([x_arm_end - 0.5*rf_corner, r_arm_top - 0.5*rf_corner], dtype=np.float64),
        "rear_arm_load_face_centroid":np.array([x_arm_end, r_load_face_mid], dtype=np.float64),
        "rim_core_reference":         np.array([x_rear, r5], dtype=np.float64),
        # Blade traction geometry: arm end face extent
        "blade_arm_face_x_end_mm":    np.array([x_arm_end], dtype=np.float64),
        "blade_arm_face_r_min_mm":    np.array([r5], dtype=np.float64),
        "blade_arm_face_r_max_mm":    np.array([r_arm_top], dtype=np.float64),
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
    rim_feature_params: Dict[str, float] | None = None,
) -> ContourData:
    """Build bore/lower-transition/web/upper-transition/rim contour with C-groove and drive arm.

    Parameters
    ----------
    params : dict
        Core disc geometry parameters (PUBLIC_GEOMETRY_PARAMETERS keys).
    points_per_side : int
        Number of points for the front and rear faces.
    rim_feature_params : dict or None
        Sanitised rim-feature parameters (RIM_FEATURE_PARAMETERS keys).
        If *None*, use a flat outer cap (legacy / no features).
        Pass the result of ``sanitize_rim_feature_parameters`` to ensure validity.
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

    t_rim = float(params["rim_thickness"])
    bore_t = float(params["bore_thickness"])

    if rim_feature_params is not None:
        outer_cap_pts, outer_cap_subzone, rim_feature_points = _build_outer_cap_cgroove_arm(
            t_rim=t_rim, r5=r5, fp=rim_feature_params
        )
    else:
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

    front_subzone = _subzone_by_zone(front_zone)
    rear_subzone  = _subzone_by_zone(rear_zone)
    inner_cap_subzone = np.full(inner_cap.shape[0], SUBZONE_NAME_TO_ID["bore"], dtype=np.int32)
    subzone_ids = np.concatenate([
        front_subzone, outer_cap_subzone, rear_subzone, inner_cap_subzone,
    ]).astype(np.int32)

    _validate_simple_closed_contour(contour_points)
    arc_length_mm = _polyline_arc_length(contour_points)

    r_arm_outer = r5
    if rim_feature_params is not None:
        r_arm_outer = r5 + float(rim_feature_params.get("rear_arm_radial_height", 0.0))

    landmarks_mm = {
        "lower_transition_start": np.array([0.0, r1], dtype=np.float64),
        "lower_transition_end":   np.array([0.0, r2], dtype=np.float64),
        "upper_transition_start": np.array([0.0, r3], dtype=np.float64),
        "upper_transition_end":   np.array([0.0, r4], dtype=np.float64),
        "r_inner":                np.array([r0], dtype=np.float64),
        "r_outer":                np.array([r5], dtype=np.float64),
        "r_arm_outer":            np.array([r_arm_outer], dtype=np.float64),
        "r_step_outer":           np.array([r_arm_outer], dtype=np.float64),
    }
    landmarks_mm.update(rim_feature_points)

    # Override rim_core_reference to be at the interior of the rim section
    # (x=0, r=r4+40% of rim_height), well away from the arm root and C-groove
    # stress concentrations, where mesh-independent bulk stress lives.
    r_rim_core = r4 + 0.40 * (r5 - r4)
    landmarks_mm["rim_core_reference"] = np.array([0.0, r_rim_core], dtype=np.float64)

    metadata = {
        "radial_breaks_mm": radial_breaks,
        "zone_ids_by_break": np.array([
            ZONE_NAME_TO_ID["bore"],
            ZONE_NAME_TO_ID["lower_transition"],
            ZONE_NAME_TO_ID["web"],
            ZONE_NAME_TO_ID["upper_transition"],
            ZONE_NAME_TO_ID["rim"],
        ], dtype=np.int32),
        "has_rim_features": np.array([rim_feature_params is not None], dtype=bool),
    }
    if rim_feature_params is not None:
        x_rear = 0.5 * t_rim
        arm_proj = float(rim_feature_params["rear_arm_axial_projection"])
        h_arm = float(rim_feature_params["rear_arm_radial_height"])
        metadata["blade_arm_face_x_end_mm"] = np.array([x_rear + arm_proj], dtype=np.float64)
        metadata["blade_arm_face_r_min_mm"] = np.array([r5], dtype=np.float64)
        metadata["blade_arm_face_r_max_mm"] = np.array([r5 + h_arm], dtype=np.float64)

    subzone_name_list = list(SUBZONE_NAME_TO_ID.keys())

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
