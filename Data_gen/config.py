"""Centralized configuration for the 2-layer synthetic disc dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

ZONE_NAME_TO_ID = {
    "bore": 0,
    "lower_transition": 1,
    "web": 2,
    "upper_transition": 3,
    "rim": 4,
}
ZONE_ID_TO_NAME = {v: k for k, v in ZONE_NAME_TO_ID.items()}

REGION_NAME_TO_ID = {"bore": 0, "web": 1, "rim": 2}
REGION_ID_TO_NAME = {v: k for k, v in REGION_NAME_TO_ID.items()}

ZONE_TO_REGION = {
    "bore": "bore",
    "lower_transition": "web",
    "web": "web",
    "upper_transition": "web",
    "rim": "rim",
}

PUBLIC_GEOMETRY_PARAMETERS = (
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
)

NOMINAL_GEOMETRY_MM: Dict[str, float] = {
    "bore_radius_inner": 24.0,
    "bore_height": 11.0,
    "bore_thickness": 30.0,   # bore > rim > web enforced: 30 > 20 > 10
    "lower_transition_height": 8.0,
    "web_height": 40.0,
    "web_thickness": 10.0,
    "upper_transition_height": 9.0,
    "rim_height": 15.0,
    "rim_thickness": 20.0,
    "lower_fillet_radius": 2.2,
    "upper_fillet_radius": 2.6,
}

MIN_OFFSET_MM = {
    # Bore bore: inner radius tolerance on a ~24mm bore → ±0.15mm is realistic ISO H7/h6
    "bore_radius_inner":        -0.15,
    # Axial heights: typical turning/grinding tolerance ±0.3–0.5mm for these dimensions
    "bore_height":              -0.40,
    "bore_thickness":           -0.50,
    "lower_transition_height":  -0.40,
    "web_height":               -0.60,   # longest feature, slightly wider tolerance
    "web_thickness":            -0.40,   # most life-sensitive bulk dimension
    "upper_transition_height":  -0.40,
    "rim_height":               -0.40,
    "rim_thickness":            -0.50,
    # Fillet radii: form-ground or EDM'd, ±0.10mm is achievable and life-critical
    "lower_fillet_radius":      -0.10,
    "upper_fillet_radius":      -0.10,
}

MAX_OFFSET_MM = {
    "bore_radius_inner":        +0.15,
    "bore_height":              +0.40,
    "bore_thickness":           +0.50,
    "lower_transition_height":  +0.40,
    "web_height":               +0.60,
    "web_thickness":            +0.40,
    "upper_transition_height":  +0.40,
    "rim_height":               +0.40,
    "rim_thickness":            +0.50,
    "lower_fillet_radius":      +0.10,
    "upper_fillet_radius":      +0.10,
}

REPRESENTATIONS = ("edge", "edge_proximity", "full")
THICKNESS_ORDERING_GAP_MM = 0.5

# ---------------------------------------------------------------------------
# Sub-zone identifiers for fine-grained region labeling.
# These refine the existing ZONE_NAME_TO_ID labels without replacing them.
# zone_id remains the primary label for S-N curve selection; subzone_id is
# available as an additional feature channel for geometry-aware models.
# ---------------------------------------------------------------------------
SUBZONE_NAME_TO_ID: Dict[str, int] = {
    "bore":               0,
    "lower_transition":   1,
    "web":                2,
    "upper_transition":   3,
    "rim_main":           4,
    "front_face":         5,   # front axial face (above and below C-groove)
    "front_cgroove":      6,   # C-groove: entry fillet, walls, floor, exit fillet
    "rear_arm_neck":      7,   # arm root neck face + shelf + top corner + ligament at top
    "rear_arm_land":      8,   # arm body left face + arm land (horizontal at r5+h_arm)
    "rear_arm_corner":    9,   # arm outer corner fillet
    "rear_arm_end_face": 10,   # arm rear end face (load-transfer face)
}
SUBZONE_ID_TO_NAME: Dict[int, str] = {v: k for k, v in SUBZONE_NAME_TO_ID.items()}

# Map from existing zone_name to subzone for the non-feature parts of the disc.
# Rim-feature subzone labels are assigned explicitly during outer-cap construction.
ZONE_TO_SUBZONE: Dict[str, str] = {
    "bore":               "bore",
    "lower_transition":   "lower_transition",
    "web":                "web",
    "upper_transition":   "upper_transition",
    "rim":                "rim_main",
}

# ---------------------------------------------------------------------------
# Rim-feature geometry parameters: front C-groove + rear annular drive arm.
#
# Front C-groove: externally open side relief cut into the FRONT axial face
#   of the rim (at x = -t_rim/2). The groove is open toward the front exterior
#   and leaves a finite load-carrying ligament between the groove floor and the
#   rear drive-arm root.
#
# Rear drive arm: annular axial projection beyond the rim core (at x > +t_rim/2).
#   The arm has a visible narrow neck/root, a step shelf, a body section, a
#   horizontal land at r5+h_arm, an outer corner fillet, and a load-transfer
#   end face.  The blade-equivalent centrifugal resultant is applied as a
#   distributed traction on this end face.
#
# All parameters are in mm. Nominal values are chosen so that the geometry is
# robustly meshable with the medium LC settings (LC_EDGE=0.50, LC_FILLET=0.30).
# ---------------------------------------------------------------------------

RIM_FEATURE_PARAMETERS = (
    # Front C-groove
    "front_cgroove_axial_depth",     # axial penetration from x_front inward
    "front_cgroove_radial_span",     # radial height of the groove opening
    "front_cgroove_radial_pos",      # r offset of groove bottom above r5 (rim outer)
    "front_cgroove_entry_radius",    # entry fillet radius (UP→RIGHT at groove mouth)
    "front_cgroove_floor_radius",    # floor corner fillet radius (inner corners)
    "front_cgroove_exit_radius",     # exit fillet radius (LEFT→UP above groove)
    # Rear annular drive arm
    "rear_arm_axial_projection",     # axial extent of arm beyond x_rear (= t_rim/2)
    "rear_arm_radial_height",        # radial height of arm body above r5
    "rear_arm_neck_thickness",       # radial height of arm root/neck (< radial_height)
    "rear_arm_root_radius",          # fillet radius at arm root transitions
    "rear_arm_outer_corner_radius",  # fillet radius at arm outer (top-rear) corner
)

COUPLED_CGROOVE_PARAMETERS = (
    "front_cgroove_radial_pos",
    "front_cgroove_radial_span",
    "front_cgroove_entry_radius",
    "front_cgroove_floor_radius",
    "front_cgroove_exit_radius",
)

NON_COUPLED_RIM_FEATURE_PARAMETERS = tuple(
    p for p in RIM_FEATURE_PARAMETERS if p not in COUPLED_CGROOVE_PARAMETERS
)

CGROOVE_SAMPLING_CONTROLS = (
    "cgroove_radial_pos_control",
    "cgroove_span_fraction",
    "cgroove_entry_radius_fraction",
    "cgroove_floor_radius_fraction",
    "cgroove_exit_radius_fraction",
)

NOMINAL_RIM_FEATURE_MM: Dict[str, float] = {
    "front_cgroove_axial_depth":    6.0,   # 6 mm inward from x_front (deeper groove)
    "front_cgroove_radial_span":    4.0,   # 4 mm groove height (wider groove)
    "front_cgroove_radial_pos":     1.0,   # groove bottom 1 mm above r5
    "front_cgroove_entry_radius":   0.8,   # entry fillet
    "front_cgroove_floor_radius":   0.8,   # floor corners
    "front_cgroove_exit_radius":    0.8,   # exit fillet
    "rear_arm_axial_projection":    8.0,   # arm extends 8 mm beyond x_rear (credible flange)
    "rear_arm_radial_height":       8.0,   # arm body is 8 mm tall above r5 (credible rim feature)
    "rear_arm_neck_thickness":      4.0,   # neck is 4 mm tall (thick enough to be credible)
    "rear_arm_root_radius":         1.0,   # generous root/transition fillets
    "rear_arm_outer_corner_radius": 1.0,   # generous outer arm corner fillet
}

MIN_RIM_FEATURE_OFFSET_MM: Dict[str, float] = {
    "front_cgroove_axial_depth":    -1.50,
    "front_cgroove_radial_span":    -1.00,
    "front_cgroove_radial_pos":     -0.30,
    "front_cgroove_entry_radius":   -0.20,
    "front_cgroove_floor_radius":   -0.20,
    "front_cgroove_exit_radius":    -0.20,
    "rear_arm_axial_projection":    -1.50,
    "rear_arm_radial_height":       -1.00,
    "rear_arm_neck_thickness":      -0.80,
    "rear_arm_root_radius":         -0.20,
    "rear_arm_outer_corner_radius": -0.20,
}

MAX_RIM_FEATURE_OFFSET_MM: Dict[str, float] = {
    "front_cgroove_axial_depth":    +1.50,
    "front_cgroove_radial_span":    +1.00,
    "front_cgroove_radial_pos":     +0.30,
    "front_cgroove_entry_radius":   +0.20,
    "front_cgroove_floor_radius":   +0.20,
    "front_cgroove_exit_radius":    +0.20,
    "rear_arm_axial_projection":    +2.00,
    "rear_arm_radial_height":       +2.00,
    "rear_arm_neck_thickness":      +1.00,
    "rear_arm_root_radius":         +0.30,
    "rear_arm_outer_corner_radius": +0.30,
}

MIN_CGROOVE_CONTROL = {
    "cgroove_radial_pos_control": 0.10,
    "cgroove_span_fraction": 0.10,
    "cgroove_entry_radius_fraction": 0.10,
    "cgroove_floor_radius_fraction": 0.10,
    "cgroove_exit_radius_fraction": 0.10,
}

MAX_CGROOVE_CONTROL = {
    "cgroove_radial_pos_control": 0.90,
    "cgroove_span_fraction": 0.90,
    "cgroove_entry_radius_fraction": 0.90,
    "cgroove_floor_radius_fraction": 0.90,
    "cgroove_exit_radius_fraction": 0.90,
}

# ---------------------------------------------------------------------------
# Blade-equivalent centrifugal load (annular/axisymmetric surrogate)
# Physical basis: annular average of N blades of mass m_blade each,
# rotating at omega_ref with CG at r_cg.
# F_total = N * m_blade * omega_ref^2 * r_cg  [N]
#
# Applied as radial traction on the rim-top / blade-attachment surface:
# the horizontal boundary at r = r5 + h_arm (ligament + arm land face).
# This represents blades pulling the disc rim radially outward through the
# blade-root attachment region at the outer rim.
#
# The rear drive arm receives NO direct blade load by default.
# The arm experiences stress only through structural continuity with the
# loaded rim, disc centrifugal body force, and internal redistribution.
#
# This load is fixed (identical for every generated sample) — it is not part
# of the LHS-sampled parameter space.
# ---------------------------------------------------------------------------
BLADE_EQUIV_NUM_BLADES: int = 60        # representative blade count
BLADE_EQUIV_MASS_KG: float = 0.003      # mass per blade [kg]; 3 g each → ~331 kN total at 4000 rad/s
#   F = 60 × 0.003 kg × (4000 rad/s)² × 0.115 m ≈ 331 kN ≈ 25 % of rim centrifugal
#   This is physically modest and adds a visible but non-dominating rim-load contribution.
BLADE_EQUIV_CG_RADIUS_MM: float = 115.0  # blade CG radius [mm] (outboard of rim)

CYCLE_PHASES = (
    "taxi",
    "takeoff",
    "climb",
    "cruise",
    "descent",
    "reverse_thrust",
    "taxi_return",
)
CYCLE_SPEED_FACTORS = np.array([0.20, 1.00, 0.86, 0.78, 0.55, 0.46, 0.18], dtype=np.float64)
CYCLE_PHASE_WEIGHTS = np.array([0.20, 0.08, 0.15, 0.32, 0.12, 0.05, 0.08], dtype=np.float64)


# ---------------------------------------------------------------------------
# S-N (stress-life) fatigue parameters — synthetic zonal lifing curves.
#
# Physical basis for zonal discontinuities:
#   In real engineering disc lifing, per-zone S-N allowables differ due to:
#     - Surface treatment: bore is shot-peened (compressive residual stress ->
#       higher allowable), web/rim as-machined (lower allowable), transition
#       fillets lifed from notched specimen curves (steeper slope, lower knee).
#     - Inspection interval: bore inner surface is accessible; fillet roots
#       have shorter mandatory replacement lives per EASA/FAA Part 33.
#     - Material certification: allowables are zone-specific in OEM lifing
#       manuals (e.g. Rolls-Royce, GE). Step changes at zone boundaries are
#       therefore physically justified.
#   Zonal discontinuities are intentionally retained for the ML ablation study
#   (testing whether models learn that zone label adds information beyond
#   geometry alone — a genuinely meaningful engineering question).
#
# Calibration rationale:
#   FEM at OMEGA_REF_RAD_S=4000 rad/s gives von Mises range ~180-620 MPa.
#   Stress amplitude: sigma_a = 0.5 * phase_vm (ground-air-ground R=0 LCF).
#   At takeoff (factor=1.0): sigma_a ~ 90-310 MPa across the disc.
#
#   Knee stresses and slopes are set so that:
#     - Fillet zones (lower/upper_transition): steep slope_high=13 + low knee
#       -> LCF lives 1e4-1e5 at the peak fillet stress concentrator.
#       slope_high=13 is physically justified for notched Ti-6Al-4V specimens
#       (steeper Basquin slope than smooth bar due to stress gradient effect).
#     - Bore knee is HIGH (shot-peen benefit) -> bore lives 1e7-1e9 even though
#       bore sigma_a is large, reflecting real peened allowables.
#     - Web/rim sit near knee -> intermediate lives 1e6-1e8.
#     - Overall range ~4 orders of magnitude (1e4-1e8+) for meaningful ML targets.
#
#   slope_low = 4-5: shallow sub-knee branch, physical for Ti-6Al-4V near the
#   endurance limit. Prevents 1e16+ runout that collapses the ML target range.
#   slope_high = 8-13: Basquin exponent above knee; smooth bar 8-10, notched
#   fillet specimens 12-14 (steeper due to stress concentration sensitivity).
#   knee_life: fillets at 5e6, bulk zones at 1e7.
#
#   These are synthetic allowables for ML dataset generation, not certified
#   material data. The zonal structure mirrors real OEM lifing practice.
# ---------------------------------------------------------------------------

ZONAL_SN_PARAMS: Dict[str, Dict[str, float]] = {
    "bore": {
        # Shot-peened bore: high knee reflects compressive residual stress benefit.
        # bore sigma_a ~175-225 MPa sits BELOW knee -> long lives 1e7-1e9.
        # Physically correct: peened bore outlives the unpeened fillet root.
        "knee_stress_mpa": 210.0,
        "knee_life": 1.0e7,
        "slope_high": 9.5,
        "slope_low": 4.0,
    },
    "lower_transition": {
        # Fillet root: notched specimen allowable. steep slope_high=13 reflects
        # stress-gradient sensitivity of notched Ti-6Al-4V (literature: 12-14).
        # fillet peak sigma_a ~310 MPa >> knee 200 MPa -> LCF lives 1e4-1e5.
        "knee_stress_mpa": 200.0,
        "knee_life": 5.0e6,
        "slope_high": 13.0,
        "slope_low": 4.5,
    },
    "web": {
        # As-machined web: moderate knee, intermediate lives 1e6-1e8.
        "knee_stress_mpa": 140.0,
        "knee_life": 1.0e7,
        "slope_high": 8.5,
        "slope_low": 4.0,
    },
    "upper_transition": {
        # Upper fillet: same notched allowable logic as lower_transition.
        # upper fillet sigma_a ~150-200 MPa near/above knee -> 1e4-1e6.
        "knee_stress_mpa": 180.0,
        "knee_life": 5.0e6,
        "slope_high": 13.0,
        "slope_low": 4.5,
    },
    "rim": {
        # As-machined rim: low sigma_a (~100-125 MPa) near knee -> 1e7-1e9.
        "knee_stress_mpa": 120.0,
        "knee_life": 1.0e7,
        "slope_high": 9.0,
        "slope_low": 4.0,
    },
}
# Uniform mode: a single S-N curve for every zone, equal to the web-zone set.
UNIFORM_SN_PARAMS: Dict[str, float] = dict(ZONAL_SN_PARAMS["web"])


@dataclass(frozen=True)
class SampleGenerationConfig:
    contour_points_per_side: int = 220
    mesh_grid_points_x: int = 90
    mesh_grid_points_r: int = 130
    edge_proximity_distance_mm: float = 2.0


def _assert_all_keys(table: Dict[str, float], reference_keys: Iterable[str], table_name: str) -> None:
    missing = sorted(set(reference_keys) - set(table.keys()))
    extras = sorted(set(table.keys()) - set(reference_keys))
    if missing or extras:
        raise ValueError(f"{table_name} key mismatch; missing={missing}, extras={extras}")


def validate_config_tables() -> None:
    _assert_all_keys(NOMINAL_GEOMETRY_MM, PUBLIC_GEOMETRY_PARAMETERS, "NOMINAL_GEOMETRY_MM")
    _assert_all_keys(MIN_OFFSET_MM, PUBLIC_GEOMETRY_PARAMETERS, "MIN_OFFSET_MM")
    _assert_all_keys(MAX_OFFSET_MM, PUBLIC_GEOMETRY_PARAMETERS, "MAX_OFFSET_MM")
    bt = float(NOMINAL_GEOMETRY_MM["bore_thickness"])
    rt = float(NOMINAL_GEOMETRY_MM["rim_thickness"])
    wt = float(NOMINAL_GEOMETRY_MM["web_thickness"])
    if not (bt > rt > wt):
        raise ValueError("Nominal thickness ordering must satisfy bore_thickness > rim_thickness > web_thickness")


def resolve_geometry_parameters(param_offsets: Dict[str, float] | None) -> Dict[str, float]:
    """Apply mandatory nominal + offset model."""
    validate_config_tables()
    offsets = {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS}
    if param_offsets is not None:
        unknown = sorted(set(param_offsets.keys()) - set(PUBLIC_GEOMETRY_PARAMETERS))
        if unknown:
            raise ValueError(f"Unknown geometry offsets: {unknown}")
        for k, v in param_offsets.items():
            offsets[k] = float(v)

    actual = {
        k: float(NOMINAL_GEOMETRY_MM[k] + offsets[k])
        for k in PUBLIC_GEOMETRY_PARAMETERS
    }
    return actual


def clip_offsets_to_bounds(param_offsets: Dict[str, float]) -> Dict[str, float]:
    """Clip offsets to configured min/max bounds."""
    out: Dict[str, float] = {}
    for k in PUBLIC_GEOMETRY_PARAMETERS:
        v = float(param_offsets.get(k, 0.0))
        out[k] = float(np.clip(v, MIN_OFFSET_MM[k], MAX_OFFSET_MM[k]))
    return out


def offset_vector_to_dict(vector: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(PUBLIC_GEOMETRY_PARAMETERS, vector)}


def offsets_dict_to_vector(offsets: Dict[str, float]) -> np.ndarray:
    return np.array([float(offsets.get(k, 0.0)) for k in PUBLIC_GEOMETRY_PARAMETERS], dtype=np.float64)


def radial_stations_from_params(params: Dict[str, float]) -> np.ndarray:
    """Return [r0, r1, r2, r3, r4, r5] from required radial-threshold geometry keys.

    Required keys in `params` (all in mm): bore_radius_inner, bore_height,
    lower_transition_height, web_height, upper_transition_height, rim_height.
    """
    r0 = float(params["bore_radius_inner"])
    r1 = r0 + float(params["bore_height"])
    r2 = r1 + float(params["lower_transition_height"])
    r3 = r2 + float(params["web_height"])
    r4 = r3 + float(params["upper_transition_height"])
    r5 = r4 + float(params["rim_height"])
    return np.array([r0, r1, r2, r3, r4, r5], dtype=np.float64)


def resolve_rim_feature_parameters(rim_feature_offsets: Dict[str, float] | None) -> Dict[str, float]:
    """Return actual rim-feature parameter values by applying offsets to nominal."""
    actual: Dict[str, float] = {}
    for k in RIM_FEATURE_PARAMETERS:
        offset = float(rim_feature_offsets.get(k, 0.0)) if rim_feature_offsets else 0.0
        actual[k] = float(NOMINAL_RIM_FEATURE_MM[k] + offset)
    return actual


def clip_rim_feature_offsets_to_bounds(rim_feature_offsets: Dict[str, float]) -> Dict[str, float]:
    """Clip rim-feature offsets to their configured min/max bounds."""
    out: Dict[str, float] = {}
    for k in RIM_FEATURE_PARAMETERS:
        v = float(rim_feature_offsets.get(k, 0.0))
        out[k] = float(np.clip(v, MIN_RIM_FEATURE_OFFSET_MM[k], MAX_RIM_FEATURE_OFFSET_MM[k]))
    return out


def rim_feature_offset_vector_to_dict(vector: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(RIM_FEATURE_PARAMETERS, vector)}


def rim_feature_offsets_dict_to_vector(offsets: Dict[str, float]) -> np.ndarray:
    return np.array([float(offsets.get(k, 0.0)) for k in RIM_FEATURE_PARAMETERS], dtype=np.float64)


def clip_cgroove_controls_to_bounds(controls: Dict[str, float] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in CGROOVE_SAMPLING_CONTROLS:
        v = 0.5 if controls is None else float(controls.get(k, 0.5))
        out[k] = float(np.clip(v, MIN_CGROOVE_CONTROL[k], MAX_CGROOVE_CONTROL[k]))
    return out


def _lerp(lo: float, hi: float, u: float) -> float:
    if hi <= lo:
        return float(lo)
    return float(lo + u * (hi - lo))


def map_cgroove_controls_to_parameters(
    controls: Dict[str, float],
    resolved_rim: Dict[str, float],
    t_rim: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Map normalized C-groove controls to physically coupled mm parameters.

    Mapping uses the same inequalities enforced in sanitize_rim_feature_parameters,
    with conservative clearance margins to avoid aggressive edge-filling.
    """
    c = clip_cgroove_controls_to_bounds(controls)

    h_arm = min(max(float(resolved_rim["rear_arm_radial_height"]), 2.0), 0.55 * float(t_rim))
    neck_t = min(max(float(resolved_rim["rear_arm_neck_thickness"]), 0.8), h_arm - 1.0)
    rf_root_max = max(min(0.45 * neck_t, 0.45 * (h_arm - neck_t)), 0.2)
    rf_root = min(max(float(resolved_rim["rear_arm_root_radius"]), 0.2), rf_root_max)

    def _cfg_bounds(key: str) -> tuple[float, float]:
        lo = float(NOMINAL_RIM_FEATURE_MM[key] + MIN_RIM_FEATURE_OFFSET_MM[key])
        hi = float(NOMINAL_RIM_FEATURE_MM[key] + MAX_RIM_FEATURE_OFFSET_MM[key])
        return lo, hi

    pos_lo_cfg, pos_hi_cfg = _cfg_bounds("front_cgroove_radial_pos")
    span_lo_cfg, span_hi_cfg = _cfg_bounds("front_cgroove_radial_span")
    entry_lo_cfg, entry_hi_cfg = _cfg_bounds("front_cgroove_entry_radius")
    floor_lo_cfg, floor_hi_cfg = _cfg_bounds("front_cgroove_floor_radius")
    exit_lo_cfg, exit_hi_cfg = _cfg_bounds("front_cgroove_exit_radius")

    geometric_clearance_mm = 0.30
    span_floor_req = floor_lo_cfg / 0.225 + 0.05
    span_lo_target = max(span_lo_cfg, span_floor_req)

    pos_hi_phys = h_arm - (rf_root + geometric_clearance_mm + span_lo_target + entry_lo_cfg + exit_lo_cfg)
    pos_lo = max(pos_lo_cfg, 0.3)
    pos_hi = min(pos_hi_cfg, pos_hi_phys)
    if pos_hi < pos_lo:
        pos_hi = pos_lo
    cg_pos = _lerp(pos_lo, pos_hi, c["cgroove_radial_pos_control"])

    entry_hi_phys = min(
        0.8 * cg_pos,
        h_arm - (cg_pos + span_lo_target + exit_lo_cfg + rf_root + geometric_clearance_mm),
    )
    entry_lo = entry_lo_cfg
    entry_hi = min(entry_hi_cfg, entry_hi_phys)
    if entry_hi < entry_lo:
        entry_hi = entry_lo
    rf_entry = _lerp(entry_lo, entry_hi, c["cgroove_entry_radius_fraction"])

    span_hi_phys = h_arm - (cg_pos + rf_entry + exit_lo_cfg + rf_root + geometric_clearance_mm)
    span_lo = span_lo_target
    span_hi = min(span_hi_cfg, span_hi_phys)
    if span_hi < span_lo:
        span_hi = span_lo
    cg_span = _lerp(span_lo, span_hi, c["cgroove_span_fraction"])

    exit_hi_phys = min(
        0.45 * min(max(h_arm - cg_pos - cg_span, 0.0), cg_span),
        h_arm - (cg_pos + rf_entry + cg_span + rf_root + geometric_clearance_mm),
    )
    exit_lo = exit_lo_cfg
    exit_hi = min(exit_hi_cfg, exit_hi_phys)
    if exit_hi < exit_lo:
        exit_hi = exit_lo
    rf_exit = _lerp(exit_lo, exit_hi, c["cgroove_exit_radius_fraction"])

    required_h = cg_pos + rf_entry + cg_span + rf_exit + rf_root + geometric_clearance_mm
    allowed_h = h_arm
    if required_h > allowed_h:
        cg_span = max(span_lo_cfg, cg_span - (required_h - allowed_h))

    floor_cap = 0.225 * cg_span
    floor_lo = floor_lo_cfg
    floor_hi = min(floor_hi_cfg, floor_cap)
    if floor_hi < floor_lo:
        floor_hi = floor_lo
    rf_floor = _lerp(floor_lo, floor_hi, c["cgroove_floor_radius_fraction"])

    params = {
        "front_cgroove_radial_pos": float(cg_pos),
        "front_cgroove_radial_span": float(cg_span),
        "front_cgroove_entry_radius": float(rf_entry),
        "front_cgroove_floor_radius": float(rf_floor),
        "front_cgroove_exit_radius": float(rf_exit),
    }
    for key in COUPLED_CGROOVE_PARAMETERS:
        lo, hi = _cfg_bounds(key)
        params[key] = float(np.clip(params[key], lo, hi))

    required_h_after_clip = (
        params["front_cgroove_radial_pos"]
        + params["front_cgroove_entry_radius"]
        + params["front_cgroove_radial_span"]
        + params["front_cgroove_exit_radius"]
        + rf_root
        + geometric_clearance_mm
    )
    if required_h_after_clip > h_arm - 0.05:
        overflow = required_h_after_clip - (h_arm - 0.05)
        params["front_cgroove_radial_span"] = max(
            float(NOMINAL_RIM_FEATURE_MM["front_cgroove_radial_span"] + MIN_RIM_FEATURE_OFFSET_MM["front_cgroove_radial_span"]),
            params["front_cgroove_radial_span"] - overflow,
        )

    meta = {
        "effective_h_arm_mm": float(h_arm),
        "effective_neck_thickness_mm": float(neck_t),
        "effective_root_radius_mm": float(rf_root),
        "geometric_clearance_mm": float(geometric_clearance_mm),
        "configured_floor_lower_bound_mm": float(floor_lo_cfg),
        "span_min_required_for_floor_lower_mm": float(span_floor_req),
        "position_max_phys_mm": float(pos_hi_phys),
        "position_sampling_min_mm": float(pos_lo),
        "position_sampling_max_mm": float(pos_hi),
        "entry_sampling_min_mm": float(entry_lo),
        "entry_sampling_max_mm": float(entry_hi),
        "span_sampling_min_mm": float(span_lo),
        "span_sampling_max_mm": float(span_hi),
        "exit_sampling_min_mm": float(exit_lo),
        "exit_sampling_max_mm": float(exit_hi),
        "floor_cap_mm": float(floor_cap),
        "floor_sampling_min_mm": float(floor_lo),
        "floor_sampling_max_mm": float(floor_hi),
    }
    return params, meta
