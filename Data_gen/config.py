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
