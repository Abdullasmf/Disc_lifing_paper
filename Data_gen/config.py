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

MIN_OFFSET_MM: Dict[str, float] = {
    "bore_radius_inner":        -0.5,
    "bore_height":              -0.5,
    "bore_thickness":           -0.8,
    "lower_transition_height":  -0.4,
    "web_height":               -1.0,
    "web_thickness":            -0.5,
    "upper_transition_height":  -0.4,
    "rim_height":               -0.8,
    "rim_thickness":            -0.8,
    "lower_fillet_radius":      -0.3,
    "upper_fillet_radius":      -0.3,
}

MAX_OFFSET_MM: Dict[str, float] = {
    "bore_radius_inner":        +0.5,
    "bore_height":              +0.5,
    "bore_thickness":           +0.8,
    "lower_transition_height":  +0.4,
    "web_height":               +1.0,
    "web_thickness":            +0.5,
    "upper_transition_height":  +0.4,
    "rim_height":               +0.8,
    "rim_thickness":            +0.8,
    "lower_fillet_radius":      +0.3,
    "upper_fillet_radius":      +0.3,
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
# S-N (stress-life) fatigue parameters for the life model.
#
# Calibration rationale (synthetic dataset for ML benchmarking):
#   The axisymmetric FEM at OMEGA_REF_RAD_S=4000 rad/s on the nominal geometry
#   produces von Mises stresses in the range ~180-620 MPa. The stress amplitude
#   entering the S-N curve is sigma_a = 0.5 * phase_vm (ground-air-ground LCF
#   convention, R=0 from rest). At takeoff (speed_factor=1.0) this gives
#   sigma_a ~ 90-310 MPa across the disc. At the lowest phase (taxi_return,
#   speed_factor=0.18, scale=0.032) sigma_a drops to ~3-20 MPa.
#
#   Knee stresses are set within the sigma_a range so that:
#     - Critical fillet zones (lower/upper_transition) sit above the knee at
#       takeoff -> short LCF lives (1e5-1e6 cycles) as expected physically.
#     - Bulk bore/web/rim nodes sit near or slightly above the knee at takeoff
#       -> moderate lives (1e6-1e9 cycles) giving meaningful ML targets.
#     - No zone is permanently sub-knee at all phases -> avoids 1e16+ runout.
#
#   These are synthetic allowables tuned to produce a physically plausible life
#   distribution for ML dataset generation, not certified material data.
#
#   slope_high: Basquin exponent above the knee (~8-12 for Ti-6Al-4V HCF).
#   slope_low:  Sub-knee branch exponent; larger than slope_high so the curve
#               flattens in log(sigma)-log(N) space.
#   knee_life:  Transition cycle count; transition zones at 5e6, bulk at 1e7.
# ---------------------------------------------------------------------------
SIGMA_E0_MPA = 620.0
ZONE_KNOCKDOWN = {
    "bore": 0.80,
    "lower_transition": 0.75,
    "web": 0.90,
    "upper_transition": 0.75,
    "rim": 0.85,
}

ZONAL_SN_PARAMS: Dict[str, Dict[str, float]] = {
    "bore": {
        # knee calibrated to ~bore takeoff sigma_a (0.5 * ~160 MPa vm) = ~80 MPa
        "knee_stress_mpa": 80.0,
        "knee_life": 1.0e7,
        "slope_high": 9.5,
        "slope_low": 22.0,
    },
    "lower_transition": {
        # knee calibrated below fillet peak sigma_a (~310 MPa) to give LCF lives
        "knee_stress_mpa": 60.0,
        "knee_life": 5.0e6,
        "slope_high": 11.0,
        "slope_low": 24.0,
    },
    "web": {
        # knee calibrated to ~web takeoff sigma_a (0.5 * ~200 MPa vm) = ~100 MPa
        "knee_stress_mpa": 100.0,
        "knee_life": 1.0e7,
        "slope_high": 8.5,
        "slope_low": 20.0,
    },
    "upper_transition": {
        # knee calibrated below upper fillet sigma_a to give LCF lives
        "knee_stress_mpa": 60.0,
        "knee_life": 5.0e6,
        "slope_high": 11.0,
        "slope_low": 24.0,
    },
    "rim": {
        # knee calibrated to ~rim takeoff sigma_a (0.5 * ~150 MPa vm) = ~75 MPa
        "knee_stress_mpa": 75.0,
        "knee_life": 1.0e7,
        "slope_high": 9.0,
        "slope_low": 21.0,
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
