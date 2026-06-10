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
    "bore_radius_inner": -4.0,
    "bore_height": -3.0,
    "bore_thickness": -4.0,
    "lower_transition_height": -2.5,
    "web_height": -8.0,
    "web_thickness": -2.8,
    "upper_transition_height": -2.5,
    "rim_height": -4.0,
    "rim_thickness": -4.5,
    "lower_fillet_radius": -1.5,
    "upper_fillet_radius": -1.6,
}

MAX_OFFSET_MM: Dict[str, float] = {
    "bore_radius_inner": 4.0,
    "bore_height": 3.0,
    "bore_thickness": 4.0,
    "lower_transition_height": 2.5,
    "web_height": 8.0,
    "web_thickness": 2.8,
    "upper_transition_height": 2.5,
    "rim_height": 4.0,
    "rim_thickness": 4.5,
    "lower_fillet_radius": 1.5,
    "upper_fillet_radius": 1.6,
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
# Derivation (Ti-6Al-4V, fully-reversed R=-1, polished baseline):
#   * Baseline high-cycle fatigue strength of Ti-6Al-4V is ~600 MPa at the knee
#     (~1e7 cycles) for smooth, polished, R=-1 specimens. We anchor the knee
#     stress to a base endurance limit SIGMA_E0 = 620 MPa and apply a
#     per-zone knockdown factor that lumps notch sensitivity, surface finish,
#     size and stress-gradient effects for each engineering zone:
#         bore=0.80, lower_transition=0.75, web=0.90,
#         upper_transition=0.75, rim=0.85
#     giving knee_stress_mpa = SIGMA_E0 * knockdown (≈465–558 MPa, within the
#     ~500–620 MPa target band; the transition shoulders are the harshest).
#   * knee_life is the cycle count at the knee: ~1e6–1e7. Stress-concentrating
#     transition shoulders are placed at the lower end (5e6) and the bulk
#     bore/web/rim at 1e7.
#   * slope_high is the Basquin exponent of the high-cycle (above-knee) branch.
#     Ti-6Al-4V high-cycle slopes lie in the ~8–12 range; notched transition
#     zones are slightly steeper (more stress-sensitive) than the bulk.
#   * slope_low is the shallower long-life (below-knee) branch used past the
#     knee, kept distinctly lower so the piecewise log-log curve flattens out.
# These are engineering allowables per zone, not random per-sample materials.
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
        "knee_stress_mpa": SIGMA_E0_MPA * ZONE_KNOCKDOWN["bore"],          # 496.0
        "knee_life": 1.0e7,
        "slope_high": 9.5,
        "slope_low": 22.0,
    },
    "lower_transition": {
        "knee_stress_mpa": SIGMA_E0_MPA * ZONE_KNOCKDOWN["lower_transition"],  # 465.0
        "knee_life": 5.0e6,
        "slope_high": 11.0,
        "slope_low": 24.0,
    },
    "web": {
        "knee_stress_mpa": SIGMA_E0_MPA * ZONE_KNOCKDOWN["web"],           # 558.0
        "knee_life": 1.0e7,
        "slope_high": 8.5,
        "slope_low": 20.0,
    },
    "upper_transition": {
        "knee_stress_mpa": SIGMA_E0_MPA * ZONE_KNOCKDOWN["upper_transition"],  # 465.0
        "knee_life": 5.0e6,
        "slope_high": 11.0,
        "slope_low": 24.0,
    },
    "rim": {
        "knee_stress_mpa": SIGMA_E0_MPA * ZONE_KNOCKDOWN["rim"],           # 527.0
        "knee_life": 1.0e7,
        "slope_high": 9.0,
        "slope_low": 21.0,
    },
}

# Uniform mode: a single S-N curve for every zone, equal to the web-zone set
# (the highest-knockdown bulk allowable per CHANGE 2c).
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
