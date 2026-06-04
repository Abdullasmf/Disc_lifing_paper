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
    "bore_thickness": 20.0,
    "lower_transition_height": 8.0,
    "web_height": 40.0,
    "web_thickness": 10.5,
    "upper_transition_height": 9.0,
    "rim_height": 15.0,
    "rim_thickness": 24.0,
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
