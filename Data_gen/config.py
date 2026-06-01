"""Centralized configuration for synthetic rotor-disc data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


REGION_NAME_TO_ID = {"bore": 0, "web": 1, "rim": 2}
REGION_ID_TO_NAME = {v: k for k, v in REGION_NAME_TO_ID.items()}

CYCLE_PHASES = (
    "taxi",
    "takeoff",
    "climb",
    "cruise",
    "descent",
    "reverse_thrust",
    "taxi_return",
)
CYCLE_SPEED_FACTORS = np.array([0.20, 1.00, 0.85, 0.78, 0.55, 0.45, 0.18], dtype=float)
CYCLE_PHASE_WEIGHTS = np.array([0.20, 0.08, 0.15, 0.32, 0.12, 0.05, 0.08], dtype=float)


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level generation configuration."""

    num_samples: int = 100
    seed: int = 7
    output_dir: Path = Path("Data_gen/output")
    contour_points_per_segment: int = 40
    mesh_grid_points_x: int = 80
    mesh_grid_points_r: int = 110
    edge_proximity_distance_mm: float = 2.0
    save_validation_plots: bool = False
    validation_plot_count: int = 3


NOMINAL_GEOMETRY_MM: Dict[str, float] = {
    "bore_radius": 20.0,
    "bore_thickness": 14.0,
    "web_length": 38.0,
    "web_thickness": 9.0,
    "web_slope": 0.23,
    "bore_web_fillet_radius": 2.2,
    "rim_length": 18.0,
    "rim_thickness": 16.0,
    "web_rim_fillet_radius": 2.6,
}

PERTURBATION_MM: Dict[str, float] = {
    "bore_radius": 2.2,
    "bore_thickness": 1.8,
    "web_length": 5.0,
    "web_thickness": 1.8,
    "web_slope": 0.06,
    "bore_web_fillet_radius": 0.8,
    "rim_length": 3.2,
    "rim_thickness": 2.8,
    "web_rim_fillet_radius": 0.9,
}


def sample_geometry_parameters(rng: np.random.Generator) -> Dict[str, float]:
    """Sample signed perturbations around nominal geometry."""
    params = {}
    for key, nominal in NOMINAL_GEOMETRY_MM.items():
        delta = PERTURBATION_MM[key]
        params[key] = float(nominal + rng.uniform(-delta, delta))
    params["bore_thickness"] = max(params["bore_thickness"], 6.0)
    params["web_length"] = max(params["web_length"], 10.0)
    params["web_thickness"] = max(params["web_thickness"], 4.0)
    params["rim_length"] = max(params["rim_length"], 6.0)
    params["rim_thickness"] = max(params["rim_thickness"], 5.0)
    params["bore_web_fillet_radius"] = max(params["bore_web_fillet_radius"], 0.4)
    params["web_rim_fillet_radius"] = max(params["web_rim_fillet_radius"], 0.4)
    return params

