"""Cycle loading, equivalent stress, and raw life computation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from skfem import Basis, ElementTriP1, LinearForm, asm, condense, solve
from skfem.models.poisson import laplace

from .config import CYCLE_PHASE_WEIGHTS, CYCLE_SPEED_FACTORS


REGION_STRESS_SCALE = np.array([1.10, 1.00, 1.30], dtype=float)

# Basquin-style region coefficients, intentionally discontinuous by region.
REGION_BASQUIN_C = np.array([1.2e13, 6.0e12, 1.8e13], dtype=float)
REGION_BASQUIN_M = np.array([4.5, 5.1, 5.8], dtype=float)


def compute_phase_equivalent_stresses(
    mesh,
    nodes: np.ndarray,
    region_ids: np.ndarray,
    geometry_params: Dict[str, float],
) -> np.ndarray:
    """Compute per-node equivalent stress for each mission phase."""
    basis = Basis(mesh, ElementTriP1())
    A = asm(laplace, basis)

    @LinearForm
    def rhs(v, w):
        return (w.x[1] + 1e-9) * v

    b = asm(rhs, basis)
    bnodes = mesh.boundary_nodes()
    potential = solve(*condense(A, b, D=bnodes))
    potential = np.abs(potential)
    potential /= potential.max() + 1e-12

    geom_gain = 1.0 + 0.002 * (
        geometry_params["rim_thickness"] + geometry_params["web_length"] + geometry_params["bore_radius"]
    )
    base = 420.0 * potential * REGION_STRESS_SCALE[region_ids] * geom_gain + 5.0
    phase_stress = base[:, None] * (CYCLE_SPEED_FACTORS[None, :] ** 2)
    return phase_stress.astype(np.float64)


def compute_stress_max(phase_stress: np.ndarray) -> np.ndarray:
    return np.max(phase_stress, axis=1)


def compute_life_raw(phase_stress: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    """Miner accumulation over one fixed cycle with region-specific nonlinear S-N."""
    sigma = np.maximum(phase_stress, 1e-6)
    c = REGION_BASQUIN_C[region_ids][:, None]
    m = REGION_BASQUIN_M[region_ids][:, None]
    n_fail = c * np.power(sigma, -m)
    damage_per_cycle = np.sum(CYCLE_PHASE_WEIGHTS[None, :] / (n_fail + 1e-20), axis=1)
    return (1.0 / np.maximum(damage_per_cycle, 1e-20)).astype(np.float64)
