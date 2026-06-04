"""Lightweight physically-motivated stress/life surrogate."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import CYCLE_PHASE_WEIGHTS, CYCLE_SPEED_FACTORS, REGION_NAME_TO_ID

REGION_STRESS_SCALE = np.array([1.08, 1.00, 1.13], dtype=np.float64)
# Region scaling by index: 0=bore, 1=web, 2=rim.

# Zone-specific Basquin fatigue coefficients (5 zones: bore, lower_transition, web, upper_transition, rim).
# Distinct values per zone are mandatory so the benchmark tests subregion-dependent life learning.
ZONE_BASQUIN_C = np.array([1.8e12, 9.0e11, 7.2e11, 8.5e11, 4.5e12], dtype=np.float64)
ZONE_BASQUIN_M = np.array([4.2, 4.6, 5.1, 4.8, 5.6], dtype=np.float64)

MIN_THICKNESS_MM = 1e-3
SIGMA_REF_MPA = 178.0
AMPLITUDE_HALF_RANGE = 0.5
AMPLITUDE_SPEED_BASE = 0.85
AMPLITUDE_SPEED_GAIN = 0.15
EPS = 1e-6


def _geometry_section_thickness(
    nodes: np.ndarray,
    params: Dict[str, float],
    radial_breaks: np.ndarray,
) -> np.ndarray:
    """Local thickness at each node = x_rear(r) - x_front(r).

    The disc profile is symmetric about x=0, so thickness(r) = t_profile(r) from
    the parameterised front/rear surfaces, not from any contour-envelope method.
    """
    from .geometry import _thickness_profile
    r = nodes[:, 1]
    return np.maximum(_thickness_profile(r, params, radial_breaks), MIN_THICKNESS_MM)


def _transition_concentration(
    nodes: np.ndarray,
    params: Dict[str, float],
    radial_breaks: np.ndarray,
) -> np.ndarray:
    """Stress concentration factor from transition bands, defined by radial position only.

    Concentration is computed from the node's r-coordinate relative to the lower
    [r1, r2] and upper [r3, r4] transition bands.  A Gaussian bell is centred on
    each band; the peak amplitude is modulated by the inverse of the fillet radius.
    This avoids the previous centrelline-stripe artifact caused by distance to a
    fixed 2-D landmark point.
    """
    r = nodes[:, 1]
    r1 = float(radial_breaks[1])
    r2 = float(radial_breaks[2])
    r3 = float(radial_breaks[3])
    r4 = float(radial_breaks[4])

    lower_width = max(r2 - r1, 1e-6)
    upper_width = max(r4 - r3, 1e-6)

    r_lower_mid = 0.5 * (r1 + r2)
    r_upper_mid = 0.5 * (r3 + r4)

    sigma_lower = max(0.4 * lower_width, 0.5)
    sigma_upper = max(0.4 * upper_width, 0.5)

    r_lower = max(params["lower_fillet_radius"], 0.3)
    r_upper = max(params["upper_fillet_radius"], 0.3)

    gain_lower = 0.55 * np.clip((2.8 / r_lower) ** 1.1, 0.35, 2.8)
    gain_upper = 0.45 * np.clip((2.8 / r_upper) ** 1.1, 0.35, 2.8)

    k_lower = 1.0 + gain_lower * np.exp(-0.5 * ((r - r_lower_mid) / sigma_lower) ** 2)
    k_upper = 1.0 + gain_upper * np.exp(-0.5 * ((r - r_upper_mid) / sigma_upper) ** 2)
    return k_lower * k_upper


def compute_phase_equivalent_stresses(
    nodes: np.ndarray,
    zone_ids: np.ndarray,
    region_ids: np.ndarray,
    geometry_params: Dict[str, float],
    radial_breaks: np.ndarray,
    contour_points: Optional[np.ndarray] = None,  # retained for call-site compatibility, not used
) -> np.ndarray:
    r = nodes[:, 1]
    r_inner = float(radial_breaks[0])
    r_outer = float(radial_breaks[5])
    span = max(r_outer - r_inner, EPS)
    r_norm = np.clip((r - r_inner) / span, 0.0, 1.0)

    radial_term = 0.30 + 0.70 * np.power(r_norm, 1.30)
    hoop_term = 0.55 + 1.32 * np.power(r_norm, 2.0)
    rotor_shape = 0.44 * radial_term + 0.56 * hoop_term

    section_thickness = _geometry_section_thickness(nodes, geometry_params, radial_breaks)
    t_ref = np.median(section_thickness)
    thin_amp = np.power(np.clip(t_ref / section_thickness, 0.55, 1.95), 0.72)

    transition_conc = _transition_concentration(nodes, geometry_params, radial_breaks)

    zone_multiplier = np.ones(nodes.shape[0], dtype=np.float64)
    zone_multiplier[zone_ids == 1] *= 1.03  # lower_transition
    zone_multiplier[zone_ids == 3] *= 1.03  # upper_transition

    geom_gain = 1.0 + 0.0012 * (
        geometry_params["rim_thickness"] + geometry_params["web_height"] + geometry_params["bore_radius_inner"]
    )

    base = (
        SIGMA_REF_MPA
        * rotor_shape
        * thin_amp
        * transition_conc
        * zone_multiplier
        * REGION_STRESS_SCALE[region_ids]
        * geom_gain
    )

    phase_scale = CYCLE_SPEED_FACTORS**2
    phase_stress = base[:, None] * phase_scale[None, :] + 12.0
    return phase_stress.astype(np.float64)


def compute_stress_max(phase_stress: np.ndarray) -> np.ndarray:
    return np.max(phase_stress, axis=1).astype(np.float64)


def compute_life_raw(phase_stress: np.ndarray, zone_ids: np.ndarray) -> np.ndarray:
    """Compute life using zone-specific Basquin fatigue parameters.

    Life must be computed from zone_ids (5 zones), not region_ids (3 regions),
    so that transition-zone fatigue behaviour is distinct from core web behaviour.
    """
    sigma_eq = np.maximum(phase_stress, EPS)
    amplitude_scale = AMPLITUDE_HALF_RANGE * (AMPLITUDE_SPEED_BASE + AMPLITUDE_SPEED_GAIN * CYCLE_SPEED_FACTORS)
    sigma_a = sigma_eq * amplitude_scale[None, :]

    basquin_c = ZONE_BASQUIN_C[zone_ids][:, None]
    basquin_m = ZONE_BASQUIN_M[zone_ids][:, None]
    n_fail = basquin_c * np.power(np.maximum(sigma_a, EPS), -basquin_m)

    damage_per_cycle = np.sum(CYCLE_PHASE_WEIGHTS[None, :] / np.maximum(n_fail, 1e-20), axis=1)
    return (1.0 / np.maximum(damage_per_cycle, 1e-20)).astype(np.float64)
