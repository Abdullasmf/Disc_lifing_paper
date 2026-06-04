"""Lightweight physically-motivated stress/life surrogate."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import CYCLE_PHASE_WEIGHTS, CYCLE_SPEED_FACTORS, REGION_NAME_TO_ID

REGION_STRESS_SCALE = np.array([1.08, 1.00, 1.13], dtype=np.float64)
REGION_BASQUIN_C = np.array([2.0e12, 7.2e11, 4.0e12], dtype=np.float64)
REGION_BASQUIN_M = np.array([4.2, 5.1, 5.6], dtype=np.float64)

MIN_THICKNESS_MM = 1e-3
SIGMA_REF_MPA = 178.0
AMPLITUDE_HALF_RANGE = 0.5
AMPLITUDE_SPEED_BASE = 0.85
AMPLITUDE_SPEED_GAIN = 0.15
EPS = 1e-6


def _geometry_section_thickness(nodes: np.ndarray, contour_points: np.ndarray) -> np.ndarray:
    cx = contour_points[:, 0]
    cr = contour_points[:, 1]

    sort_idx = np.argsort(cx)
    cx_s = cx[sort_idx]
    cr_s = cr[sort_idx]

    x_unique, inv = np.unique(cx_s, return_inverse=True)
    r_lower = np.full(x_unique.shape[0], np.inf, dtype=np.float64)
    r_upper = np.full(x_unique.shape[0], -np.inf, dtype=np.float64)
    np.minimum.at(r_lower, inv, cr_s)
    np.maximum.at(r_upper, inv, cr_s)

    node_x = np.clip(nodes[:, 0], x_unique[0], x_unique[-1])
    lower = np.interp(node_x, x_unique, r_lower)
    upper = np.interp(node_x, x_unique, r_upper)
    return np.maximum(upper - lower, MIN_THICKNESS_MM)


def _transition_concentration(
    nodes: np.ndarray,
    params: Dict[str, float],
    landmarks_mm: Dict[str, np.ndarray],
) -> np.ndarray:
    lower_marks = np.vstack([
        landmarks_mm["lower_transition_start"],
        landmarks_mm["lower_transition_end"],
    ])
    upper_marks = np.vstack([
        landmarks_mm["upper_transition_start"],
        landmarks_mm["upper_transition_end"],
    ])

    d_lower = np.min(np.linalg.norm(nodes[:, None, :] - lower_marks[None, :, :], axis=2), axis=1)
    d_upper = np.min(np.linalg.norm(nodes[:, None, :] - upper_marks[None, :, :], axis=2), axis=1)

    ll = max(1.4 * params["lower_fillet_radius"], 0.7)
    lu = max(1.4 * params["upper_fillet_radius"], 0.7)

    k_lower = 1.0 + 0.26 * np.exp(-(d_lower / ll) ** 2)
    k_upper = 1.0 + 0.20 * np.exp(-(d_upper / lu) ** 2)
    return k_lower * k_upper


def compute_phase_equivalent_stresses(
    nodes: np.ndarray,
    zone_ids: np.ndarray,
    region_ids: np.ndarray,
    geometry_params: Dict[str, float],
    landmarks_mm: Dict[str, np.ndarray],
    contour_points: np.ndarray,
) -> np.ndarray:
    r = nodes[:, 1]
    r_inner = float(landmarks_mm["r_inner"][0])
    r_outer = float(landmarks_mm["r_outer"][0])
    span = max(r_outer - r_inner, EPS)
    r_norm = np.clip((r - r_inner) / span, 0.0, 1.0)

    radial_term = 0.30 + 0.70 * np.power(r_norm, 1.30)
    hoop_term = 0.55 + 1.32 * np.power(r_norm, 2.0)
    rotor_shape = 0.44 * radial_term + 0.56 * hoop_term

    section_thickness = _geometry_section_thickness(nodes, contour_points)
    t_ref = np.median(section_thickness)
    thin_amp = np.power(np.clip(t_ref / section_thickness, 0.55, 1.95), 0.72)

    transition_conc = _transition_concentration(nodes, geometry_params, landmarks_mm)

    zone_multiplier = np.ones(nodes.shape[0], dtype=np.float64)
    zone_multiplier[zone_ids == 1] *= 1.03
    zone_multiplier[zone_ids == 3] *= 1.03

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


def compute_life_raw(phase_stress: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    sigma_eq = np.maximum(phase_stress, EPS)
    amplitude_scale = AMPLITUDE_HALF_RANGE * (AMPLITUDE_SPEED_BASE + AMPLITUDE_SPEED_GAIN * CYCLE_SPEED_FACTORS)
    sigma_a = sigma_eq * amplitude_scale[None, :]

    basquin_c = REGION_BASQUIN_C[region_ids][:, None]
    basquin_m = REGION_BASQUIN_M[region_ids][:, None]
    n_fail = basquin_c * np.power(np.maximum(sigma_a, EPS), -basquin_m)

    damage_per_cycle = np.sum(CYCLE_PHASE_WEIGHTS[None, :] / np.maximum(n_fail, 1e-20), axis=1)
    return (1.0 / np.maximum(damage_per_cycle, 1e-20)).astype(np.float64)
