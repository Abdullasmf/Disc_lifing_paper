"""Lightweight physically-motivated rotating-disc stress/life surrogate.

Model summary (intentionally approximate, generation-focused):
- Per-phase equivalent stress is built from radial/hoop-like rotating-disc terms.
- Phase scaling follows speed_factor^2 to mimic centrifugal dependence.
- Local thickness and transition-landmark proximity add geometric concentration effects.
- Fatigue life uses phase-wise Miner accumulation with region-specific Basquin S-N laws.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import CYCLE_PHASE_WEIGHTS, CYCLE_SPEED_FACTORS


# Mild region scaling: keeps region discontinuity but avoids dominating all other effects.
REGION_STRESS_SCALE = np.array([1.08, 1.00, 1.14], dtype=np.float64)
# Bore and rim are slightly amplified vs web; kept close to 1.0 so geometry and phase effects dominate.

# Basquin coefficients intentionally discontinuous across bore/web/rim.
REGION_BASQUIN_C = np.array([2.2e12, 7.5e11, 4.2e12], dtype=np.float64)
REGION_BASQUIN_M = np.array([4.2, 5.0, 5.6], dtype=np.float64)
# Synthetic calibration targets plausible relative discontinuity (not material-card fidelity).

MIN_THICKNESS_MM = 1e-3
SIGMA_REF_MPA = 180.0
AMPLITUDE_HALF_RANGE = 0.5
AMPLITUDE_SPEED_BASE = 0.85
AMPLITUDE_SPEED_GAIN = 0.15
EPSILON_GUARD = 1e-6


def _local_section_thickness(nodes: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    x = nodes[:, 0]
    x1 = params["bore_thickness"]
    x2 = x1 + params["web_length"]
    x3 = x2 + params["rim_length"]

    t_bore = params["bore_thickness"]
    t_web = params["web_thickness"]
    t_rim = params["rim_thickness"]

    thickness = np.full_like(x, t_web, dtype=np.float64)
    thickness[x <= x1] = t_bore
    thickness[x >= x2] = t_rim

    in_web = (x > x1) & (x < x2)
    if np.any(in_web):
        xi = (x[in_web] - x1) / max(x2 - x1, EPSILON_GUARD)
        thickness[in_web] = (1.0 - xi) * t_web + xi * (0.85 * t_web + 0.15 * t_rim)

    in_rim_transition = (x >= x2) & (x < x3)
    if np.any(in_rim_transition):
        xi = (x[in_rim_transition] - x2) / max(x3 - x2, EPSILON_GUARD)
        thickness[in_rim_transition] = (1.0 - xi) * (0.85 * t_web + 0.15 * t_rim) + xi * t_rim

    # Minimum section floor avoids singular amplification in very thin synthetic geometries.
    return np.maximum(thickness, MIN_THICKNESS_MM)


def _concentration_factor(nodes: np.ndarray, params: Dict[str, float], landmarks_mm: Dict[str, np.ndarray]) -> np.ndarray:
    bore_landmarks = np.vstack([landmarks_mm["bore_web_lower"], landmarks_mm["bore_web_upper"]])
    rim_landmarks = np.vstack([landmarks_mm["web_rim_lower"], landmarks_mm["web_rim_upper"]])

    d_bore = np.min(np.linalg.norm(nodes[:, None, :] - bore_landmarks[None, :, :], axis=2), axis=1)
    d_rim = np.min(np.linalg.norm(nodes[:, None, :] - rim_landmarks[None, :, :], axis=2), axis=1)

    lb = max(1.5 * params["bore_web_fillet_radius"], 0.8)
    lr = max(1.5 * params["web_rim_fillet_radius"], 0.8)

    # Bore-web transition is biased slightly stronger than web-rim in this synthetic v2 model.
    k_bore = 1.0 + 0.24 * np.exp(-(d_bore / lb) ** 2)
    k_rim = 1.0 + 0.18 * np.exp(-(d_rim / lr) ** 2)
    return k_bore * k_rim


def compute_phase_equivalent_stresses(
    nodes: np.ndarray,
    region_ids: np.ndarray,
    geometry_params: Dict[str, float],
    landmarks_mm: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute synthetic equivalent stress for all 7 phases at each node (MPa-like scale)."""
    r = nodes[:, 1]
    r_inner = float(landmarks_mm["r_inner"][0])
    r_outer = float(landmarks_mm["r_outer"][0])
    span = max(r_outer - r_inner, EPSILON_GUARD)
    r_norm = np.clip((r - r_inner) / span, 0.0, 1.0)

    radial_term = 0.30 + 0.70 * np.power(r_norm, 1.35)
    hoop_term = 0.55 + 1.30 * np.power(r_norm, 2.0)
    combined_rotor_shape = 0.45 * radial_term + 0.55 * hoop_term

    section_thickness = _local_section_thickness(nodes, geometry_params)
    t_ref = np.median(section_thickness)
    thin_section_amp = np.power(np.clip(t_ref / section_thickness, 0.55, 1.9), 0.70)

    geom_conc = _concentration_factor(nodes, geometry_params, landmarks_mm)
    geom_gain = 1.0 + 0.0015 * (
        geometry_params["rim_thickness"] + geometry_params["web_length"] + geometry_params["bore_radius"]
    )

    base = (
        SIGMA_REF_MPA
        * combined_rotor_shape
        * thin_section_amp
        * geom_conc
        * REGION_STRESS_SCALE[region_ids]
        * geom_gain
    )

    phase_scale = CYCLE_SPEED_FACTORS**2
    phase_stress = base[:, None] * phase_scale[None, :] + 12.0
    return phase_stress.astype(np.float64)


def compute_stress_max(phase_stress: np.ndarray) -> np.ndarray:
    return np.max(phase_stress, axis=1)


def compute_life_raw(phase_stress: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    """Compute raw life via phase-wise stress amplitudes and Miner accumulation.

    amplitude surrogate:
      sigma_a = AMPLITUDE_HALF_RANGE * sigma_eq * (AMPLITUDE_SPEED_BASE + AMPLITUDE_SPEED_GAIN * speed_factor)
    region S-N law: N = C_region * sigma_a^(-m_region)
    cycle damage: D = sum_i(w_i / N_i), life_raw = 1 / D
    """
    sigma_eq = np.maximum(phase_stress, EPSILON_GUARD)
    amplitude_scale = AMPLITUDE_HALF_RANGE * (AMPLITUDE_SPEED_BASE + AMPLITUDE_SPEED_GAIN * CYCLE_SPEED_FACTORS)
    sigma_a = sigma_eq * amplitude_scale[None, :]  # shape: (nodes, phases)

    basquin_c = REGION_BASQUIN_C[region_ids][:, None]
    basquin_m = REGION_BASQUIN_M[region_ids][:, None]
    n_fail = basquin_c * np.power(np.maximum(sigma_a, EPSILON_GUARD), -basquin_m)

    damage_per_cycle = np.sum(CYCLE_PHASE_WEIGHTS[None, :] / np.maximum(n_fail, 1e-20), axis=1)
    return (1.0 / np.maximum(damage_per_cycle, 1e-20)).astype(np.float64)
