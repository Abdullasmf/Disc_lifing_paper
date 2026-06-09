"""Lightweight geometry-driven stress/life surrogate.

Life intentionally uses explicit five-zone fatigue laws with a knee-based
stress-life model plus deterministic geometry-coupled severity.  The zone laws
represent engineering-zone allowables/knockdowns (not random materials).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import CYCLE_PHASE_WEIGHTS, CYCLE_SPEED_FACTORS, NOMINAL_GEOMETRY_MM, REGION_NAME_TO_ID

REGION_STRESS_SCALE = np.array([1.10, 1.00, 1.07], dtype=np.float64)
# Region scaling by index: 0=bore, 1=web, 2=rim.

# Piecewise log-log S-N parameters by zone:
# index order = bore, lower_transition, web, upper_transition, rim.
# These encode zone-specific design allowables/notch sensitivity/engineering
# severity and are intentionally distinct to produce explicit life
# discontinuities across zone thresholds.
ZONE_KNEE_STRESS_MPA = np.array([60.0, 77.0, 104.0, 107.0, 108.0], dtype=np.float64)
ZONE_KNEE_LIFE = np.array([2.0e4, 6.5e4, 1.5e5, 2.1e5, 2.0e5], dtype=np.float64)
ZONE_SLOPE_HIGH = np.array([9.1, 9.2, 7.6, 8.4, 7.8], dtype=np.float64)
ZONE_SLOPE_LOW = np.array([4.0, 4.1, 3.3, 3.7, 3.4], dtype=np.float64)

MIN_THICKNESS_MM = 1e-3
SIGMA_REF_MPA = 178.0
AMPLITUDE_HALF_RANGE = 0.5
AMPLITUDE_SPEED_BASE = 0.85
AMPLITUDE_SPEED_GAIN = 0.15
EPS = 1e-6


def _smooth_band_gate(r: np.ndarray, r_start: float, r_end: float, edge_fraction: float = 0.22) -> np.ndarray:
    """Smoothly gate values to a radial band [r_start, r_end].

    Parameters
    ----------
    r:
        Radial coordinates.
    r_start, r_end:
        Start and end radial stations of the target band.
    edge_fraction:
        Fractional edge smoothing width relative to the band width.
    """
    width = max(r_end - r_start, 1e-9)
    s = np.clip((r - r_start) / width, 0.0, 1.0)
    edge = np.clip(edge_fraction, 0.08, 0.45)
    gate_in = 1.0 / (1.0 + np.exp(-(s - edge) / max(0.25 * edge, 1e-3)))
    gate_out = 1.0 / (1.0 + np.exp(-((1.0 - edge) - s) / max(0.25 * edge, 1e-3)))
    return gate_in * gate_out


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


def _reference_section_thickness(
    params: Dict[str, float],
    radial_breaks: np.ndarray,
) -> float:
    """Representation-invariant reference thickness for stress amplification.

    Must depend only on the geometry, never on the evaluated node set: otherwise
    the same physical point receives a different stress label in the edge vs
    edge_proximity vs full representation (the median of `nodes` differs between
    sampling schemes).  Uses the axisymmetric volume-weighted (r-weighted) mean
    thickness over a fixed dense radial sampling of the disc span.
    """
    from .geometry import _thickness_profile
    r_samp = np.linspace(float(radial_breaks[0]), float(radial_breaks[5]), 512)
    t_samp = np.maximum(_thickness_profile(r_samp, params, radial_breaks), MIN_THICKNESS_MM)
    weight_sum = float(np.sum(r_samp))
    return float(np.sum(t_samp * r_samp) / max(weight_sum, EPS))


def _transition_concentration(
    nodes: np.ndarray,
    params: Dict[str, float],
    radial_breaks: np.ndarray,
) -> np.ndarray:
    """Stress concentration tied to transition radial bands and shoulder proximity."""
    r = nodes[:, 1]
    x = nodes[:, 0]
    r1 = float(radial_breaks[1])
    r2 = float(radial_breaks[2])
    r3 = float(radial_breaks[3])
    r4 = float(radial_breaks[4])

    lower_width = max(r2 - r1, 1e-6)
    upper_width = max(r4 - r3, 1e-6)

    u_lower = np.clip((r - r1) / lower_width, 0.0, 1.0)
    u_upper = np.clip((r - r3) / upper_width, 0.0, 1.0)

    shoulder_lower = np.exp(-0.5 * ((u_lower - 0.12) / 0.18) ** 2) + np.exp(-0.5 * ((u_lower - 0.88) / 0.18) ** 2)
    shoulder_upper = np.exp(-0.5 * ((u_upper - 0.12) / 0.18) ** 2) + np.exp(-0.5 * ((u_upper - 0.88) / 0.18) ** 2)

    gate_lower = _smooth_band_gate(r, r1, r2)
    gate_upper = _smooth_band_gate(r, r3, r4)

    r_lower = max(params["lower_fillet_radius"], 0.3)
    r_upper = max(params["upper_fillet_radius"], 0.3)

    gain_lower = 0.40 * np.clip((2.8 / r_lower) ** 1.05, 0.30, 2.7)
    gain_upper = 0.34 * np.clip((2.8 / r_upper) ** 1.05, 0.30, 2.7)

    t_local = _geometry_section_thickness(nodes, params, radial_breaks)
    surface_proximity = np.clip((2.0 * np.abs(x)) / np.maximum(t_local, MIN_THICKNESS_MM), 0.0, 1.0)
    shoulder_surface = 0.60 + 0.40 * np.power(surface_proximity, 1.45)

    k_lower = 1.0 + gain_lower * gate_lower * shoulder_lower * shoulder_surface
    k_upper = 1.0 + gain_upper * gate_upper * shoulder_upper * shoulder_surface
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
    r1 = float(radial_breaks[1])
    r2 = float(radial_breaks[2])
    r3 = float(radial_breaks[3])
    r_outer = float(radial_breaks[5])
    span = max(r_outer - r_inner, EPS)
    r_norm = np.clip((r - r_inner) / span, 0.0, 1.0)

    radial_term = 0.30 + 0.70 * np.power(r_norm, 1.30)
    hoop_term = 0.55 + 1.32 * np.power(r_norm, 2.0)
    rotor_shape = 0.44 * radial_term + 0.56 * hoop_term

    section_thickness = _geometry_section_thickness(nodes, geometry_params, radial_breaks)
    # Geometry-only reference (independent of the evaluated node set) so the same
    # physical point yields an identical stress label across edge/edge_proximity/full.
    t_ref = _reference_section_thickness(geometry_params, radial_breaks)
    thin_amp = np.power(np.clip(t_ref / section_thickness, 0.55, 1.95), 0.72)
    surface_proximity = np.clip((2.0 * np.abs(nodes[:, 0])) / np.maximum(section_thickness, MIN_THICKNESS_MM), 0.0, 1.0)

    transition_conc = _transition_concentration(nodes, geometry_params, radial_breaks)

    bore_width = max(r1 - r_inner, 1e-9)
    u_bore = np.clip((r - r_inner) / bore_width, 0.0, 1.0)
    bore_gate = _smooth_band_gate(r, r_inner, r1, edge_fraction=0.28)
    bore_shoulder = np.exp(-0.5 * ((u_bore - 0.88) / 0.18) ** 2)
    bore_slender = np.clip(
        (geometry_params["bore_height"] / max(geometry_params["bore_thickness"], 1e-6))
        / (NOMINAL_GEOMETRY_MM["bore_height"] / NOMINAL_GEOMETRY_MM["bore_thickness"]),
        0.65,
        2.0,
    )
    bore_radius_factor = np.clip(NOMINAL_GEOMETRY_MM["bore_radius_inner"] / max(geometry_params["bore_radius_inner"], 1e-6), 0.75, 1.35)
    bore_conc = 1.0 + 2.40 * bore_gate * bore_shoulder * np.power(surface_proximity, 1.2) * (0.35 + 0.65 * bore_radius_factor) * max(0.0, bore_slender - 0.70)

    web_width = max(r3 - r2, 1e-9)
    u_web = np.clip((r - r2) / web_width, 0.0, 1.0)
    web_mid = np.exp(-0.5 * ((u_web - 0.50) / 0.24) ** 2)
    web_gate = _smooth_band_gate(r, r2, r3, edge_fraction=0.20)
    web_slender = np.clip(
        (geometry_params["web_height"] / max(geometry_params["web_thickness"], 1e-6))
        / (NOMINAL_GEOMETRY_MM["web_height"] / NOMINAL_GEOMETRY_MM["web_thickness"]),
        0.65,
        2.2,
    )
    web_conc = 1.0 + 0.18 * web_gate * web_mid * np.power(surface_proximity, 1.1) * max(0.0, web_slender - 0.90)

    zone_multiplier = np.ones(nodes.shape[0], dtype=np.float64)
    zone_multiplier[zone_ids == 1] *= 1.04  # lower_transition
    zone_multiplier[zone_ids == 3] *= 1.04  # upper_transition

    geom_gain = 1.0 + 0.0012 * (
        geometry_params["rim_thickness"] + geometry_params["web_height"] + geometry_params["bore_radius_inner"]
    )

    base = (
        SIGMA_REF_MPA
        * rotor_shape
        * thin_amp
        * transition_conc
        * bore_conc
        * web_conc
        * zone_multiplier
        * REGION_STRESS_SCALE[region_ids]
        * geom_gain
    )

    phase_scale = CYCLE_SPEED_FACTORS**2
    phase_stress = base[:, None] * phase_scale[None, :] + 12.0
    return phase_stress.astype(np.float64)


def compute_stress_max(phase_stress: np.ndarray) -> np.ndarray:
    return np.max(phase_stress, axis=1).astype(np.float64)


def _piecewise_sn_life_per_phase(sigma_a: np.ndarray, zone_ids: np.ndarray) -> np.ndarray:
    """Return per-phase cycles-to-failure from zone-specific knee-based S-N laws."""
    sigma_knee = ZONE_KNEE_STRESS_MPA[zone_ids][:, None]
    n_knee = ZONE_KNEE_LIFE[zone_ids][:, None]
    m_high = ZONE_SLOPE_HIGH[zone_ids][:, None]
    m_low = ZONE_SLOPE_LOW[zone_ids][:, None]

    ratio = np.maximum(sigma_a, EPS) / np.maximum(sigma_knee, EPS)
    high_branch = n_knee * np.power(np.maximum(ratio, EPS), -m_high)
    low_branch = n_knee * np.power(np.maximum(ratio, EPS), -m_low)
    return np.where(sigma_a >= sigma_knee, high_branch, low_branch)


def _geometry_life_severity(
    nodes: np.ndarray,
    zone_ids: np.ndarray,
    params: Dict[str, float],
    radial_breaks: np.ndarray,
) -> np.ndarray:
    r = nodes[:, 1]
    x = nodes[:, 0]
    t_local = _geometry_section_thickness(nodes, params, radial_breaks)
    t_nom = np.maximum(_geometry_section_thickness(nodes, NOMINAL_GEOMETRY_MM, radial_breaks), MIN_THICKNESS_MM)
    t_ratio = np.clip(t_nom / t_local, 0.72, 1.95)

    r_inner = float(radial_breaks[0])
    r_outer = float(radial_breaks[5])
    r_norm = np.clip((r - r_inner) / max(r_outer - r_inner, EPS), 0.0, 1.0)
    radial_severity = np.power(r_norm, 1.25)

    lower_gate = _smooth_band_gate(r, float(radial_breaks[1]), float(radial_breaks[2]))
    upper_gate = _smooth_band_gate(r, float(radial_breaks[3]), float(radial_breaks[4]))
    surface_proximity = np.clip((2.0 * np.abs(x)) / np.maximum(t_local, MIN_THICKNESS_MM), 0.0, 1.0)

    lower_fillet_severity = np.clip(NOMINAL_GEOMETRY_MM["lower_fillet_radius"] / max(params["lower_fillet_radius"], 1e-6), 0.55, 2.1)
    upper_fillet_severity = np.clip(NOMINAL_GEOMETRY_MM["upper_fillet_radius"] / max(params["upper_fillet_radius"], 1e-6), 0.55, 2.1)

    bore_slender = np.clip(
        (params["bore_height"] / max(params["bore_thickness"], 1e-6))
        / (NOMINAL_GEOMETRY_MM["bore_height"] / NOMINAL_GEOMETRY_MM["bore_thickness"]),
        0.65,
        1.75,
    )
    web_slender = np.clip(
        (params["web_height"] / max(params["web_thickness"], 1e-6))
        / (NOMINAL_GEOMETRY_MM["web_height"] / NOMINAL_GEOMETRY_MM["web_thickness"]),
        0.70,
        1.80,
    )
    rim_slender = np.clip(
        (params["rim_height"] / max(params["rim_thickness"], 1e-6))
        / (NOMINAL_GEOMETRY_MM["rim_height"] / NOMINAL_GEOMETRY_MM["rim_thickness"]),
        0.65,
        2.00,
    )

    severity = np.ones(nodes.shape[0], dtype=np.float64)
    severity *= np.power(t_ratio, 0.56)
    severity *= 1.0 + 0.11 * radial_severity

    mask_bore = zone_ids == 0
    mask_lower = zone_ids == 1
    mask_web = zone_ids == 2
    mask_upper = zone_ids == 3
    mask_rim = zone_ids == 4

    severity[mask_bore] *= 1.0 + 0.68 * (bore_slender - 1.0) + 0.46 * (t_ratio[mask_bore] - 1.0)

    severity[mask_lower] *= 1.0 + 0.34 * lower_gate[mask_lower] * (
        0.55 + 0.45 * surface_proximity[mask_lower]
    ) * (lower_fillet_severity - 0.85)

    web_transition_tail = np.maximum(lower_gate, upper_gate)
    severity[mask_web] *= 1.0 + 0.22 * (web_slender - 1.0) + 0.20 * (t_ratio[mask_web] - 1.0) + 0.08 * web_transition_tail[mask_web]

    severity[mask_upper] *= 1.0 + 0.30 * upper_gate[mask_upper] * (
        0.55 + 0.45 * surface_proximity[mask_upper]
    ) * (upper_fillet_severity - 0.85)

    rim_thin = np.clip(NOMINAL_GEOMETRY_MM["rim_thickness"] / max(params["rim_thickness"], 1e-6), 0.70, 1.95)
    severity[mask_rim] *= 1.0 + 0.22 * (rim_slender - 1.0) + 0.34 * (rim_thin - 1.0) + 0.10 * radial_severity[mask_rim]

    return np.clip(severity, 0.65, 2.85)


def compute_life_raw(
    phase_stress: np.ndarray,
    zone_ids: np.ndarray,
    nodes: np.ndarray,
    geometry_params: Dict[str, float],
    radial_breaks: np.ndarray,
) -> np.ndarray:
    """Compute life using deterministic zone-specific knee-based S-N laws.

    Inputs:
    - geometry-driven local stress (phase_stress)
    - five-zone fatigue law (zone_ids)
    - geometry-coupled deterministic local severity (nodes + geometry parameters)
    """
    sigma_eq = np.maximum(phase_stress, EPS)
    amplitude_scale = AMPLITUDE_HALF_RANGE * (AMPLITUDE_SPEED_BASE + AMPLITUDE_SPEED_GAIN * CYCLE_SPEED_FACTORS)
    sigma_a = sigma_eq * amplitude_scale[None, :]
    local_severity = _geometry_life_severity(
        nodes=nodes,
        zone_ids=zone_ids,
        params=geometry_params,
        radial_breaks=radial_breaks,
    )
    sigma_a_eff = sigma_a * local_severity[:, None]
    n_fail = _piecewise_sn_life_per_phase(sigma_a_eff, zone_ids)

    damage_per_cycle = np.sum(CYCLE_PHASE_WEIGHTS[None, :] / np.maximum(n_fail, 1e-20), axis=1)
    return (1.0 / np.maximum(damage_per_cycle, 1e-20)).astype(np.float64)
