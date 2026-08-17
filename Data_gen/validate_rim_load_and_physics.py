from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import Data_gen.mesh_ops as _mesh_ops
from Data_gen.config import (
    CGROOVE_SAMPLING_CONTROLS,
    COUPLED_CGROOVE_PARAMETERS,
    CYCLE_PHASES,
    CYCLE_SPEED_FACTORS,
    MAX_CGROOVE_CONTROL,
    MAX_OFFSET_MM,
    MAX_RIM_FEATURE_OFFSET_MM,
    MIN_CGROOVE_CONTROL,
    MIN_OFFSET_MM,
    MIN_RIM_FEATURE_OFFSET_MM,
    NOMINAL_GEOMETRY_MM,
    NOMINAL_RIM_FEATURE_MM,
    PUBLIC_GEOMETRY_PARAMETERS,
    RIM_FEATURE_PARAMETERS,
    SUBZONE_ID_TO_NAME,
    THICKNESS_ORDERING_GAP_MM,
    ZONE_ID_TO_NAME,
    clip_cgroove_controls_to_bounds,
    clip_offsets_to_bounds,
    clip_rim_feature_offsets_to_bounds,
    map_cgroove_controls_to_parameters,
    resolve_geometry_parameters,
    resolve_rim_feature_parameters,
)
from Data_gen.dataset_generator import sample_offsets_lhs, sample_rim_feature_lhs_design
from Data_gen.features import contour_derivative_features, resample_contour_uniform_arc_length
from Data_gen.geometry import (
    build_disc_contour,
    sanitize_geometry_parameters,
    sanitize_rim_feature_parameters,
)
from Data_gen.mesh_ops import assign_zone_and_region_from_radius
from Data_gen.physics import (
    OMEGA_REF_RAD_S,
    blade_equiv_force_n,
    compute_blade_rim_traction_pa,
    compute_life_raw,
    compute_stress_max,
    recover_blade_rim_resultant_n,
    solve_axisymmetric_response,
)

DEFAULT_OUTPUT_DIR = Path("Data_gen/output/rim_load_validation")
MESH_CONFIGS = {
    "medium": {"lc_edge": 0.50, "lc_fillet": 0.30},
    "fine": {"lc_edge": 0.30, "lc_fillet": 0.18},
}
LANDMARK_NEIGHBOURHOODS_MM = {
    "rim_core_reference": 4.0,
    "lower_transition": 4.0,
    "upper_transition": 4.0,
    "front_cgroove_entry": 2.0,
    "front_cgroove_floor": 2.0,
    "front_cgroove_exit": 2.0,
    "rear_arm_root": 1.5,
    "rear_arm_neck": 1.5,
    "rear_arm_outer_corner": 1.5,
}
PHYSICAL_THRESHOLDS = {
    "preferred_peak_low_mpa": 300.0,
    "preferred_peak_high_mpa": 1300.0,
    "warning_peak_mpa": 1300.0,
    "invalid_peak_mpa": 1500.0,
    "invalid_life_cycles": 1.0,
}
DECOMPOSITION_LOADS = {
    "body_only": {"include_body_force": True, "include_blade_rim_load": False},
    "rim_load_only": {"include_body_force": False, "include_blade_rim_load": True},
    "combined": {"include_body_force": True, "include_blade_rim_load": True},
}
INTENDED_RIM_LOAD_SUBZONES = {"rear_arm_neck", "rear_arm_land", "rear_arm_corner"}
REF_PHASE_NAME = "takeoff"
REF_PHASE_INDEX = list(CYCLE_PHASES).index(REF_PHASE_NAME)


class ValidationFailure(RuntimeError):
    pass


def _patch_mesh(mesh_name: str) -> None:
    cfg = MESH_CONFIGS[mesh_name]
    _mesh_ops.LC_EDGE = float(cfg["lc_edge"])
    _mesh_ops.LC_FILLET = float(cfg["lc_fillet"])
    if hasattr(_mesh_ops, "LC_BULK"):
        _mesh_ops.LC_BULK = max(1.5, float(cfg["lc_edge"]) * 4.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _flatten_dict(prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_dict(f"{name}__", value))
        else:
            flat[name] = value
    return flat


def _build_geometry(
    core_offsets: Dict[str, float],
    rim_offsets: Dict[str, float],
    cgroove_controls: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    requested_core_offsets = clip_offsets_to_bounds(core_offsets)
    requested_rim_offsets = clip_rim_feature_offsets_to_bounds(rim_offsets)
    resolved_core = resolve_geometry_parameters(requested_core_offsets)
    actual_core = sanitize_geometry_parameters(resolved_core)
    resolved_rim = resolve_rim_feature_parameters(requested_rim_offsets)
    cgroove_controls_clipped = None
    cgroove_mapping_metadata = None
    if cgroove_controls is not None:
        cgroove_controls_clipped = clip_cgroove_controls_to_bounds(cgroove_controls)
        mapped_cgroove, cgroove_mapping_metadata = map_cgroove_controls_to_parameters(
            controls=cgroove_controls_clipped,
            resolved_rim=resolved_rim,
            t_rim=actual_core["rim_thickness"],
        )
        resolved_rim = {**resolved_rim, **mapped_cgroove}
        requested_rim_offsets = {
            k: float(resolved_rim[k] - NOMINAL_RIM_FEATURE_MM[k]) for k in RIM_FEATURE_PARAMETERS
        }
    actual_rim = sanitize_rim_feature_parameters(
        resolved_rim,
        t_rim=actual_core["rim_thickness"],
        bore_thickness=actual_core["bore_thickness"],
    )
    contour = build_disc_contour(actual_core, points_per_side=220, rim_feature_params=actual_rim)
    return {
        "requested_core_offsets": requested_core_offsets,
        "requested_rim_offsets": requested_rim_offsets,
        "resolved_core": resolved_core,
        "resolved_rim": resolved_rim,
        "actual_core": actual_core,
        "actual_rim": actual_rim,
        "cgroove_controls_requested": cgroove_controls_clipped,
        "cgroove_mapping_metadata": cgroove_mapping_metadata,
        "contour": contour,
        "radial_breaks": contour.metadata["radial_breaks_mm"],
    }


def _generate_mesh(contour, radial_breaks: np.ndarray, actual_core: Dict[str, float], actual_rim: Dict[str, float], seed: int):
    return _mesh_ops.generate_mesh(
        contour_points=contour.points,
        grid_x=90,
        grid_r=130,
        seed=seed,
        radial_breaks=radial_breaks,
        geometry_params=actual_core,
        rim_feature_params=actual_rim,
    )


def _subzone_name(subzone_id: int) -> str:
    return SUBZONE_ID_TO_NAME.get(int(subzone_id), f"unknown_{subzone_id}")


def _node_subzone_names(contour, nearest_contour_index: np.ndarray) -> np.ndarray:
    contour_subzones = contour.subzone_ids[nearest_contour_index]
    return np.array([_subzone_name(v) for v in contour_subzones], dtype=object)


def _landmark_centres(contour, radial_breaks: np.ndarray) -> Dict[str, np.ndarray]:
    centres: Dict[str, np.ndarray] = {}
    for name in LANDMARK_NEIGHBOURHOODS_MM:
        if name in contour.landmarks_mm:
            value = contour.landmarks_mm[name]
            if np.asarray(value).shape == (2,):
                centres[name] = np.asarray(value, dtype=np.float64)
                continue
        if name == "lower_transition":
            centres[name] = np.array([0.0, float(radial_breaks[1])], dtype=np.float64)
        elif name == "upper_transition":
            centres[name] = np.array([0.0, float(radial_breaks[3])], dtype=np.float64)
    return centres


def _local_indices(tree: cKDTree, centre: np.ndarray, radius_mm: float, n_nodes: int) -> np.ndarray:
    idx = tree.query_ball_point(centre, radius_mm)
    if len(idx) == 0:
        _, nearest = tree.query(centre, k=min(10, n_nodes))
        idx = list(np.atleast_1d(nearest))
    return np.array(idx, dtype=int)


def _phase_scaling_audit(phase_stress: np.ndarray, nodes: np.ndarray, contour, radial_breaks: np.ndarray) -> Dict[str, Any]:
    tree = cKDTree(nodes)
    centres = _landmark_centres(contour, radial_breaks)
    expected = CYCLE_SPEED_FACTORS ** 2
    per_landmark: Dict[str, Any] = {}
    worst_error = 0.0
    for name, radius in LANDMARK_NEIGHBOURHOODS_MM.items():
        if name not in centres:
            continue
        idx = _local_indices(tree, centres[name], radius, len(nodes))
        ref = float(np.percentile(phase_stress[idx, REF_PHASE_INDEX], 90))
        ratios = []
        for phase_idx, (phase_name, scale) in enumerate(zip(CYCLE_PHASES, expected)):
            value = float(np.percentile(phase_stress[idx, phase_idx], 90))
            ratio = value / max(ref, 1e-12)
            err = abs(ratio - float(scale))
            worst_error = max(worst_error, err)
            ratios.append({
                "phase": phase_name,
                "expected_scale": float(scale),
                "stress_ratio": float(ratio),
                "absolute_error": float(err),
                "relative_error_pct": float(100.0 * err / max(scale, 1e-12)),
            })
        per_landmark[name] = {"radius_mm": float(radius), "node_count": int(len(idx)), "ratios": ratios}
    return {
        "reference_phase": REF_PHASE_NAME,
        "acceptance_tolerance_rel": 0.02,
        "worst_absolute_error": float(worst_error),
        "worst_relative_error_pct": float(100.0 * worst_error / max(float(np.max(expected)), 1e-12)),
        "landmarks": per_landmark,
    }


def _life_stats(life_raw: np.ndarray) -> Dict[str, float]:
    return {
        "global_min_life_cycles": float(np.min(life_raw)),
        "median_life_cycles": float(np.median(life_raw)),
        "median_log10_life": float(np.median(np.log10(np.maximum(life_raw, 1e-300)))),
        "fraction_life_lt_1": float(np.mean(life_raw < 1.0)),
        "fraction_life_lt_10": float(np.mean(life_raw < 10.0)),
        "fraction_life_lt_100": float(np.mean(life_raw < 100.0)),
        "fraction_life_lt_1000": float(np.mean(life_raw < 1000.0)),
    }


def _stress_stats(stress_max: np.ndarray) -> Dict[str, float]:
    return {
        "global_peak_stress_mpa": float(np.max(stress_max)),
        "stress_p95_mpa": float(np.percentile(stress_max, 95)),
        "stress_p99_mpa": float(np.percentile(stress_max, 99)),
        "fraction_stress_gt_1300": float(np.mean(stress_max > 1300.0)),
        "fraction_stress_gt_1500": float(np.mean(stress_max > 1500.0)),
    }


def _nearest_landmark(coord_mm: np.ndarray, contour) -> Tuple[str, float]:
    best_name = "unknown"
    best_dist = float("inf")
    for name, value in contour.landmarks_mm.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (2,):
            continue
        dist = float(np.linalg.norm(coord_mm - arr))
        if dist < best_dist:
            best_name = name
            best_dist = dist
    return best_name, best_dist


def _plot_selected_facets(case_name: str, contour, mesh, load_diag: Dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    bf = mesh.mesh.boundary_facets()
    neutral = mesh.mesh.facets[:, bf]
    for i in range(neutral.shape[1]):
        pts = mesh.mesh.p[:, neutral[:, i]].T * 1e3
        ax.plot(pts[:, 0], pts[:, 1], color="0.75", lw=0.6)
    selected_ids = np.asarray(load_diag["selected_facet_ids"], dtype=int)
    for facet_id in selected_ids:
        pts = mesh.mesh.p[:, mesh.mesh.facets[:, facet_id]].T * 1e3
        ax.plot(pts[:, 0], pts[:, 1], color="magenta", lw=2.0)
    r_mm = load_diag.get("rim_top_r_m")
    x_min_mm = load_diag.get("rim_top_x_min_m")
    x_max_mm = load_diag.get("rim_top_x_max_m")
    if r_mm is not None and x_min_mm is not None and x_max_mm is not None:
        ax.plot([x_min_mm * 1e3, x_max_mm * 1e3], [r_mm * 1e3, r_mm * 1e3], color="cyan", lw=2.2, ls="--")
        ax.scatter([x_min_mm * 1e3, x_max_mm * 1e3], [r_mm * 1e3, r_mm * 1e3], color="cyan", s=24)
    for name in [
        "front_cgroove_entry",
        "front_cgroove_floor",
        "front_cgroove_exit",
        "ligament_reference",
        "rear_arm_root",
        "rear_arm_neck",
        "rear_arm_outer_corner",
        "rear_arm_load_face_centroid",
    ]:
        if name in contour.landmarks_mm and np.asarray(contour.landmarks_mm[name]).shape == (2,):
            p = contour.landmarks_mm[name]
            ax.scatter(p[0], p[1], s=20, label=name)
    traction_mpa = float(load_diag["traction_pa"]) * 1e-6
    title = (
        f"{case_name}\nselected={len(selected_ids)}  "
        f"l_face={float(load_diag['face_length_m']) * 1e3:.3f} mm  "
        f"traction={traction_mpa:.3f} MPa"
    )
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_field(nodes: np.ndarray, triangles: np.ndarray, values: np.ndarray, title: str, cbar_label: str, out_path: Path, peak_xy: np.ndarray | None = None) -> None:
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(6, 8))
    tcf = ax.tripcolor(triang, values, cmap="inferno" if "stress" in cbar_label.lower() else "viridis", shading="gouraud")
    if peak_xy is not None:
        ax.plot(peak_xy[0], peak_xy[1], "c*", ms=12)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(tcf, ax=ax, fraction=0.046, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _selected_subzone_names(contour, selected_midpoints_mm: np.ndarray) -> List[str]:
    if selected_midpoints_mm.size == 0:
        return []
    tree = cKDTree(contour.points)
    _, idx = tree.query(selected_midpoints_mm, k=1)
    return [_subzone_name(contour.subzone_ids[int(i)]) for i in np.atleast_1d(idx)]


def _resultant_force_audit(load_diag: Dict[str, Any]) -> Dict[str, Any]:
    ref_force = float(load_diag["target_force_n"])
    face_length_m = float(load_diag["face_length_m"])
    mean_radius_m = float(load_diag["mean_radius_m"])
    facet_r = np.asarray(load_diag["selected_facet_mid_r_m"], dtype=np.float64)
    facet_l = np.asarray(load_diag["selected_facet_lengths_m"], dtype=np.float64)
    phases = []
    ref_recovered = None
    for phase_name, speed_factor in zip(CYCLE_PHASES, CYCLE_SPEED_FACTORS):
        omega = float(OMEGA_REF_RAD_S * speed_factor)
        target_force = float(blade_equiv_force_n(omega))
        traction = float(compute_blade_rim_traction_pa(target_force, mean_radius_m, face_length_m))
        recovered = float(recover_blade_rim_resultant_n(traction, facet_r, facet_l))
        if phase_name == REF_PHASE_NAME:
            ref_recovered = recovered
        phases.append({
            "phase": phase_name,
            "omega_rad_s": omega,
            "reference_omega_rad_s": float(OMEGA_REF_RAD_S),
            "expected_force_scale": float(speed_factor ** 2),
            "target_resultant_force_n": target_force,
            "traction_pa": traction,
            "traction_mpa": traction * 1e-6,
            "recovered_resultant_force_n": recovered,
            "closure_error_rel": float(abs(recovered - target_force) / max(abs(target_force), 1e-12)) if abs(target_force) > 0.0 else 0.0,
            "effective_area_m2": float(np.sum(2.0 * np.pi * facet_r * facet_l)),
        })
    for item in phases:
        item["recovered_force_scale"] = float(item["recovered_resultant_force_n"] / max(ref_recovered or 1e-12, 1e-12))
    return {"reference_phase": REF_PHASE_NAME, "phases": phases}


def _evaluate_case_status(load_audit: Dict[str, Any], stress_stats: Dict[str, float], life_stats: Dict[str, float], face_outside_interval: bool, face_bad_subzone: bool, geometry_ok: bool, mesh_ok: bool, fem_ok: bool, metadata_present: bool, reason_codes: List[str]) -> Tuple[str, List[str]]:
    if not geometry_ok:
        reason_codes.append("geometry_generation_failed")
    if not mesh_ok:
        reason_codes.append("mesh_generation_failed")
    if not fem_ok:
        reason_codes.append("fem_failed")
    if not metadata_present:
        reason_codes.append("missing_rim_top_metadata")
    if int(len(load_audit["selected_facet_ids"])) == 0:
        reason_codes.append("empty_blade_load_face")
    if float(load_audit["closure_error_rel"]) > 0.01:
        reason_codes.append("force_closure_gt_1pct")
    if face_outside_interval:
        reason_codes.append("selected_facet_outside_rim_top_interval")
    if face_bad_subzone:
        reason_codes.append("selected_facet_on_unintended_boundary_region")
    if stress_stats["global_peak_stress_mpa"] > PHYSICAL_THRESHOLDS["invalid_peak_mpa"]:
        reason_codes.append("peak_stress_gt_1500_mpa")
    if life_stats["global_min_life_cycles"] < PHYSICAL_THRESHOLDS["invalid_life_cycles"]:
        reason_codes.append("life_lt_1_cycle")
    if reason_codes:
        return "FAIL", reason_codes
    warning_reasons = []
    if PHYSICAL_THRESHOLDS["warning_peak_mpa"] <= stress_stats["global_peak_stress_mpa"] <= PHYSICAL_THRESHOLDS["invalid_peak_mpa"]:
        warning_reasons.append("peak_stress_1300_to_1500_mpa")
    if 1.0 <= life_stats["global_min_life_cycles"] < 10.0:
        warning_reasons.append("life_1_to_10_cycles")
    if warning_reasons:
        return "WARNING", warning_reasons
    return "PASS", []


def _core_bounds(resolved: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    lower = {k: 1e-3 for k in PUBLIC_GEOMETRY_PARAMETERS}
    upper = {k: float("inf") for k in PUBLIC_GEOMETRY_PARAMETERS}
    rim_low = max(1e-3, float(resolved["web_thickness"]) + THICKNESS_ORDERING_GAP_MM)
    rim_actual = max(float(resolved["rim_thickness"]), rim_low)
    bore_low = max(1e-3, rim_actual + THICKNESS_ORDERING_GAP_MM)
    bore_actual = max(float(resolved["bore_thickness"]), bore_low)
    lower["rim_thickness"] = rim_low
    lower["bore_thickness"] = bore_low
    lower_dt = abs(bore_actual - float(resolved["web_thickness"]))
    upper_dt = abs(rim_actual - float(resolved["web_thickness"]))
    upper["lower_fillet_radius"] = 0.5 * min(float(resolved["lower_transition_height"]), max(lower_dt, 1e-6))
    upper["upper_fillet_radius"] = 0.5 * min(float(resolved["upper_transition_height"]), max(upper_dt, 1e-6))
    return lower, upper


def _rim_bounds(resolved: Dict[str, float], actual_core: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    t_rim = float(actual_core["rim_thickness"])
    lower = {k: 1e-3 for k in RIM_FEATURE_PARAMETERS}
    upper = {k: float("inf") for k in RIM_FEATURE_PARAMETERS}

    h_arm_hi = 0.55 * t_rim
    h_arm = min(max(float(resolved["rear_arm_radial_height"]), 2.0), h_arm_hi)
    lower["rear_arm_radial_height"] = 2.0
    upper["rear_arm_radial_height"] = h_arm_hi

    neck_hi = h_arm - 1.0
    neck_t = min(max(float(resolved["rear_arm_neck_thickness"]), 0.8), neck_hi)
    lower["rear_arm_neck_thickness"] = 0.8
    upper["rear_arm_neck_thickness"] = neck_hi

    rf_root_hi = max(min(0.45 * neck_t, 0.45 * (h_arm - neck_t)), 0.2)
    rf_root = min(max(float(resolved["rear_arm_root_radius"]), 0.2), rf_root_hi)
    lower["rear_arm_root_radius"] = 0.2
    upper["rear_arm_root_radius"] = rf_root_hi

    rf_corner_hi = max(0.45 * (h_arm - neck_t), 0.2)
    rf_corner = min(max(float(resolved["rear_arm_outer_corner_radius"]), 0.2), rf_corner_hi)
    lower["rear_arm_outer_corner_radius"] = 0.2
    upper["rear_arm_outer_corner_radius"] = rf_corner_hi

    proj_lo = 3.0 * rf_root + 0.4 + 0.3 + rf_corner
    lower["rear_arm_axial_projection"] = proj_lo
    upper["rear_arm_axial_projection"] = 20.0

    cg_pos = max(float(resolved["front_cgroove_radial_pos"]), 0.3)
    lower["front_cgroove_radial_pos"] = 0.3

    lower["front_cgroove_radial_span"] = 1.5
    lower["front_cgroove_axial_depth"] = 1.5
    upper["front_cgroove_axial_depth"] = t_rim - 2.0

    lower["front_cgroove_entry_radius"] = 0.15
    upper["front_cgroove_entry_radius"] = max(0.8 * cg_pos, 0.15)

    span_guess = max(float(resolved["front_cgroove_radial_span"]), 1.5)
    lower["front_cgroove_floor_radius"] = 0.15
    upper["front_cgroove_floor_radius"] = max(0.225 * span_guess, 0.15)

    exit_hi = max(min(h_arm - cg_pos - span_guess, span_guess) * 0.45, 0.15)
    lower["front_cgroove_exit_radius"] = 0.15
    upper["front_cgroove_exit_radius"] = exit_hi
    return lower, upper


def _coverage_status(
    range_ratio: float,
    clipped_fraction: float,
    std_value: float,
    unique_count: int,
    sample_count: int,
    lower_bound_hit_fraction: float = 0.0,
    upper_bound_hit_fraction: float = 0.0,
) -> str:
    meaningful_spread = std_value > 1e-9
    min_unique = max(4, min(20, int(math.ceil(0.40 * max(sample_count, 1)))))
    excessive_bound_hugging = max(lower_bound_hit_fraction, upper_bound_hit_fraction) >= 0.60
    collapsed = (range_ratio < 0.05) or (unique_count < min_unique) or (not meaningful_spread)
    if collapsed or excessive_bound_hugging:
        return "FAIL"
    if (range_ratio < 0.40) or (clipped_fraction > 0.30):
        return "FAIL"
    if (range_ratio < 0.70) or (clipped_fraction >= 0.10) or (max(lower_bound_hit_fraction, upper_bound_hit_fraction) >= 0.30):
        return "WARNING"
    return "PASS"


def _sampling_control_coverage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    controls_summary: Dict[str, Dict[str, Any]] = {}
    for control in CGROOVE_SAMPLING_CONTROLS:
        values = np.array(
            [
                (r.get("cgroove_controls_requested") or {}).get(control, np.nan)
                for r in results
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            controls_summary[control] = {"status": "FAIL", "reason": "missing_control_values"}
            continue
        lo = float(MIN_CGROOVE_CONTROL[control])
        hi = float(MAX_CGROOVE_CONTROL[control])
        configured_range = max(hi - lo, 1e-12)
        sampled_range = float(np.max(values) - np.min(values))
        range_ratio = float(sampled_range / configured_range)
        status = _coverage_status(
            range_ratio,
            clipped_fraction=0.0,
            std_value=float(np.std(values)),
            unique_count=int(np.unique(np.round(values, 12)).size),
            sample_count=int(values.size),
        )
        controls_summary[control] = {
            "configured_min": lo,
            "configured_max": hi,
            "sampled_min": float(np.min(values)),
            "sampled_max": float(np.max(values)),
            "sampled_mean": float(np.mean(values)),
            "sampled_std": float(np.std(values)),
            "sampled_unique_value_count": int(np.unique(np.round(values, 12)).size),
            "sampled_range_over_configured_range": range_ratio,
            "status": status,
        }
    return {
        "controls": controls_summary,
        "all_controls_sampled": all(k in controls_summary for k in CGROOVE_SAMPLING_CONTROLS),
        "all_controls_retain_meaningful_variation": all(v.get("status") != "FAIL" for v in controls_summary.values()),
    }


def _parameter_coverage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    param_summary: Dict[str, Dict[str, Any]] = {}
    reaches_geometry = True
    for group_name, params, nominal, min_off, max_off in [
        ("core", PUBLIC_GEOMETRY_PARAMETERS, NOMINAL_GEOMETRY_MM, MIN_OFFSET_MM, MAX_OFFSET_MM),
        ("rim_feature", RIM_FEATURE_PARAMETERS, NOMINAL_RIM_FEATURE_MM, MIN_RIM_FEATURE_OFFSET_MM, MAX_RIM_FEATURE_OFFSET_MM),
    ]:
        for param in params:
            requested_offsets = np.array(
                [
                    (r.get("requested_core_offsets", {}) if group_name == "core" else r.get("requested_rim_offsets", {})).get(param, np.nan)
                    for r in results
                ],
                dtype=np.float64,
            )
            resolved_values = np.array(
                [
                    (r.get("resolved_core", {}) if group_name == "core" else r.get("resolved_rim", {})).get(param, np.nan)
                    for r in results
                ],
                dtype=np.float64,
            )
            actual_values = np.array(
                [
                    (r.get("actual_core", {}) if group_name == "core" else r.get("actual_rim", {})).get(param, np.nan)
                    for r in results
                ],
                dtype=np.float64,
            )
            valid_mask = np.isfinite(requested_offsets) & np.isfinite(resolved_values) & np.isfinite(actual_values)
            if not np.any(valid_mask):
                key = f"{group_name}:{param}"
                param_summary[key] = {
                    "group": group_name,
                    "parameter": param,
                    "status": "FAIL",
                    "reason": "missing_values",
                }
                reaches_geometry = False
                continue
            requested_offsets = requested_offsets[valid_mask]
            resolved_values = resolved_values[valid_mask]
            actual_values = actual_values[valid_mask]
            valid_indices = [i for i, ok in enumerate(valid_mask.tolist()) if ok]
            lower_bound_hits = 0
            upper_bound_hits = 0
            changed = np.abs(actual_values - resolved_values) > 1e-9
            valid_results = [results[i] for i in valid_indices]
            for idx, r in enumerate(valid_results):
                if group_name == "core":
                    lower, upper = _core_bounds(r["resolved_core"])
                else:
                    lower, upper = _rim_bounds(r["resolved_rim"], r["actual_core"])
                lo = lower.get(param, -float("inf"))
                hi = upper.get(param, float("inf"))
                if changed[idx] and abs(actual_values[idx] - lo) <= 1e-8:
                    lower_bound_hits += 1
                if changed[idx] and math.isfinite(hi) and abs(actual_values[idx] - hi) <= 1e-8:
                    upper_bound_hits += 1
            intended_min = float(nominal[param] + min_off[param])
            intended_max = float(nominal[param] + max_off[param])
            intended_range = max(intended_max - intended_min, 1e-12)
            actual_range = float(np.max(actual_values) - np.min(actual_values))
            range_ratio = float(actual_range / intended_range)
            clipped_fraction = float(np.mean(changed))
            unique_count = int(np.unique(np.round(actual_values, 12)).size)
            lower_hit_fraction = float(lower_bound_hits / max(len(valid_results), 1))
            upper_hit_fraction = float(upper_bound_hits / max(len(valid_results), 1))
            status = _coverage_status(
                range_ratio,
                clipped_fraction,
                float(np.std(actual_values)),
                unique_count=unique_count,
                sample_count=len(valid_results),
                lower_bound_hit_fraction=lower_hit_fraction,
                upper_bound_hit_fraction=upper_hit_fraction,
            )
            if float(np.max(resolved_values) - np.min(resolved_values)) <= 1e-12:
                reaches_geometry = False
            key = f"{group_name}:{param}"
            param_summary[key] = {
                "group": group_name,
                "parameter": param,
                "configured_requested_min_offset": float(min_off[param]),
                "configured_requested_max_offset": float(max_off[param]),
                "requested_sampled_min_offset": float(np.min(requested_offsets)),
                "requested_sampled_max_offset": float(np.max(requested_offsets)),
                "requested_offset_mean": float(np.mean(requested_offsets)),
                "requested_offset_std": float(np.std(requested_offsets)),
                "requested_unique_value_count": int(np.unique(np.round(requested_offsets, 12)).size),
                "resolved_pre_sanitization_min": float(np.min(resolved_values)),
                "resolved_pre_sanitization_max": float(np.max(resolved_values)),
                "actual_final_min": float(np.min(actual_values)),
                "actual_final_max": float(np.max(actual_values)),
                "actual_final_mean": float(np.mean(actual_values)),
                "actual_final_std": float(np.std(actual_values)),
                "actual_unique_value_count": unique_count,
                "fraction_changed_by_sanitizer": clipped_fraction,
                "max_abs_requested_to_actual_change": float(np.max(np.abs(actual_values - resolved_values))),
                "actual_range_over_intended_range": range_ratio,
                "samples_at_active_lower_limit": int(lower_bound_hits),
                "samples_at_active_upper_limit": int(upper_bound_hits),
                "samples_at_active_lower_limit_pct": lower_hit_fraction,
                "samples_at_active_upper_limit_pct": upper_hit_fraction,
                "status": status,
            }
    cgroove_clip_rates = {
        param: float(np.mean([param in r.get("sanitized_rim_parameters_changed", []) for r in results]))
        for param in COUPLED_CGROOVE_PARAMETERS
    }
    discrepancy = {
        param: {
            "mean_abs_requested_to_final_mm": float(
                np.mean([abs(float(r["actual_rim"][param]) - float(r["resolved_rim"][param])) for r in results])
            ),
            "max_abs_requested_to_final_mm": float(
                np.max([abs(float(r["actual_rim"][param]) - float(r["resolved_rim"][param])) for r in results])
            ),
        }
        for param in COUPLED_CGROOVE_PARAMETERS
    }
    return {
        "parameters": param_summary,
        "all_declared_core_sampled": all(f"core:{k}" in param_summary for k in PUBLIC_GEOMETRY_PARAMETERS),
        "all_declared_rim_feature_sampled": all(f"rim_feature:{k}" in param_summary for k in RIM_FEATURE_PARAMETERS),
        "all_sampled_parameters_reach_geometry_generator": reaches_geometry,
        "all_sampled_parameters_retain_meaningful_variation": all(v["status"] != "FAIL" for v in param_summary.values()),
        "cgroove_parameter_clipping_rate": cgroove_clip_rates,
        "requested_to_final_discrepancy_mm": discrepancy,
    }


def _save_lhs_plots(out_dir: Path, lhs_results: List[Dict[str, Any]]) -> None:
    plot_dir = _ensure_dir(out_dir / "lhs_plots")
    fig_ctrl, axes_ctrl = plt.subplots(len(CGROOVE_SAMPLING_CONTROLS), 1, figsize=(7, max(2.2 * len(CGROOVE_SAMPLING_CONTROLS), 6)), sharex=False)
    if len(CGROOVE_SAMPLING_CONTROLS) == 1:
        axes_ctrl = [axes_ctrl]
    for ax, control in zip(axes_ctrl, CGROOVE_SAMPLING_CONTROLS):
        vals = np.array([(r.get("cgroove_controls_requested") or {}).get(control, np.nan) for r in lhs_results], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.axvline(MIN_CGROOVE_CONTROL[control], color="0.5", ls="--", lw=0.8)
            ax.axvline(MAX_CGROOVE_CONTROL[control], color="0.5", ls="--", lw=0.8)
            ax.scatter(vals, np.zeros_like(vals), c="tab:green", s=18)
        ax.set_yticks([])
        ax.set_title(control, fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
    fig_ctrl.tight_layout()
    fig_ctrl.savefig(plot_dir / "cgroove_control_coverage_panels.png", dpi=180)
    plt.close(fig_ctrl)

    for group_name, params in [("core", PUBLIC_GEOMETRY_PARAMETERS), ("rim_feature", RIM_FEATURE_PARAMETERS)]:
        for param in params:
            if group_name == "core":
                x = np.array([r["resolved_core"][param] for r in lhs_results], dtype=np.float64)
                y = np.array([r["actual_core"][param] for r in lhs_results], dtype=np.float64)
            else:
                x = np.array([r["resolved_rim"][param] for r in lhs_results], dtype=np.float64)
                y = np.array([r["actual_rim"][param] for r in lhs_results], dtype=np.float64)
            clipped = np.abs(y - x) > 1e-9
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x[~clipped], y[~clipped], c="tab:blue", s=20, label="unchanged")
            if np.any(clipped):
                ax.scatter(x[clipped], y[clipped], c="tab:red", s=28, marker="x", label="clipped")
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y=x")
            ax.set_xlabel("resolved pre-sanitization value [mm]")
            ax.set_ylabel("final sanitized value [mm]")
            ax.set_title(f"{group_name}: {param}")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{group_name}_{param}_requested_vs_actual.png", dpi=180)
            plt.close(fig)

    ordered = [("core", list(PUBLIC_GEOMETRY_PARAMETERS)), ("rim_feature", list(RIM_FEATURE_PARAMETERS))]
    for group_name, params in ordered:
        fig, axes = plt.subplots(len(params), 1, figsize=(7, max(2.2 * len(params), 6)), sharex=False)
        if len(params) == 1:
            axes = [axes]
        for ax, param in zip(axes, params):
            if group_name == "core":
                requested = np.array([r["resolved_core"][param] for r in lhs_results], dtype=np.float64)
                actual = np.array([r["actual_core"][param] for r in lhs_results], dtype=np.float64)
                lo = NOMINAL_GEOMETRY_MM[param] + MIN_OFFSET_MM[param]
                hi = NOMINAL_GEOMETRY_MM[param] + MAX_OFFSET_MM[param]
            else:
                requested = np.array([r["resolved_rim"][param] for r in lhs_results], dtype=np.float64)
                actual = np.array([r["actual_rim"][param] for r in lhs_results], dtype=np.float64)
                lo = NOMINAL_RIM_FEATURE_MM[param] + MIN_RIM_FEATURE_OFFSET_MM[param]
                hi = NOMINAL_RIM_FEATURE_MM[param] + MAX_RIM_FEATURE_OFFSET_MM[param]
            ax.axvline(lo, color="0.5", ls="--", lw=0.8)
            ax.axvline(hi, color="0.5", ls="--", lw=0.8)
            ax.scatter(requested, np.zeros_like(requested), c="tab:blue", s=18, label="requested/resolved")
            ax.scatter(actual, np.ones_like(actual), c="tab:orange", s=18, label="actual")
            ax.set_yticks([0.0, 1.0], ["requested", "actual"])
            ax.set_title(param, fontsize=9, loc="left")
            ax.grid(True, alpha=0.25)
        axes[0].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{group_name}_coverage_panels.png", dpi=180)
        plt.close(fig)


def _actual_param_vector(case_result: Dict[str, Any], keys: List[str]) -> np.ndarray:
    core = case_result.get("actual_core", {})
    rim = case_result.get("actual_rim", {})
    values = []
    for key in keys:
        if key in core:
            values.append(float(core[key]))
        else:
            values.append(float(rim[key]))
    return np.asarray(values, dtype=np.float64)


def _compute_allowed_actual_half_ranges(lhs_results: List[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    allowed: Dict[str, float] = {}
    for key in keys:
        vals = np.array(
            [
                (r.get("actual_core", {}).get(key, r.get("actual_rim", {}).get(key)))
                for r in lhs_results
            ],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            allowed[key] = 1.0
            continue
        span = float(np.max(vals) - np.min(vals))
        if span <= 1e-12:
            if key in NOMINAL_GEOMETRY_MM:
                span = float(MAX_OFFSET_MM[key] - MIN_OFFSET_MM[key])
            else:
                span = float(MAX_RIM_FEATURE_OFFSET_MM[key] - MIN_RIM_FEATURE_OFFSET_MM[key])
        allowed[key] = max(0.5 * span, 1e-9)
    return allowed


def _severity_from_actual(
    case_result: Dict[str, Any],
    keys: List[str],
    allowed_half_ranges: Dict[str, float],
) -> float:
    deviations = []
    for key in keys:
        actual = float(case_result.get("actual_core", {}).get(key, case_result.get("actual_rim", {}).get(key)))
        nominal = float(NOMINAL_GEOMETRY_MM.get(key, NOMINAL_RIM_FEATURE_MM.get(key)))
        denom = max(float(allowed_half_ranges[key]), 1e-9)
        deviations.append(((actual - nominal) / denom) ** 2)
    return float(np.sqrt(np.mean(deviations))) if deviations else 0.0


def _contour_log_life_profile(case_result: Dict[str, Any], n_samples: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contour = case_result["_contour"]
    mesh = case_result["_mesh"]
    life_raw = np.asarray(case_result["_life_raw"], dtype=np.float64)
    stress_max = np.asarray(case_result["_stress_max"], dtype=np.float64)
    points, _ = resample_contour_uniform_arc_length(
        points=contour.points,
        arc_length_mm=contour.arc_length_mm,
        n_samples=n_samples,
    )
    deriv = contour_derivative_features(points, np.linspace(0.0, float(contour.arc_length_mm[-1]), n_samples, endpoint=False))
    tree = cKDTree(mesh.nodes)
    _, idx = tree.query(points, k=1)
    log_life = np.log10(np.maximum(life_raw[idx], 1e-300))
    return points, log_life, stress_max[idx], deriv["curvature"], deriv["curvature_gradient"]


def _local_mesh_metrics(mesh, node_index: int, centre: np.ndarray, radius_mm: float = 1.0) -> Dict[str, float]:
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    tri = np.asarray(mesh.triangles, dtype=int)
    if tri.size == 0:
        return {"local_mesh_size_mm": np.nan, "local_node_density_per_mm2": np.nan}
    incident = np.where(np.any(tri == int(node_index), axis=1))[0]
    edge_lengths: List[float] = []
    for tid in incident:
        tv = tri[tid]
        p = nodes[tv]
        edge_lengths.extend(
            [
                float(np.linalg.norm(p[0] - p[1])),
                float(np.linalg.norm(p[1] - p[2])),
                float(np.linalg.norm(p[2] - p[0])),
            ]
        )
    tree = cKDTree(nodes)
    idx = tree.query_ball_point(np.asarray(centre, dtype=np.float64), radius_mm)
    area = math.pi * max(radius_mm, 1e-6) ** 2
    return {
        "local_mesh_size_mm": float(np.median(edge_lengths)) if edge_lengths else np.nan,
        "local_node_density_per_mm2": float(len(idx) / area),
    }


def _landmark_delta_stats(sample: Dict[str, Any], nominal: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    sample_mesh = sample["_mesh"]
    nominal_mesh = nominal["_mesh"]
    sample_tree = cKDTree(sample_mesh.nodes)
    nominal_tree = cKDTree(nominal_mesh.nodes)
    sample_centres = _landmark_centres(sample["_contour"], np.asarray(sample["radial_breaks_mm"], dtype=np.float64))
    nominal_centres = _landmark_centres(nominal["_contour"], np.asarray(nominal["radial_breaks_mm"], dtype=np.float64))
    sample_stress = np.asarray(sample["_stress_max"], dtype=np.float64)
    nominal_stress = np.asarray(nominal["_stress_max"], dtype=np.float64)
    sample_loglife_nodes = np.log10(np.maximum(np.asarray(sample["_life_raw"], dtype=np.float64), 1e-300))
    nominal_loglife_nodes = np.log10(np.maximum(np.asarray(nominal["_life_raw"], dtype=np.float64), 1e-300))
    for landmark, radius in LANDMARK_NEIGHBOURHOODS_MM.items():
        if landmark not in sample_centres or landmark not in nominal_centres:
            continue
        s_idx = _local_indices(sample_tree, sample_centres[landmark], radius, len(sample_mesh.nodes))
        n_idx = _local_indices(nominal_tree, nominal_centres[landmark], radius, len(nominal_mesh.nodes))
        s_ll = sample_loglife_nodes[s_idx]
        n_ll = nominal_loglife_nodes[n_idx]
        out[landmark] = {
            "delta_p90_stress_mpa": float(np.percentile(sample_stress[s_idx], 90) - np.percentile(nominal_stress[n_idx], 90)),
            "delta_median_loglife": float(np.median(s_ll) - np.median(n_ll)),
            "delta_p05_loglife": float(np.percentile(s_ll, 5) - np.percentile(n_ll, 5)),
            "delta_min_loglife": float(np.min(s_ll) - np.min(n_ll)),
        }
    return out


def _landmark_delta_table(sample: Dict[str, Any], nominal: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    sample_metrics = sample.get("landmark_metrics", {})
    nominal_metrics = nominal.get("landmark_metrics", {})
    for landmark in LANDMARK_NEIGHBOURHOODS_MM:
        s = sample_metrics.get(landmark, {})
        n = nominal_metrics.get(landmark, {})
        s_p90 = float(s.get("p90_stress_mpa", np.nan))
        n_p90 = float(n.get("p90_stress_mpa", np.nan))
        s_mlife = float(s.get("median_life_cycles", np.nan))
        n_mlife = float(n.get("median_life_cycles", np.nan))
        s_log = float(np.log10(max(s_mlife, 1e-300))) if np.isfinite(s_mlife) else np.nan
        n_log = float(np.log10(max(n_mlife, 1e-300))) if np.isfinite(n_mlife) else np.nan
        out[landmark] = {
            "delta_p90_stress_mpa": float(s_p90 - n_p90) if np.isfinite(s_p90) and np.isfinite(n_p90) else np.nan,
            "delta_median_loglife": float(s_log - n_log) if np.isfinite(s_log) and np.isfinite(n_log) else np.nan,
        }
    return out


def _life_sensitivity_audit(
    lhs_results_raw: List[Dict[str, Any]],
    nominal_raw: Dict[str, Any],
    out_dir: Path,
    save_plots: bool,
) -> Dict[str, Any]:
    all_keys = list(PUBLIC_GEOMETRY_PARAMETERS) + list(RIM_FEATURE_PARAMETERS)
    cgroove_keys = [
        "front_cgroove_axial_depth",
        "front_cgroove_radial_pos",
        "front_cgroove_radial_span",
        "front_cgroove_entry_radius",
        "front_cgroove_floor_radius",
        "front_cgroove_exit_radius",
    ]
    rear_arm_keys = [
        "rear_arm_axial_projection",
        "rear_arm_radial_height",
        "rear_arm_neck_thickness",
        "rear_arm_root_radius",
        "rear_arm_outer_corner_radius",
    ]
    allowed_half = _compute_allowed_actual_half_ranges(lhs_results_raw, all_keys)
    nominal_points, nominal_loglife_contour, nominal_stress_contour, _, _ = _contour_log_life_profile(nominal_raw)
    nominal_tree = cKDTree(nominal_raw["_mesh"].nodes)
    rows: List[Dict[str, Any]] = []
    severe_rows: List[Dict[str, Any]] = []
    landmark_names = list(LANDMARK_NEIGHBOURHOODS_MM.keys())
    robust_delta_all: List[np.ndarray] = []

    for sample in lhs_results_raw:
        row: Dict[str, Any] = {
            "sample_id": sample.get("sample_id"),
            "case_id": sample.get("case_id"),
            "status": sample.get("status"),
            "severity_total": _severity_from_actual(sample, all_keys, allowed_half),
            "severity_core": _severity_from_actual(sample, list(PUBLIC_GEOMETRY_PARAMETERS), allowed_half),
            "severity_cgroove": _severity_from_actual(sample, cgroove_keys, allowed_half),
            "severity_rear_arm": _severity_from_actual(sample, rear_arm_keys, allowed_half),
        }
        if sample.get("status") != "PASS":
            rows.append(row)
            continue

        sample_points, sample_loglife_contour, sample_stress_contour, sample_curvature, sample_curv_grad = _contour_log_life_profile(
            sample, n_samples=len(nominal_points)
        )
        delta_contour = sample_loglife_contour - nominal_loglife_contour
        robust_delta_all.append(delta_contour)
        min_idx = int(np.argmin(delta_contour))
        controlling_landmark, _ = _nearest_landmark(sample_points[min_idx], sample["_contour"])

        stress_peak_sample = float(sample.get("stress_stats", {}).get("global_peak_stress_mpa", np.nan))
        stress_peak_nom = float(nominal_raw.get("stress_stats", {}).get("global_peak_stress_mpa", np.nan))
        sample_life = np.asarray(sample["_life_raw"], dtype=np.float64)
        nominal_life = np.asarray(nominal_raw["_life_raw"], dtype=np.float64)
        row.update(
            {
                "global_min_loglife": float(np.min(np.log10(np.maximum(sample_life, 1e-300)))),
                "median_loglife": float(np.median(np.log10(np.maximum(sample_life, 1e-300)))),
                "delta_loglife_min": float(np.min(delta_contour)),
                "delta_loglife_median": float(np.median(delta_contour)),
                "delta_loglife_p01": float(np.percentile(delta_contour, 1)),
                "delta_loglife_p05": float(np.percentile(delta_contour, 5)),
                "delta_loglife_p10": float(np.percentile(delta_contour, 10)),
                "delta_loglife_p90": float(np.percentile(delta_contour, 90)),
                "delta_loglife_p99": float(np.percentile(delta_contour, 99)),
                "delta_loglife_p95": float(np.percentile(delta_contour, 95)),
                "delta_loglife_max_abs": float(np.max(np.abs(delta_contour))),
                "fraction_contour_delta_loglife_le_neg0p3": float(np.mean(delta_contour <= -0.3)),
                "fraction_contour_delta_loglife_le_neg0p5": float(np.mean(delta_contour <= -0.5)),
                "fraction_contour_delta_loglife_le_neg1p0": float(np.mean(delta_contour <= -1.0)),
                "fraction_contour_delta_loglife_le_neg2p0": float(np.mean(delta_contour <= -2.0)),
                "controlling_feature_or_landmark": controlling_landmark,
                "delta_global_peak_stress_mpa": float(stress_peak_sample - stress_peak_nom),
                "delta_global_min_loglife_vs_nominal": float(
                    np.min(np.log10(np.maximum(sample_life, 1e-300))) - np.min(np.log10(np.maximum(nominal_life, 1e-300)))
                ),
                "severe_coordinate_x_mm": float(sample_points[min_idx, 0]),
                "severe_coordinate_r_mm": float(sample_points[min_idx, 1]),
                "severe_local_stress_mpa": float(sample_stress_contour[min_idx]),
                "severe_local_stress_delta_mpa": float(sample_stress_contour[min_idx] - nominal_stress_contour[min_idx]),
            }
        )
        landmark_deltas = _landmark_delta_stats(sample, nominal_raw)
        for landmark, metrics in landmark_deltas.items():
            row[f"delta_p90_stress__{landmark}"] = metrics["delta_p90_stress_mpa"]
            row[f"delta_median_loglife__{landmark}"] = metrics["delta_median_loglife"]
            row[f"delta_p05_loglife__{landmark}"] = metrics["delta_p05_loglife"]
            row[f"delta_min_loglife__{landmark}"] = metrics["delta_min_loglife"]
        rows.append(row)

        if row["delta_loglife_min"] <= -2.0:
            sample_tree = cKDTree(sample["_mesh"].nodes)
            severe_coord = sample_points[min_idx]
            severe_node = int(sample_tree.query(severe_coord, k=1)[1])
            nom_node = int(nominal_tree.query(nominal_points[min_idx], k=1)[1])
            mesh_metrics = _local_mesh_metrics(sample["_mesh"], severe_node, severe_coord, radius_mm=1.0)
            severe_rows.append({
                "sample_id": sample.get("sample_id"),
                "case_id": sample.get("case_id"),
                "requested_normalized_controls": json.dumps(sample.get("cgroove_controls_requested", {}), sort_keys=True),
                "resolved_pre_sanitization_cgroove_params_mm": json.dumps(
                    {k: float(sample.get("resolved_rim", {}).get(k, np.nan)) for k in COUPLED_CGROOVE_PARAMETERS},
                    sort_keys=True,
                ),
                "final_actual_cgroove_params_mm": json.dumps(
                    {k: float(sample.get("actual_rim", {}).get(k, np.nan)) for k in COUPLED_CGROOVE_PARAMETERS},
                    sort_keys=True,
                ),
                "severe_point_x_mm": float(severe_coord[0]),
                "severe_point_r_mm": float(severe_coord[1]),
                "severe_landmark": controlling_landmark,
                "named_region": str(sample.get("controlling_subzone", "unknown")),
                "local_stress_mpa": float(sample_stress_contour[min_idx]),
                "local_stress_increase_vs_nominal_mpa": float(sample_stress_contour[min_idx] - nominal_stress_contour[min_idx]),
                "local_mesh_size_mm": mesh_metrics["local_mesh_size_mm"],
                "local_node_density_per_mm2": mesh_metrics["local_node_density_per_mm2"],
                "local_curvature_1_per_mm": float(sample_curvature[min_idx]),
                "local_curvature_gradient_1_per_mm2": float(sample_curv_grad[min_idx]),
                "min_delta_log10_life": float(row["delta_loglife_min"]),
                "local_delta_log10_life": float(delta_contour[min_idx]),
                "sample_local_stress_node_mpa": float(np.asarray(sample["_stress_max"], dtype=np.float64)[severe_node]),
                "nominal_local_stress_node_mpa": float(np.asarray(nominal_raw["_stress_max"], dtype=np.float64)[nom_node]),
                "mesh_feature_stability_flag": "requires_medium_fine_check",
            })

    _write_csv(out_dir / "lhs_geometry_to_life_sensitivity.csv", rows)
    _write_csv(out_dir / "severe_life_change_cases.csv", severe_rows)

    finite_rows = [r for r in rows if np.isfinite(r.get("delta_loglife_min", np.nan))]
    corr_rows = []
    if finite_rows:
        y = np.array([r["delta_loglife_min"] for r in finite_rows], dtype=np.float64)
        for key in all_keys:
            x = np.array(
                [float(rf.get("actual_core", {}).get(key, rf.get("actual_rim", {}).get(key))) for rf in lhs_results_raw if rf.get("status") == "PASS"],
                dtype=np.float64,
            )
            if x.size != y.size or np.std(x) <= 1e-12:
                continue
            corr, pval = spearmanr(x, y)
            corr_rows.append({"parameter": key, "spearman_rho_vs_delta_loglife_min": float(corr), "p_value": float(pval)})
        corr_rows = sorted(corr_rows, key=lambda d: abs(d["spearman_rho_vs_delta_loglife_min"]), reverse=True)

    summary = {
        "num_samples": len(rows),
        "num_valid_samples": len(finite_rows),
        "contour_correspondence_method": "uniform_arc_length_resampling_with_nearest_fem_node_query",
        "delta_loglife_min_distribution": {
            "min": float(np.min([r["delta_loglife_min"] for r in finite_rows])) if finite_rows else np.nan,
            "p05": float(np.percentile([r["delta_loglife_min"] for r in finite_rows], 5)) if finite_rows else np.nan,
            "median": float(np.median([r["delta_loglife_min"] for r in finite_rows])) if finite_rows else np.nan,
            "p95": float(np.percentile([r["delta_loglife_min"] for r in finite_rows], 95)) if finite_rows else np.nan,
        },
        "robust_delta_loglife_contour_distribution": {
            "p01": float(np.percentile(np.concatenate(robust_delta_all), 1)) if robust_delta_all else np.nan,
            "p05": float(np.percentile(np.concatenate(robust_delta_all), 5)) if robust_delta_all else np.nan,
            "p10": float(np.percentile(np.concatenate(robust_delta_all), 10)) if robust_delta_all else np.nan,
            "median": float(np.percentile(np.concatenate(robust_delta_all), 50)) if robust_delta_all else np.nan,
            "p90": float(np.percentile(np.concatenate(robust_delta_all), 90)) if robust_delta_all else np.nan,
            "p99": float(np.percentile(np.concatenate(robust_delta_all), 99)) if robust_delta_all else np.nan,
            "fraction_le_neg0p3": float(np.mean(np.concatenate(robust_delta_all) <= -0.3)) if robust_delta_all else np.nan,
            "fraction_le_neg0p5": float(np.mean(np.concatenate(robust_delta_all) <= -0.5)) if robust_delta_all else np.nan,
            "fraction_le_neg1p0": float(np.mean(np.concatenate(robust_delta_all) <= -1.0)) if robust_delta_all else np.nan,
            "fraction_le_neg2p0": float(np.mean(np.concatenate(robust_delta_all) <= -2.0)) if robust_delta_all else np.nan,
        },
        "sensitivity_correlation_ranked": corr_rows,
        "allowed_actual_half_ranges_mm": allowed_half,
        "landmarks_reported": landmark_names,
        "severe_case_count_min_delta_le_neg2": int(len(severe_rows)),
    }
    _write_json(out_dir / "lhs_geometry_to_life_sensitivity_summary.json", summary)

    if save_plots and finite_rows:
        plot_dir = _ensure_dir(out_dir / "lhs_life_sensitivity_plots")
        sev = np.array([r["severity_total"] for r in finite_rows], dtype=np.float64)
        peak = np.array([r["delta_global_peak_stress_mpa"] for r in finite_rows], dtype=np.float64)
        gmin = np.array([r["global_min_loglife"] for r in finite_rows], dtype=np.float64)
        dmin = np.array([r["delta_loglife_min"] for r in finite_rows], dtype=np.float64)

        def _scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, name: str) -> None:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y, c="tab:blue", s=24)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / name, dpi=180)
            plt.close(fig)

        _scatter(sev, peak, "normalized severity (total)", "delta global peak stress [MPa]", "severity_vs_peak_stress.png")
        _scatter(sev, gmin, "normalized severity (total)", "global min log10 life", "severity_vs_global_min_loglife.png")
        _scatter(sev, dmin, "normalized severity (total)", "min delta log10 life", "severity_vs_min_delta_loglife.png")

        for landmark in landmark_names:
            y_lm = np.array([r.get(f"delta_median_loglife__{landmark}", np.nan) for r in finite_rows], dtype=np.float64)
            mask = np.isfinite(y_lm)
            if np.any(mask):
                _scatter(sev[mask], y_lm[mask], "normalized severity (total)", f"delta median loglife @ {landmark}", f"severity_vs_delta_loglife_{landmark}.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dmin, bins=16, color="tab:purple", alpha=0.85)
        ax.set_xlabel("min delta log10 life vs nominal")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "hist_min_delta_loglife.png", dpi=180)
        plt.close(fig)

        heat = np.full((len(finite_rows), len(landmark_names)), np.nan, dtype=np.float64)
        for i, r in enumerate(finite_rows):
            for j, landmark in enumerate(landmark_names):
                heat[i, j] = float(r.get(f"delta_median_loglife__{landmark}", np.nan))
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(landmark_names)), max(4, 0.25 * len(finite_rows))))
        im = ax.imshow(heat, aspect="auto", cmap="coolwarm")
        ax.set_yticks(np.arange(len(finite_rows)))
        ax.set_yticklabels([int(r.get("sample_id", i)) for i, r in enumerate(finite_rows)], fontsize=6)
        ax.set_xticks(np.arange(len(landmark_names)))
        ax.set_xticklabels(landmark_names, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("landmark")
        ax.set_ylabel("sample id")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="delta median log10 life")
        fig.tight_layout()
        fig.savefig(plot_dir / "heatmap_delta_loglife_landmarks.png", dpi=180)
        plt.close(fig)
    return summary


def _build_case_specs(case: str, num_samples: int, seed: int) -> Tuple[List[Dict[str, Any]], bool]:
    specs: List[Dict[str, Any]] = []
    includes_lhs = False
    if case in {"nominal", "all"}:
        specs.append({
            "group": "nominal",
            "case_name": "nominal",
            "case_id": "nominal",
            "core_offsets": {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS},
            "rim_offsets": {k: 0.0 for k in RIM_FEATURE_PARAMETERS},
            "cgroove_controls": None,
            "seed": seed,
        })
    if case in {"extrema", "all"}:
        for param in PUBLIC_GEOMETRY_PARAMETERS:
            for side, table in [("min", MIN_OFFSET_MM), ("max", MAX_OFFSET_MM)]:
                core = {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS}
                rim = {k: 0.0 for k in RIM_FEATURE_PARAMETERS}
                core[param] = float(table[param])
                specs.append({
                    "group": "extrema",
                    "case_name": f"core_{param}_{side}",
                    "case_id": f"extrema_core_{param}_{side}",
                    "core_offsets": core,
                    "rim_offsets": rim,
                    "cgroove_controls": None,
                    "seed": seed,
                })
        for param in RIM_FEATURE_PARAMETERS:
            for side, table in [("min", MIN_RIM_FEATURE_OFFSET_MM), ("max", MAX_RIM_FEATURE_OFFSET_MM)]:
                core = {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS}
                rim = {k: 0.0 for k in RIM_FEATURE_PARAMETERS}
                rim[param] = float(table[param])
                specs.append({
                    "group": "extrema",
                    "case_name": f"rim_{param}_{side}",
                    "case_id": f"extrema_rim_{param}_{side}",
                    "core_offsets": core,
                    "rim_offsets": rim,
                    "cgroove_controls": None,
                    "seed": seed,
                })
        coupled = {
            "deep_groove_min_ligament": {
                "front_cgroove_axial_depth": MAX_RIM_FEATURE_OFFSET_MM["front_cgroove_axial_depth"],
                "front_cgroove_radial_span": MAX_RIM_FEATURE_OFFSET_MM["front_cgroove_radial_span"],
                "rear_arm_root_radius": MIN_RIM_FEATURE_OFFSET_MM["rear_arm_root_radius"],
            },
            "thin_arm_neck_small_root_fillet": {
                "rear_arm_neck_thickness": MIN_RIM_FEATURE_OFFSET_MM["rear_arm_neck_thickness"],
                "rear_arm_root_radius": MIN_RIM_FEATURE_OFFSET_MM["rear_arm_root_radius"],
                "rear_arm_radial_height": MIN_RIM_FEATURE_OFFSET_MM["rear_arm_radial_height"],
            },
            "max_arm_projection": {
                "rear_arm_axial_projection": MAX_RIM_FEATURE_OFFSET_MM["rear_arm_axial_projection"],
                "rear_arm_radial_height": MAX_RIM_FEATURE_OFFSET_MM["rear_arm_radial_height"],
            },
            "max_rim_feature_combination": {k: float(v) for k, v in MAX_RIM_FEATURE_OFFSET_MM.items()},
            "min_rim_feature_combination": {k: float(v) for k, v in MIN_RIM_FEATURE_OFFSET_MM.items()},
        }
        for name, rim in coupled.items():
            specs.append({
                "group": "extrema",
                "case_name": name,
                "case_id": f"extrema_{name}",
                "core_offsets": {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS},
                "rim_offsets": {k: rim.get(k, 0.0) for k in RIM_FEATURE_PARAMETERS},
                "cgroove_controls": None,
                "seed": seed,
            })
    if case in {"lhs", "all"}:
        includes_lhs = True
        core_offsets = sample_offsets_lhs(num_samples, MIN_OFFSET_MM, MAX_OFFSET_MM, seed)
        rim_design = sample_rim_feature_lhs_design(num_samples, MIN_RIM_FEATURE_OFFSET_MM, MAX_RIM_FEATURE_OFFSET_MM, seed)
        for i, (core, rim_item) in enumerate(zip(core_offsets, rim_design)):
            sample_seed = int((int(seed) * 1_000_003 + i * 7_919 + 97) % (2**31 - 1))
            specs.append({
                "group": "lhs",
                "case_name": f"lhs_{i:03d}",
                "case_id": f"lhs_{i:03d}",
                "sample_id": i,
                "core_offsets": core,
                "rim_offsets": rim_item["rim_feature_offsets"],
                "cgroove_controls": rim_item["cgroove_controls_requested"],
                "seed": sample_seed,
            })
    return specs, includes_lhs


def _decomposition_summary(case_result: Dict[str, Any], out_dir: Path, save_plots: bool) -> Dict[str, Any]:
    contour = case_result["_contour"]
    mesh = case_result["_mesh"]
    radial_breaks = np.asarray(case_result["radial_breaks_mm"], dtype=np.float64)
    actual_core = case_result["actual_core"]
    actual_rim = case_result["actual_rim"]
    nodes = mesh.nodes
    triangles = mesh.triangles
    zone_ids, region_ids = assign_zone_and_region_from_radius(nodes=nodes, radial_breaks=radial_breaks)
    rim_meta = {k: v for k, v in contour.metadata.items() if k.startswith("blade_rim_top_")}
    outputs: Dict[str, Any] = {}
    plots_dir = _ensure_dir(out_dir / "decomposition_plots")
    for load_name, flags in DECOMPOSITION_LOADS.items():
        response = solve_axisymmetric_response(
            nodes=nodes,
            zone_ids=zone_ids,
            region_ids=region_ids,
            geometry_params=actual_core,
            radial_breaks=radial_breaks,
            mesh_obj=mesh.mesh,
            triangles=triangles,
            rim_face_metadata=rim_meta,
            include_body_force=flags["include_body_force"],
            include_blade_rim_load=flags["include_blade_rim_load"],
        )
        phase_stress = response["phase_stress_mpa"]
        stress_max = compute_stress_max(phase_stress)
        life_raw = compute_life_raw(phase_stress=phase_stress, zone_ids=zone_ids, nodes=nodes, geometry_params=actual_core, radial_breaks=radial_breaks, lifing_mode="zonal")
        peak_idx = int(np.argmax(stress_max))
        peak_xy = nodes[peak_idx]
        landmark_metrics = {}
        tree = cKDTree(nodes)
        centres = _landmark_centres(contour, radial_breaks)
        for landmark, radius in LANDMARK_NEIGHBOURHOODS_MM.items():
            if landmark not in centres:
                continue
            idx = _local_indices(tree, centres[landmark], radius, len(nodes))
            landmark_metrics[landmark] = {
                "p90_stress_mpa": float(np.percentile(stress_max[idx], 90)),
                "node_count": int(len(idx)),
            }
        outputs[load_name] = {
            "include_body_force": flags["include_body_force"],
            "include_blade_rim_load": flags["include_blade_rim_load"],
            "global_peak_stress_mpa": float(np.max(stress_max)),
            "peak_coordinate_mm": [float(peak_xy[0]), float(peak_xy[1])],
            "global_min_life_cycles": float(np.min(life_raw)),
            "median_life_cycles": float(np.median(life_raw)),
            "median_log10_life": float(np.median(np.log10(np.maximum(life_raw, 1e-300)))),
            "fraction_life_lt_1": float(np.mean(life_raw < 1.0)),
            "fraction_life_lt_10": float(np.mean(life_raw < 10.0)),
            "fraction_life_lt_100": float(np.mean(life_raw < 100.0)),
            "fraction_life_lt_1000": float(np.mean(life_raw < 1000.0)),
            "landmark_p90_stress_mpa": landmark_metrics,
            "force_metadata": response["load_diagnostics"],
        }
        if save_plots:
            stress_path = plots_dir / f"{load_name}_stress.png"
            life_path = plots_dir / f"{load_name}_loglife.png"
            _plot_field(nodes, triangles, stress_max, f"{load_name} stress", "von Mises stress [MPa]", stress_path, peak_xy=peak_xy)
            _plot_field(nodes, triangles, np.log10(np.maximum(life_raw, 1e-300)), f"{load_name} log10 life", "log10(cycles)", life_path)
            outputs[load_name]["stress_plot"] = str(stress_path)
            outputs[load_name]["loglife_plot"] = str(life_path)
    classes = {}
    for name in LANDMARK_NEIGHBOURHOODS_MM:
        values = {load: outputs[load]["landmark_p90_stress_mpa"].get(name, {}).get("p90_stress_mpa", 0.0) for load in DECOMPOSITION_LOADS}
        body = values.get("body_only", 0.0)
        rim = values.get("rim_load_only", 0.0)
        if body > 1.2 * max(rim, 1e-12):
            classes[name] = "body_load_dominated"
        elif rim > 1.2 * max(body, 1e-12):
            classes[name] = "rim_load_influenced"
        else:
            classes[name] = "mixed"
    outputs["landmark_classification"] = classes
    return outputs


def validate_geometry_case(spec: Dict[str, Any], mesh_name: str, out_dir: Path, save_plots: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "group": spec["group"],
        "case_name": spec["case_name"],
        "case_id": spec["case_id"],
        "sample_id": spec.get("sample_id"),
        "mesh": mesh_name,
        "seed": int(spec["seed"]),
        "requested_core_offsets": clip_offsets_to_bounds(spec["core_offsets"]),
        "requested_rim_offsets": clip_rim_feature_offsets_to_bounds(spec["rim_offsets"]),
        "requested_cgroove_controls": clip_cgroove_controls_to_bounds(spec["cgroove_controls"]) if spec.get("cgroove_controls") is not None else None,
    }
    geometry_ok = mesh_ok = fem_ok = False
    reason_codes: List[str] = []
    try:
        built = _build_geometry(spec["core_offsets"], spec["rim_offsets"], spec.get("cgroove_controls"))
        geometry_ok = True
        contour = built["contour"]
        radial_breaks = built["radial_breaks"]
        result.update({
            "requested_core_offsets": built["requested_core_offsets"],
            "requested_rim_offsets": built["requested_rim_offsets"],
            "resolved_core": built["resolved_core"],
            "resolved_rim": built["resolved_rim"],
            "actual_core": built["actual_core"],
            "actual_rim": built["actual_rim"],
            "cgroove_controls_requested": built["cgroove_controls_requested"],
            "cgroove_mapping_metadata": built["cgroove_mapping_metadata"],
            "sanitized_core_parameters_changed": [
                k for k in PUBLIC_GEOMETRY_PARAMETERS
                if abs(float(built["actual_core"][k]) - float(built["resolved_core"][k])) > 1e-9
            ],
            "sanitized_rim_parameters_changed": [
                k for k in RIM_FEATURE_PARAMETERS
                if abs(float(built["actual_rim"][k]) - float(built["resolved_rim"][k])) > 1e-9
            ],
            "radial_breaks_mm": radial_breaks.tolist(),
            "rim_top_metadata_mm": {k: float(v[0]) for k, v in contour.metadata.items() if k.startswith("blade_rim_top_")},
        })
        mesh = _generate_mesh(contour, radial_breaks, built["actual_core"], built["actual_rim"], spec["seed"])
        mesh_ok = True
        zone_ids, region_ids = assign_zone_and_region_from_radius(nodes=mesh.nodes, radial_breaks=radial_breaks)
        rim_meta = {k: v for k, v in contour.metadata.items() if k.startswith("blade_rim_top_")}
        response = solve_axisymmetric_response(
            nodes=mesh.nodes,
            zone_ids=zone_ids,
            region_ids=region_ids,
            geometry_params=built["actual_core"],
            radial_breaks=radial_breaks,
            mesh_obj=mesh.mesh,
            triangles=mesh.triangles,
            rim_face_metadata=rim_meta,
        )
        fem_ok = bool(response["ok"])
        phase_stress = response["phase_stress_mpa"]
        stress_max = compute_stress_max(phase_stress)
        life_raw = compute_life_raw(phase_stress=phase_stress, zone_ids=zone_ids, nodes=mesh.nodes, geometry_params=built["actual_core"], radial_breaks=radial_breaks, lifing_mode="zonal")
        peak_idx = int(np.argmax(stress_max))
        peak_xy = mesh.nodes[peak_idx]
        peak_landmark, peak_landmark_dist = _nearest_landmark(peak_xy, contour)
        nearest_contour_idx = mesh.nearest_contour_index
        node_subzones = _node_subzone_names(contour, nearest_contour_idx)
        controlling_subzone = str(node_subzones[peak_idx])

        load_diag = response["load_diagnostics"]
        selected_ids = np.asarray(load_diag["selected_facet_ids"], dtype=int)
        selected_midpoints_mm = np.column_stack([
            np.asarray(load_diag["selected_facet_mid_x_m"], dtype=np.float64) * 1e3,
            np.asarray(load_diag["selected_facet_mid_r_m"], dtype=np.float64) * 1e3,
        ]) if selected_ids.size else np.empty((0, 2), dtype=np.float64)
        selected_subzones = _selected_subzone_names(contour, selected_midpoints_mm)
        facet_node_ids = np.unique(mesh.mesh.facets[:, selected_ids]).astype(int) if selected_ids.size else np.empty(0, dtype=int)
        selected_x_mm = np.asarray(load_diag["selected_facet_mid_x_m"], dtype=np.float64) * 1e3
        selected_r_mm = np.asarray(load_diag["selected_facet_mid_r_m"], dtype=np.float64) * 1e3
        radius_dev_mm = np.asarray(load_diag["selected_facet_radius_deviation_m"], dtype=np.float64) * 1e3
        expected_x_min_mm = result["rim_top_metadata_mm"].get("blade_rim_top_x_min_mm")
        expected_x_max_mm = result["rim_top_metadata_mm"].get("blade_rim_top_x_max_mm")
        expected_r_mm = result["rim_top_metadata_mm"].get("blade_rim_top_r_mm")
        face_outside_interval = (
            bool(np.any(selected_x_mm < expected_x_min_mm - 0.5) or np.any(selected_x_mm > expected_x_max_mm + 0.5))
            if selected_ids.size and expected_x_min_mm is not None and expected_x_max_mm is not None
            else False
        )
        face_bad_subzone = any(name not in INTENDED_RIM_LOAD_SUBZONES for name in selected_subzones)
        face_non_horizontal = bool(np.any(np.abs(np.asarray(load_diag["selected_facet_delta_r_m"], dtype=np.float64)) > 1e-6))
        tree = cKDTree(mesh.nodes)
        centres = _landmark_centres(contour, radial_breaks)
        landmark_metrics = {}
        for landmark, radius in LANDMARK_NEIGHBOURHOODS_MM.items():
            if landmark not in centres:
                continue
            idx = _local_indices(tree, centres[landmark], radius, len(mesh.nodes))
            landmark_metrics[landmark] = {
                "node_count": int(len(idx)),
                "p90_stress_mpa": float(np.percentile(stress_max[idx], 90)),
                "max_stress_mpa": float(np.max(stress_max[idx])),
                "median_life_cycles": float(np.median(life_raw[idx])),
                "min_life_cycles": float(np.min(life_raw[idx])),
            }

        stress_stats = _stress_stats(stress_max)
        life_stats = _life_stats(life_raw)
        status, status_reasons = _evaluate_case_status(
            load_diag,
            stress_stats,
            life_stats,
            face_outside_interval,
            face_bad_subzone,
            geometry_ok,
            mesh_ok,
            fem_ok,
            metadata_present=all(k in result["rim_top_metadata_mm"] for k in ["blade_rim_top_r_mm", "blade_rim_top_x_min_mm", "blade_rim_top_x_max_mm"]),
            reason_codes=reason_codes,
        )
        result.update({
            "mesh_node_count": int(mesh.nodes.shape[0]),
            "mesh_element_count": int(mesh.triangles.shape[0]),
            "stress_stats": stress_stats,
            "life_stats": life_stats,
            "peak_coordinate_mm": [float(peak_xy[0]), float(peak_xy[1])],
            "nearest_peak_landmark": peak_landmark,
            "nearest_peak_landmark_distance_mm": float(peak_landmark_dist),
            "controlling_subzone": controlling_subzone,
            "geometry_valid": geometry_ok,
            "mesh_valid": mesh_ok,
            "rim_load_selection_valid": bool(selected_ids.size > 0 and not face_outside_interval and not face_bad_subzone and not face_non_horizontal),
            "force_closure_valid": bool(float(load_diag["closure_error_rel"]) <= 0.01),
            "closure_error_rel": float(load_diag["closure_error_rel"]),
            "landmark_metrics": landmark_metrics,
            "load_face_audit": {
                "selected_facet_count": int(selected_ids.size),
                "selected_node_count": int(facet_node_ids.size),
                "selected_facet_indices": selected_ids.tolist(),
                "selected_facet_midpoints_mm": selected_midpoints_mm.tolist(),
                "selected_facet_subzones": selected_subzones,
                "selected_meridional_face_length_mm": float(load_diag["face_length_m"]) * 1e3,
                "selected_radius_mean_mm": float(load_diag["mean_radius_m"]) * 1e3,
                "selected_radius_min_mm": float(np.min(selected_r_mm)) if selected_r_mm.size else None,
                "selected_radius_max_mm": float(np.max(selected_r_mm)) if selected_r_mm.size else None,
                "selected_x_range_mm": [float(np.min(selected_x_mm)), float(np.max(selected_x_mm))] if selected_x_mm.size else [None, None],
                "expected_rim_top_r_mm": expected_r_mm,
                "radial_deviation_per_facet_mm": radius_dev_mm.tolist(),
                "loaded_face_area_mm2": float(load_diag["face_area_m2"]) * 1e6,
                "any_selected_facet_non_horizontal": face_non_horizontal,
                "any_selected_facet_outside_interval": face_outside_interval,
                "any_selected_facet_on_unintended_region": face_bad_subzone,
            },
            "force_resultant_audit": _resultant_force_audit(load_diag),
            "phase_scaling_audit": _phase_scaling_audit(phase_stress, mesh.nodes, contour, radial_breaks),
            "status": status,
            "status_reasons": status_reasons,
        })
        if spec["case_name"] == "nominal":
            result["load_decomposition"] = _decomposition_summary(result | {"_contour": contour, "_mesh": mesh}, out_dir, save_plots)
        if save_plots:
            plot_dir = _ensure_dir(out_dir / "plots")
            face_plot = plot_dir / f"{spec['case_id']}_selected_load_face.png"
            _plot_selected_facets(spec["case_name"], contour, mesh, load_diag, face_plot)
            result["selected_load_face_plot"] = str(face_plot)
        result["_stress_max"] = stress_max
        result["_life_raw"] = life_raw
        result["_phase_stress"] = phase_stress
        result["_contour"] = contour
        result["_mesh"] = mesh
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["status_reasons"] = [type(exc).__name__, str(exc)]
        result["geometry_valid"] = geometry_ok
        result["mesh_valid"] = mesh_ok
        result["rim_load_selection_valid"] = False
        result["force_closure_valid"] = False
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
    return result


def run_validation(case: str, output_dir: Path, mesh_name: str, save_plots: bool, fail_on_invalid: bool, num_samples: int, seed: int) -> int:
    _patch_mesh(mesh_name)
    out_dir = _ensure_dir(output_dir)
    result_dir = _ensure_dir(out_dir / "results")
    specs, includes_lhs = _build_case_specs(case, num_samples, seed)
    case_results: List[Dict[str, Any]] = []
    case_results_raw: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    lhs_rows_requested: List[Dict[str, Any]] = []
    lhs_rows_actual: List[Dict[str, Any]] = []
    lhs_rows_controls: List[Dict[str, Any]] = []

    for spec in specs:
        res = validate_geometry_case(spec, mesh_name, out_dir, save_plots)
        case_results_raw.append(res)
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        _write_json(result_dir / f"{spec['case_id']}.json", clean)
        case_results.append(clean)
        csv_rows.append({
            "group": clean["group"],
            "case_id": clean["case_id"],
            "case_name": clean["case_name"],
            "status": clean["status"],
            "mesh": clean["mesh"],
            "peak_stress_mpa": clean.get("stress_stats", {}).get("global_peak_stress_mpa"),
            "stress_p95_mpa": clean.get("stress_stats", {}).get("stress_p95_mpa"),
            "stress_p99_mpa": clean.get("stress_stats", {}).get("stress_p99_mpa"),
            "min_life_cycles": clean.get("life_stats", {}).get("global_min_life_cycles"),
            "median_life_cycles": clean.get("life_stats", {}).get("median_life_cycles"),
            "median_log10_life": clean.get("life_stats", {}).get("median_log10_life"),
            "selected_facet_count": clean.get("load_face_audit", {}).get("selected_facet_count"),
            "selected_face_length_mm": clean.get("load_face_audit", {}).get("selected_meridional_face_length_mm"),
            "loaded_face_area_mm2": clean.get("load_face_audit", {}).get("loaded_face_area_mm2"),
            "closure_error_rel": clean.get("closure_error_rel"),
            "status_reasons": ";".join(clean.get("status_reasons", [])),
        })
        if clean["group"] == "lhs":
            lhs_rows_controls.append({
                "sample_id": spec.get("sample_id"),
                "seed": clean["seed"],
                **{f"requested_control__{k}": (clean.get("cgroove_controls_requested") or {}).get(k) for k in CGROOVE_SAMPLING_CONTROLS},
            })
            lhs_rows_requested.append({
                "sample_id": spec.get("sample_id"),
                "seed": clean["seed"],
                **{f"requested_core_offset__{k}": clean["requested_core_offsets"][k] for k in PUBLIC_GEOMETRY_PARAMETERS},
                **{f"requested_rim_offset__{k}": clean["requested_rim_offsets"][k] for k in RIM_FEATURE_PARAMETERS},
                **{f"resolved_core__{k}": clean.get("resolved_core", {}).get(k) for k in PUBLIC_GEOMETRY_PARAMETERS},
                **{f"resolved_rim__{k}": clean.get("resolved_rim", {}).get(k) for k in RIM_FEATURE_PARAMETERS},
            })
            lhs_rows_actual.append({
                "sample_id": spec.get("sample_id"),
                "seed": clean["seed"],
                **{f"actual_core__{k}": clean.get("actual_core", {}).get(k) for k in PUBLIC_GEOMETRY_PARAMETERS},
                **{f"actual_rim__{k}": clean.get("actual_rim", {}).get(k) for k in RIM_FEATURE_PARAMETERS},
                "status": clean["status"],
                "status_reasons": ";".join(clean.get("status_reasons", [])),
            })

    summary = {
        "case": case,
        "mesh": mesh_name,
        "num_cases": len(case_results),
        "pass_count": sum(r["status"] == "PASS" for r in case_results),
        "warning_count": sum(r["status"] == "WARNING" for r in case_results),
        "fail_count": sum(r["status"] == "FAIL" for r in case_results),
        "by_group": {
            group: {
                "pass_count": sum(r["status"] == "PASS" and r["group"] == group for r in case_results),
                "warning_count": sum(r["status"] == "WARNING" and r["group"] == group for r in case_results),
                "fail_count": sum(r["status"] == "FAIL" and r["group"] == group for r in case_results),
            }
            for group in sorted(set(r["group"] for r in case_results))
        },
    }

    _write_csv(out_dir / "all_cases_summary.csv", csv_rows)
    _write_json(out_dir / "summary.json", summary)
    if includes_lhs:
        lhs_results = [r for r in case_results if r["group"] == "lhs"]
        lhs_results_raw = [r for r in case_results_raw if r["group"] == "lhs"]
        nominal_spec = {
            "group": "nominal_reference",
            "case_name": "nominal_reference",
            "case_id": "nominal_reference",
            "sample_id": None,
            "core_offsets": {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS},
            "rim_offsets": {k: 0.0 for k in RIM_FEATURE_PARAMETERS},
            "cgroove_controls": None,
            "seed": int(seed),
        }
        nominal_raw = validate_geometry_case(nominal_spec, mesh_name, out_dir, save_plots=False)
        _write_csv(out_dir / "lhs_coverage_controls.csv", lhs_rows_controls)
        _write_csv(out_dir / "lhs_coverage_requested.csv", lhs_rows_requested)
        _write_csv(out_dir / "lhs_coverage_actual.csv", lhs_rows_actual)
        coverage = {
            "sampling_control_coverage": _sampling_control_coverage(lhs_results),
            "actual_geometry_parameter_coverage": _parameter_coverage(lhs_results),
            "life_sensitivity_audit": _life_sensitivity_audit(lhs_results_raw, nominal_raw, out_dir, save_plots),
        }
        _write_json(out_dir / "lhs_sanitization_summary.json", coverage)
        if save_plots and lhs_results:
            _save_lhs_plots(out_dir, lhs_results)
    if fail_on_invalid and summary["fail_count"] > 0:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rim load placement, force closure, phase scaling, and physical plausibility.")
    parser.add_argument("--case", choices=["nominal", "extrema", "lhs", "all"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mesh", choices=["medium", "fine"], default="medium")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rc = run_validation(
        case=args.case,
        output_dir=args.output_dir,
        mesh_name=args.mesh,
        save_plots=args.save_plots,
        fail_on_invalid=args.fail_on_invalid,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
