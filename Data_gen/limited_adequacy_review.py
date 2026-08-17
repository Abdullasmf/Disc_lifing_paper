from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from Data_gen.config import (
    CYCLE_PHASES,
    CYCLE_SPEED_FACTORS,
    NOMINAL_GEOMETRY_MM,
    NOMINAL_RIM_FEATURE_MM,
    PUBLIC_GEOMETRY_PARAMETERS,
    RIM_FEATURE_PARAMETERS,
    ZONAL_SN_PARAMS,
)
from Data_gen.validate_rim_load_and_physics import (
    LANDMARK_NEIGHBOURHOODS_MM,
    _patch_mesh,
    validate_geometry_case,
)


MESH_LANDMARKS = [
    "lower_transition",
    "upper_transition",
    "rim_core_reference",
    "front_cgroove_entry",
    "front_cgroove_floor",
    "front_cgroove_exit",
    "rear_arm_root",
    "rear_arm_neck",
    "rear_arm_outer_corner",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _choose_severe_case(lhs_dirs: List[Path]) -> Dict[str, Any]:
    best_row: Dict[str, Any] | None = None
    for out_dir in lhs_dirs:
        rows = _read_csv(out_dir / "lhs_geometry_to_life_sensitivity.csv")
        for row in rows:
            if row.get("status") != "PASS":
                continue
            d = float(row.get("delta_loglife_min", "nan"))
            if not np.isfinite(d):
                continue
            if best_row is None or d < float(best_row["delta_loglife_min"]):
                best_row = {"output_dir": str(out_dir), **row}
    if best_row is None:
        raise RuntimeError("No valid LHS rows found for severe-case selection.")
    src = Path(best_row["output_dir"]) / "results" / f"{best_row['case_id']}.json"
    payload = json.loads(src.read_text())
    return payload


def _build_spec_from_result(row: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    return {
        "group": "adequacy",
        "case_name": case_id,
        "case_id": case_id,
        "sample_id": row.get("sample_id"),
        "core_offsets": row.get("requested_core_offsets", {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS}),
        "rim_offsets": row.get("requested_rim_offsets", {k: 0.0 for k in RIM_FEATURE_PARAMETERS}),
        "cgroove_controls": row.get("cgroove_controls_requested"),
        "seed": int(row.get("seed", 7)),
    }


def _local_mesh_size(mesh_obj, centre: np.ndarray, radius_mm: float = 1.0) -> float:
    nodes = np.asarray(mesh_obj.nodes, dtype=np.float64)
    tri = np.asarray(mesh_obj.triangles, dtype=int)
    tree = cKDTree(nodes)
    idx = tree.query_ball_point(centre, radius_mm)
    if len(idx) == 0:
        idx = [int(tree.query(centre, k=1)[1])]
    idx_set = set(int(i) for i in idx)
    incident = [tid for tid, t in enumerate(tri) if any(int(v) in idx_set for v in t)]
    edges = []
    for tid in incident:
        t = tri[tid]
        p = nodes[t]
        edges.extend([np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[2] - p[0])])
    return float(np.median(edges)) if edges else float("nan")


def _run_case_at_mesh(case_spec: Dict[str, Any], mesh_name: str, out_dir: Path) -> Dict[str, Any]:
    _patch_mesh(mesh_name)
    return validate_geometry_case(case_spec, mesh_name, out_dir=out_dir, save_plots=False)


def _mesh_convergence_row(case_name: str, med: Dict[str, Any], fine: Dict[str, Any]) -> Dict[str, Any]:
    med_lm = med.get("landmark_metrics", {})
    fine_lm = fine.get("landmark_metrics", {})
    stress_changes = []
    loglife_changes = []
    for lm in MESH_LANDMARKS:
        if lm not in med_lm or lm not in fine_lm:
            continue
        m_st = float(med_lm[lm]["p90_stress_mpa"])
        f_st = float(fine_lm[lm]["p90_stress_mpa"])
        stress_changes.append(abs(f_st - m_st) / max(abs(m_st), 1e-9))
        m_ll = math.log10(max(float(med_lm[lm]["median_life_cycles"]), 1e-300))
        f_ll = math.log10(max(float(fine_lm[lm]["median_life_cycles"]), 1e-300))
        loglife_changes.append(abs(f_ll - m_ll))

    p90_stress_conv = float(np.percentile(stress_changes, 90)) if stress_changes else float("nan")
    median_loglife_conv = float(np.median(loglife_changes)) if loglife_changes else float("nan")
    peak_delta = abs(
        float(fine.get("stress_stats", {}).get("global_peak_stress_mpa", np.nan))
        - float(med.get("stress_stats", {}).get("global_peak_stress_mpa", np.nan))
    ) / max(abs(float(med.get("stress_stats", {}).get("global_peak_stress_mpa", 1.0))), 1e-9)
    feature_switch = (
        med.get("nearest_peak_landmark") != fine.get("nearest_peak_landmark")
        or med.get("controlling_subzone") != fine.get("controlling_subzone")
    )
    rim = med.get("actual_rim", {})
    core = med.get("actual_core", {})
    smallest_radius = min(
        float(core.get("lower_fillet_radius", np.inf)),
        float(core.get("upper_fillet_radius", np.inf)),
        float(rim.get("front_cgroove_entry_radius", np.inf)),
        float(rim.get("front_cgroove_floor_radius", np.inf)),
        float(rim.get("front_cgroove_exit_radius", np.inf)),
        float(rim.get("rear_arm_root_radius", np.inf)),
        float(rim.get("rear_arm_outer_corner_radius", np.inf)),
    )
    peak_xy = np.asarray(med.get("peak_coordinate_mm", [0.0, 0.0]), dtype=np.float64)
    local_h = _local_mesh_size(med["_mesh"], peak_xy, radius_mm=1.0)
    h_over_r = local_h / max(smallest_radius, 1e-9)
    verdict = "MESH ADEQUATE"
    if feature_switch or peak_delta > 0.15:
        verdict = "MESH ADEQUATE WITH LIMITATIONS"
    if (p90_stress_conv > 0.15) or (median_loglife_conv > 0.10):
        verdict = "MESH INADEQUATE"
    return {
        "geometry_checked": case_name,
        "critical_feature": med.get("nearest_peak_landmark"),
        "smallest_radius_mm": smallest_radius,
        "medium_nodes": int(med.get("mesh_node_count", 0)),
        "medium_elements": int(med.get("mesh_element_count", 0)),
        "fine_nodes": int(fine.get("mesh_node_count", 0)),
        "fine_elements": int(fine.get("mesh_element_count", 0)),
        "local_mesh_size_over_smallest_radius": h_over_r,
        "stress_convergence_p90_rel": p90_stress_conv,
        "loglife_convergence_abs_median_decades": median_loglife_conv,
        "peak_stress_rel_change": peak_delta,
        "governing_feature_changed": feature_switch,
        "verdict": verdict,
    }


def _sn_regime_row(case_name: str, case_res: Dict[str, Any]) -> Dict[str, Any]:
    phase_stress = np.asarray(case_res["_phase_stress"], dtype=np.float64)
    life = np.asarray(case_res["_life_raw"], dtype=np.float64)
    sigma_a = 0.5 * np.maximum(phase_stress, 0.0)
    smin = float(np.min(sigma_a))
    smax = float(np.max(sigma_a))
    knee_vals = [float(v["knee_stress_mpa"]) for v in ZONAL_SN_PARAMS.values()]
    knee_min = min(knee_vals)
    knee_max = max(knee_vals)
    inside_domain = (smin >= 0.0) and (smax <= 3.0 * knee_max)
    extrapolation_risk = "low"
    if smax > 3.5 * knee_max or smin < 0.01 * knee_min:
        extrapolation_risk = "high"
    elif smax > 3.0 * knee_max or smin < 0.02 * knee_min:
        extrapolation_risk = "moderate"
    hi_life_thresh = max(float(v["knee_life"]) for v in ZONAL_SN_PARAMS.values())
    verdict = "S-N ADEQUATE" if inside_domain and extrapolation_risk == "low" else "S-N ADEQUATE WITH LIMITATIONS"
    if extrapolation_risk == "high":
        verdict = "S-N INADEQUATE FOR CURRENT DEVIATION ENVELOPE"
    return {
        "geometry_checked": case_name,
        "stress_amplitude_min_mpa": smin,
        "stress_amplitude_max_mpa": smax,
        "global_peak_stress_mpa": float(case_res.get("stress_stats", {}).get("global_peak_stress_mpa", np.nan)),
        "critical_local_stress_mpa": float(np.percentile(np.asarray(case_res["_stress_max"], dtype=np.float64), 99)),
        "global_min_life_cycles": float(np.min(life)),
        "life_bin_frac_lt_10": float(np.mean(life < 10)),
        "life_bin_frac_10_to_100": float(np.mean((life >= 10) & (life < 100))),
        "life_bin_frac_100_to_1k": float(np.mean((life >= 100) & (life < 1_000))),
        "life_bin_frac_1k_to_10k": float(np.mean((life >= 1_000) & (life < 10_000))),
        "life_bin_frac_above_hi_life_knee": float(np.mean(life >= hi_life_thresh)),
        "inside_stated_sn_domain": bool(inside_domain),
        "extrapolation_risk": extrapolation_risk,
        "verdict": verdict,
    }


def _deviation_realism_rows(lhs_dirs: List[Path]) -> List[Dict[str, Any]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    for out_dir in lhs_dirs:
        payload = json.loads((out_dir / "lhs_sanitization_summary.json").read_text())
        for entry in payload["actual_geometry_parameter_coverage"]["parameters"].values():
            k = entry["parameter"]
            lo = float(entry["actual_final_min"])
            hi = float(entry["actual_final_max"])
            if k not in ranges:
                ranges[k] = (lo, hi)
            else:
                ranges[k] = (min(ranges[k][0], lo), max(ranges[k][1], hi))
    rows: List[Dict[str, Any]] = []
    scan_visible_false = {"bore_thickness", "web_thickness", "rim_thickness"}
    design_family = {"rear_arm_axial_projection"}
    out_of_tol = {"front_cgroove_axial_depth", "front_cgroove_radial_span", "front_cgroove_radial_pos"}
    for k in list(PUBLIC_GEOMETRY_PARAMETERS) + list(RIM_FEATURE_PARAMETERS):
        nominal = float(NOMINAL_GEOMETRY_MM.get(k, NOMINAL_RIM_FEATURE_MM.get(k)))
        lo, hi = ranges.get(k, (nominal, nominal))
        rel = max(abs(lo - nominal), abs(hi - nominal)) / max(abs(nominal), 1e-9)
        if k in design_family:
            cls = "Design-family variation rather than manufacturing tolerance"
            rec = "narrow"
        elif k in out_of_tol:
            cls = "Plausible out-of-tolerance / repair / rework variation"
            rec = "keep"
        elif rel <= 0.06:
            cls = "Plausible routine manufacturing variation"
            rec = "keep"
        else:
            cls = "Plausible out-of-tolerance / repair / rework variation"
            rec = "narrow"
        rows.append({
            "parameter": k,
            "nominal_mm": nominal,
            "actual_sampled_min_mm": lo,
            "actual_sampled_max_mm": hi,
            "relative_nominal_deviation": rel,
            "scan_visible": str(k not in scan_visible_false).lower(),
            "classification": cls,
            "recommendation": rec,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Limited mesh/S-N/deviation adequacy review.")
    ap.add_argument("--lhs-output-dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lhs_dirs = [Path(p) for p in args.lhs_output_dirs]

    severe_src = _choose_severe_case(lhs_dirs)
    nominal_case = {
        "group": "adequacy",
        "case_name": "nominal",
        "case_id": "nominal",
        "sample_id": None,
        "core_offsets": {k: 0.0 for k in PUBLIC_GEOMETRY_PARAMETERS},
        "rim_offsets": {k: 0.0 for k in RIM_FEATURE_PARAMETERS},
        "cgroove_controls": None,
        "seed": 7,
    }
    severe_case = _build_spec_from_result(severe_src, case_id=f"severe_{severe_src.get('case_id')}")

    med_nom = _run_case_at_mesh(nominal_case, "medium", out_dir)
    fine_nom = _run_case_at_mesh(nominal_case, "fine", out_dir)
    med_sev = _run_case_at_mesh(severe_case, "medium", out_dir)
    fine_sev = _run_case_at_mesh(severe_case, "fine", out_dir)

    mesh_rows = [
        _mesh_convergence_row("nominal", med_nom, fine_nom),
        _mesh_convergence_row("severe_but_valid", med_sev, fine_sev),
    ]
    _write_csv(out_dir / "mesh_adequacy_table.csv", mesh_rows)

    sn_rows = [
        _sn_regime_row("nominal", med_nom),
        _sn_regime_row("severe_but_valid", med_sev),
    ]
    _write_csv(out_dir / "sn_adequacy_table.csv", sn_rows)

    realism_rows = _deviation_realism_rows(lhs_dirs)
    _write_csv(out_dir / "deviation_realism_table.csv", realism_rows)

    summary = {
        "contour_correspondence_method": "uniform arc-length resampled contour + nearest FEM node query",
        "mesh_verdict": "MESH ADEQUATE" if all(r["verdict"] == "MESH ADEQUATE" for r in mesh_rows) else "MESH ADEQUATE WITH LIMITATIONS",
        "sn_verdict": "S-N ADEQUATE" if all(r["verdict"] == "S-N ADEQUATE" for r in sn_rows) else "S-N ADEQUATE WITH LIMITATIONS",
        "severe_case_reference": severe_case["case_id"],
        "phase_names": list(CYCLE_PHASES),
        "phase_speed_factors_sq": (CYCLE_SPEED_FACTORS ** 2).tolist(),
    }
    (out_dir / "limited_adequacy_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
