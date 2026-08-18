"""Feature-neighbourhood mesh and stress diagnostics for the C-groove / drive-arm geometry.

Usage
-----
    python Data_gen/mesh_feature_diagnostics.py --mesh medium
    python Data_gen/mesh_feature_diagnostics.py --mesh fine
    python Data_gen/mesh_feature_diagnostics.py --mesh medium --geometry high_feature

Results are written to ``Data_gen/output/diagnostics/<mesh>_<geometry>.json`` for
later comparison by ``compare_mesh_feature_diagnostics.py``.

Mesh sizes
----------
    medium: LC_EDGE=0.50, LC_FILLET=0.30  (production)
    fine:   LC_EDGE=0.30, LC_FILLET=0.18  (validation)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap for script-level invocation
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import Data_gen.mesh_ops as _mesh_ops  # imported before patching so we can patch constants

from Data_gen.config import (
    FIXED_BASELINE_BLEND_LANDMARK_NEIGHBOURHOOD_MM,
    NOMINAL_GEOMETRY_MM,
    NOMINAL_RIM_FEATURE_MM,
    SUBZONE_NAME_TO_ID,
    resolve_geometry_parameters,
    resolve_rim_feature_parameters,
)
from Data_gen.geometry import (
    build_disc_contour,
    sanitize_geometry_parameters,
    sanitize_rim_feature_parameters,
)
from Data_gen.mesh_ops import assign_zone_and_region_from_radius
from Data_gen.physics import (
    OMEGA_REF_RAD_S,
    compute_life_raw,
    compute_phase_equivalent_stresses,
    compute_stress_max,
)

# ---------------------------------------------------------------------------
# Mesh configurations
# ---------------------------------------------------------------------------
MESH_CONFIGS: Dict[str, Dict[str, float]] = {
    "medium": {"lc_edge": 0.50, "lc_fillet": 0.30},
    "fine":   {"lc_edge": 0.30, "lc_fillet": 0.18},
}

# ---------------------------------------------------------------------------
# Geometry presets
# ---------------------------------------------------------------------------
# Nominal: no offsets from nominal.
NOMINAL_CORE_OFFSETS: Dict[str, float] = {}
NOMINAL_RIM_OFFSETS: Dict[str, float] = {}

# Conservative high-feature: larger C-groove and larger arm within LHS bounds.
HIGH_FEATURE_CORE_OFFSETS: Dict[str, float] = {}
HIGH_FEATURE_RIM_OFFSETS: Dict[str, float] = {
    "front_cgroove_axial_depth":    +0.80,
    "front_cgroove_radial_span":    +0.40,
    "front_cgroove_radial_pos":     +0.15,
    "front_cgroove_entry_radius":   +0.10,
    "front_cgroove_floor_radius":   +0.10,
    "front_cgroove_exit_radius":    +0.10,
    "rear_arm_axial_projection":    +0.40,
    "rear_arm_radial_height":       +0.30,
    "rear_arm_neck_thickness":      +0.20,
    "rear_arm_root_radius":         +0.10,
    "rear_arm_outer_corner_radius": +0.10,
}

GEOMETRY_PRESETS: Dict[str, tuple] = {
    "nominal":      (NOMINAL_CORE_OFFSETS,      NOMINAL_RIM_OFFSETS),
    "high_feature": (HIGH_FEATURE_CORE_OFFSETS, HIGH_FEATURE_RIM_OFFSETS),
}

# ---------------------------------------------------------------------------
# Feature neighbourhood radii (mm) for local metrics.
# Arm-root features use a smaller radius (1.5 mm) to stay within the fillet
# region and avoid blending into the adjacent bulk rim, which has different
# stress levels and inflates p90 estimates in coarser meshes.
# ---------------------------------------------------------------------------
FEATURE_NEIGHBOURHOOD_MM: Dict[str, float] = {
    "bore_lower_rear_blend":          FIXED_BASELINE_BLEND_LANDMARK_NEIGHBOURHOOD_MM,
    "bore_lower_front_blend":         FIXED_BASELINE_BLEND_LANDMARK_NEIGHBOURHOOD_MM,
    "front_cgroove_entry":          2.0,
    "front_cgroove_floor":          2.0,
    "front_cgroove_exit":           2.0,
    "ligament_reference":           3.0,
    "rear_arm_root":                1.5,
    "rear_arm_neck":                1.5,
    "rear_arm_outer_corner":        1.5,
    "rear_arm_neck_rim_lower_blend": FIXED_BASELINE_BLEND_LANDMARK_NEIGHBOURHOOD_MM,
    "rear_arm_load_face_centroid":  2.0,
    "rim_core_reference":           4.0,
    "lower_transition_start":       4.0,
    "upper_transition_start":       4.0,
}


def _patch_mesh_lc(lc_edge: float, lc_fillet: float) -> None:
    """Patch mesh_ops module-level LC constants before calling generate_mesh."""
    _mesh_ops.LC_EDGE    = lc_edge
    _mesh_ops.LC_FILLET  = lc_fillet
    # Also patch LC_BULK if it exists
    if hasattr(_mesh_ops, "LC_BULK"):
        _mesh_ops.LC_BULK = max(lc_edge * 4.0, 1.5)


def _build_geometry(core_offsets: Dict[str, float], rim_offsets: Dict[str, float]):
    """Return (contour, actual_params, actual_rim_params, radial_breaks)."""
    actual_params = sanitize_geometry_parameters(resolve_geometry_parameters(core_offsets))
    raw_rim = resolve_rim_feature_parameters(rim_offsets)
    actual_rim = sanitize_rim_feature_parameters(
        raw_rim,
        t_rim=actual_params["rim_thickness"],
        bore_thickness=actual_params["bore_thickness"],
    )
    contour = build_disc_contour(actual_params, points_per_side=220, rim_feature_params=actual_rim)
    radial_breaks = contour.metadata["radial_breaks_mm"]
    return contour, actual_params, actual_rim, radial_breaks


def _run_fem(contour, actual_params, actual_rim, radial_breaks):
    """Mesh and solve FEM; return (nodes, zone_ids, stress_max, life_raw)."""
    mesh = _mesh_ops.generate_mesh(
        contour_points=contour.points,
        grid_x=90,
        grid_r=130,
        seed=0,
        radial_breaks=radial_breaks,
        geometry_params=actual_params,
        rim_feature_params=actual_rim,
    )
    zone_ids, region_ids = assign_zone_and_region_from_radius(
        nodes=mesh.nodes,
        radial_breaks=radial_breaks,
    )
    rim_meta = {k: v for k, v in contour.metadata.items() if k.startswith("blade_rim_top_")}
    phase_stress = compute_phase_equivalent_stresses(
        nodes=mesh.nodes,
        zone_ids=zone_ids,
        region_ids=region_ids,
        geometry_params=actual_params,
        radial_breaks=radial_breaks,
        mesh_obj=mesh.mesh,
        triangles=mesh.triangles,
        rim_face_metadata=rim_meta,
    )
    stress_max = compute_stress_max(phase_stress)
    life_raw = compute_life_raw(
        phase_stress=phase_stress,
        zone_ids=zone_ids,
        nodes=mesh.nodes,
        geometry_params=actual_params,
        radial_breaks=radial_breaks,
        lifing_mode="zonal",
    )
    return mesh.nodes, zone_ids, stress_max, life_raw, len(mesh.nodes), len(mesh.triangles)


def _local_metrics(
    nodes: np.ndarray,
    stress_max: np.ndarray,
    life_raw: np.ndarray,
    centre: np.ndarray,
    radius_mm: float,
) -> Dict[str, float]:
    """Compute local metrics within a physical neighbourhood sphere."""
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    idx = tree.query_ball_point(centre, radius_mm)
    if len(idx) == 0:
        # Expand search if needed
        _, nearest = tree.query(centre, k=min(10, len(nodes)))
        idx = list(np.atleast_1d(nearest))
    idx = np.array(idx, dtype=int)
    s = stress_max[idx]
    lf = life_raw[idx]
    return {
        "p90_stress_mpa":    float(np.percentile(s, 90)),
        "max_stress_mpa":    float(np.max(s)),
        "median_life_cycles": float(np.median(lf)),
        "min_life_cycles":   float(np.min(lf)),
        "node_count":        int(len(idx)),
        "neighbourhood_radius_mm": float(radius_mm),
    }


def _landmark_centre(landmarks: Dict[str, np.ndarray], name: str) -> Optional[np.ndarray]:
    if name not in landmarks:
        return None
    v = landmarks[name]
    if v.shape == (2,):
        return v.astype(float)
    if v.shape == (1,):
        # scalar radius — use (0, r) as approximate centre
        return np.array([0.0, float(v[0])], dtype=float)
    return v[:2].astype(float)


def run_diagnostics(mesh_name: str, geometry_name: str) -> Dict:
    """Run mesh diagnostics for the given mesh/geometry preset and return results dict."""
    cfg = MESH_CONFIGS[mesh_name]
    core_offsets, rim_offsets = GEOMETRY_PRESETS[geometry_name]

    print(f"\n=== Mesh feature diagnostics: mesh={mesh_name}, geometry={geometry_name} ===")
    print(f"  LC_EDGE={cfg['lc_edge']}, LC_FILLET={cfg['lc_fillet']}")

    _patch_mesh_lc(cfg["lc_edge"], cfg["lc_fillet"])
    contour, actual_params, actual_rim, radial_breaks = _build_geometry(core_offsets, rim_offsets)

    print(f"  Contour: {contour.points.shape[0]} points, "
          f"r_max={contour.points[:, 1].max():.2f} mm")
    print(f"  C-groove: depth={actual_rim['front_cgroove_axial_depth']:.2f} mm, "
          f"span={actual_rim['front_cgroove_radial_span']:.2f} mm")
    print(f"  Drive arm: proj={actual_rim['rear_arm_axial_projection']:.2f} mm, "
          f"height={actual_rim['rear_arm_radial_height']:.2f} mm, "
          f"neck={actual_rim['rear_arm_neck_thickness']:.2f} mm")

    nodes, zone_ids, stress_max, life_raw, n_nodes, n_elems = _run_fem(
        contour, actual_params, actual_rim, radial_breaks
    )
    print(f"  Mesh: {n_nodes} nodes, {n_elems} elements")
    print(f"  Global peak stress: {stress_max.max():.1f} MPa")
    print(f"  Global min life:    {life_raw.min():.3e} cycles")

    feature_results: Dict[str, Dict] = {}
    print("\n  Feature-neighbourhood metrics:")
    print(f"  {'Feature':<30} {'p90_stress':>12} {'max_stress':>12} "
          f"{'median_life':>14} {'min_life':>14} {'N_nodes':>8}")
    print("  " + "-" * 96)

    for feat_name, radius_mm in FEATURE_NEIGHBOURHOOD_MM.items():
        ctr = _landmark_centre(contour.landmarks_mm, feat_name)
        if ctr is None:
            # Fallback: radial-break derived centres
            rb = radial_breaks
            fallback = {
                "lower_transition_start": np.array([0.0, float(rb[1])]),
                "upper_transition_start": np.array([0.0, float(rb[3])]),
            }
            ctr = fallback.get(feat_name)
        if ctr is None:
            feature_results[feat_name] = {"error": "landmark not found"}
            continue
        m = _local_metrics(nodes, stress_max, life_raw, ctr, radius_mm)
        feature_results[feat_name] = m
        print(f"  {feat_name:<30} {m['p90_stress_mpa']:>12.1f} {m['max_stress_mpa']:>12.1f} "
              f"{m['median_life_cycles']:>14.3e} {m['min_life_cycles']:>14.3e} "
              f"{m['node_count']:>8d}")

    # Comparison: C-groove features vs rim_core_reference
    rim_core = feature_results.get("rim_core_reference", {})
    rim_core_p90 = rim_core.get("p90_stress_mpa", float("nan"))
    rim_core_life = rim_core.get("median_life_cycles", float("nan"))

    print("\n  Feature vs rim-core comparison:")
    for feat in ["front_cgroove_entry", "front_cgroove_floor", "front_cgroove_exit",
                 "rear_arm_root", "rear_arm_neck", "rear_arm_outer_corner",
                 "rear_arm_load_face_centroid", "ligament_reference"]:
        fr = feature_results.get(feat, {})
        if "error" in fr or "p90_stress_mpa" not in fr:
            continue
        stress_ratio = fr["p90_stress_mpa"] / max(rim_core_p90, 1e-9)
        life_ratio   = fr["median_life_cycles"] / max(rim_core_life, 1e-9)
        print(f"  {feat:<30}  stress_ratio={stress_ratio:.3f}  life_ratio={life_ratio:.3f}")

    results = {
        "mesh": mesh_name,
        "geometry": geometry_name,
        "lc_edge": cfg["lc_edge"],
        "lc_fillet": cfg["lc_fillet"],
        "n_nodes": n_nodes,
        "n_elements": n_elems,
        "global_peak_stress_mpa": float(stress_max.max()),
        "global_min_life_cycles": float(life_raw.min()),
        "actual_rim_feature_params": {k: float(v) for k, v in actual_rim.items()},
        "features": feature_results,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-neighbourhood mesh diagnostics.")
    parser.add_argument("--mesh", choices=["medium", "fine"], default="medium",
                        help="Mesh resolution preset (default: medium).")
    parser.add_argument("--geometry", choices=list(GEOMETRY_PRESETS.keys()), default=None,
                        help="Geometry preset. If omitted, both nominal and high_feature are run.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Data_gen/output/diagnostics"),
                        help="Directory for JSON output files.")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    geom_list = [args.geometry] if args.geometry else list(GEOMETRY_PRESETS.keys())

    all_results = {}
    for geom_name in geom_list:
        results = run_diagnostics(args.mesh, geom_name)
        all_results[geom_name] = results
        out_path = out_dir / f"{args.mesh}_{geom_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n  Saved: {out_path}")

    print(f"\n=== Diagnostics complete (mesh={args.mesh}) ===")


if __name__ == "__main__":
    main()
