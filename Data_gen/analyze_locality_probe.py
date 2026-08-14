"""Locality probe analysis for the disc lifing pipeline.

Evaluates how local feature-neighbourhood stress/life extrema relate to
neighbouring regions and the global field, for nominal and asymmetric samples.

Usage
-----
    python Data_gen/analyze_locality_probe.py
    python Data_gen/analyze_locality_probe.py --geometry high_feature
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from Data_gen.config import (
    resolve_geometry_parameters,
    resolve_rim_feature_parameters,
)
from Data_gen.geometry import (
    build_disc_contour,
    sanitize_geometry_parameters,
    sanitize_rim_feature_parameters,
)
import Data_gen.mesh_ops as _mesh_ops
from Data_gen.mesh_ops import assign_zone_and_region_from_radius
from Data_gen.physics import (
    compute_life_raw,
    compute_phase_equivalent_stresses,
    compute_stress_max,
)

# ---------------------------------------------------------------------------
# Geometry presets (same as mesh_feature_diagnostics.py)
# ---------------------------------------------------------------------------
GEOMETRY_PRESETS: Dict[str, tuple] = {
    "nominal": ({}, {}),
    "high_feature": ({}, {
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
    }),
}

# Feature neighbourhoods and their local comparison baselines
FEATURE_NEIGHBOURHOODS: List[Dict] = [
    {"name": "front_cgroove_entry",       "radius": 2.0, "baseline": "rim_core_reference"},
    {"name": "front_cgroove_floor",       "radius": 2.0, "baseline": "rim_core_reference"},
    {"name": "front_cgroove_exit",        "radius": 2.0, "baseline": "ligament_reference"},
    {"name": "ligament_reference",        "radius": 3.0, "baseline": "rim_core_reference"},
    {"name": "rear_arm_root",             "radius": 1.5, "baseline": "rim_core_reference"},
    {"name": "rear_arm_neck",             "radius": 1.5, "baseline": "rim_core_reference"},
    {"name": "rear_arm_outer_corner",     "radius": 1.5, "baseline": "rim_core_reference"},
    {"name": "rear_arm_load_face_centroid", "radius": 2.0, "baseline": "rim_core_reference"},
    {"name": "rim_core_reference",        "radius": 4.0, "baseline": None},
    {"name": "lower_transition_start",    "radius": 4.0, "baseline": None},
    {"name": "upper_transition_start",    "radius": 4.0, "baseline": None},
]


def _build_geometry(core_offsets, rim_offsets):
    actual_params = sanitize_geometry_parameters(resolve_geometry_parameters(core_offsets))
    raw_rim = resolve_rim_feature_parameters(rim_offsets)
    actual_rim = sanitize_rim_feature_parameters(
        raw_rim,
        t_rim=actual_params["rim_thickness"],
        bore_thickness=actual_params["bore_thickness"],
    )
    contour = build_disc_contour(actual_params, points_per_side=220, rim_feature_params=actual_rim)
    return contour, actual_params, actual_rim, contour.metadata["radial_breaks_mm"]


def _run_fem(contour, actual_params, actual_rim, radial_breaks):
    mesh = _mesh_ops.generate_mesh(
        contour_points=contour.points,
        grid_x=90, grid_r=130, seed=0,
        radial_breaks=radial_breaks,
        geometry_params=actual_params,
        rim_feature_params=actual_rim,
    )
    zone_ids, region_ids = assign_zone_and_region_from_radius(
        nodes=mesh.nodes, radial_breaks=radial_breaks,
    )
    rim_meta = {k: v for k, v in contour.metadata.items() if k.startswith("blade_rim_top_")}
    phase_stress = compute_phase_equivalent_stresses(
        nodes=mesh.nodes, zone_ids=zone_ids, region_ids=region_ids,
        geometry_params=actual_params, radial_breaks=radial_breaks,
        mesh_obj=mesh.mesh, triangles=mesh.triangles,
        rim_face_metadata=rim_meta,
    )
    stress_max = compute_stress_max(phase_stress)
    life_raw = compute_life_raw(
        phase_stress=phase_stress, zone_ids=zone_ids,
        nodes=mesh.nodes, geometry_params=actual_params,
        radial_breaks=radial_breaks, lifing_mode="zonal",
    )
    return mesh.nodes, stress_max, life_raw


def _local_metrics(nodes, stress_max, life_raw, centre, radius_mm):
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    idx = tree.query_ball_point(centre, radius_mm)
    if len(idx) == 0:
        _, idx = tree.query(centre, k=min(10, len(nodes)))
        idx = list(np.atleast_1d(idx))
    idx = np.array(idx, dtype=int)
    s = stress_max[idx]
    lf = life_raw[idx]
    return {
        "p90_stress_mpa":      float(np.percentile(s, 90)),
        "max_stress_mpa":      float(np.max(s)),
        "median_life_cycles":  float(np.median(lf)),
        "min_life_cycles":     float(np.min(lf)),
        "node_count":          int(len(idx)),
    }


def _landmark_centre(landmarks, name, radial_breaks):
    if name in landmarks:
        v = landmarks[name]
        if v.shape == (2,):
            return v.astype(float)
        if v.shape == (1,):
            return np.array([0.0, float(v[0])], dtype=float)
    # Fallbacks for radial-break based features
    fallbacks = {
        "lower_transition_start": np.array([0.0, float(radial_breaks[1])]),
        "upper_transition_start": np.array([0.0, float(radial_breaks[3])]),
        "rim_core_reference":     np.array([0.0, float(radial_breaks[4]) + 3.0]),
    }
    return fallbacks.get(name)


def analyze(geometry_name: str) -> None:
    core_offsets, rim_offsets = GEOMETRY_PRESETS[geometry_name]

    print(f"\n{'='*60}")
    print(f"Locality probe: {geometry_name}")
    print(f"{'='*60}")

    contour, actual_params, actual_rim, radial_breaks = _build_geometry(core_offsets, rim_offsets)

    print(f"  C-groove: depth={actual_rim['front_cgroove_axial_depth']:.2f} mm, "
          f"span={actual_rim['front_cgroove_radial_span']:.2f} mm")
    print(f"  Drive arm: proj={actual_rim['rear_arm_axial_projection']:.2f} mm, "
          f"height={actual_rim['rear_arm_radial_height']:.2f} mm")
    t_rim = actual_params["rim_thickness"]
    ligament = t_rim - actual_rim["front_cgroove_axial_depth"]
    print(f"  Ligament thickness: {ligament:.2f} mm")

    nodes, stress_max, life_raw = _run_fem(contour, actual_params, actual_rim, radial_breaks)
    print(f"  Mesh nodes: {len(nodes)}")
    print(f"  Global peak stress: {stress_max.max():.1f} MPa")
    print(f"  Global min life: {life_raw.min():.3e} cycles")

    # Collect all local metrics
    metrics: Dict[str, Dict] = {}
    for feat in FEATURE_NEIGHBOURHOODS:
        ctr = _landmark_centre(contour.landmarks_mm, feat["name"], radial_breaks)
        if ctr is None:
            metrics[feat["name"]] = {}
            continue
        metrics[feat["name"]] = _local_metrics(nodes, stress_max, life_raw, ctr, feat["radius"])

    print(f"\n  {'Feature':<30} {'p90_s':>8} {'max_s':>8} {'med_N':>10} {'min_N':>10} "
          f"{'vs_baseline(s)':>16} {'vs_baseline(N)':>16}")
    print("  " + "-" * 100)

    for feat in FEATURE_NEIGHBOURHOODS:
        m = metrics.get(feat["name"], {})
        if not m:
            print(f"  {feat['name']:<30}  (landmark not found)")
            continue

        baseline_name = feat["baseline"]
        bm = metrics.get(baseline_name, {}) if baseline_name else {}
        if bm and "p90_stress_mpa" in bm:
            stress_ratio = f"{m['p90_stress_mpa'] / max(bm['p90_stress_mpa'], 1e-9):.2f}x"
            life_ratio   = f"{m['median_life_cycles'] / max(bm['median_life_cycles'], 1e-9):.2f}x"
        else:
            stress_ratio = "—"
            life_ratio   = "—"

        print(f"  {feat['name']:<30} {m['p90_stress_mpa']:>8.1f} {m['max_stress_mpa']:>8.1f} "
              f"{m['median_life_cycles']:>10.3e} {m['min_life_cycles']:>10.3e} "
              f"{stress_ratio:>16} {life_ratio:>16}")

    # Check arm load face has non-trivial stress
    alf = metrics.get("rear_arm_load_face_centroid", {})
    rc  = metrics.get("rim_core_reference", {})
    if alf and rc:
        s_ratio = alf["p90_stress_mpa"] / max(rc["p90_stress_mpa"], 1e-9)
        l_ratio = alf["median_life_cycles"] / max(rc["median_life_cycles"], 1e-9)
        ok = s_ratio > 0.5
        print(f"\n  Arm load face stress ratio vs rim core: {s_ratio:.2f}x  "
              f"({'OK' if ok else 'LOW — check blade traction'})")
        print(f"  Arm load face life ratio vs rim core:   {l_ratio:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Locality probe analysis.")
    parser.add_argument("--geometry", choices=list(GEOMETRY_PRESETS.keys()),
                        default=None,
                        help="Geometry preset. If omitted, all presets are analysed.")
    args = parser.parse_args()

    geoms = [args.geometry] if args.geometry else list(GEOMETRY_PRESETS.keys())
    for g in geoms:
        analyze(g)

    print("\nLocality probe analysis complete.")


if __name__ == "__main__":
    main()
