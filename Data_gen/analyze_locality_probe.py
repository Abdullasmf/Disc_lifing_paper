r"""First-pass locality diagnosis for a full Data_gen HDF5 dataset.

Example:
  & c:\Users\abfat\miniconda3\envs\MLEnv\python.exe Data_gen/analyze_locality_probe.py `
    --input-h5 Data_gen/output/locality_probe_50.h5 `
    --output-dir Data_gen/output/locality_probe_50_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ZONE_NAMES = {0: "bore_hub", 1: "lower_transition", 2: "web", 3: "upper_transition", 4: "rim_family"}
SUBZONE_NAMES = {
    0: "bore", 1: "lower_transition", 2: "web", 3: "upper_transition", 4: "rim_main",
    5: "front_step", 6: "rear_step", 7: "front_shoulder", 8: "rear_shoulder",
    9: "front_groove", 10: "rear_platform", 11: "rear_platform_root",
}
LANDMARK_LABELS = {
    "front_groove_entry": "front_groove_entry",
    "front_groove_floor": "front_groove_floor_root",
    "front_groove_root": "front_groove_floor_root",
    "front_groove_exit": "front_groove_exit",
    "front_root": "front_step_root",
    "front_outer_corner": "front_step_outer_corner",
    "rear_platform_root": "rear_platform_root",
    "rear_root": "rear_platform_root",
    "rear_platform_outer_corner": "rear_platform_outer_corner",
    "rear_outer_corner": "rear_platform_outer_corner",
    "rear_load_transfer": "rear_load_transfer_face",
    "rear_load_transfer_face": "rear_load_transfer_face",
    "rim_core_reference": "rim_core",
}


def _decode(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode(errors="replace")
    return str(v)


def _find_sample_groups(h5):
    groups = []
    def visit(name, obj):
        if isinstance(obj, h5py.Group) and {"node_coords_mm", "life_raw", "stress_max_vm"}.issubset(obj.keys()):
            groups.append((name, obj))
    h5.visititems(visit)
    return sorted(groups, key=lambda x: x[0])


def _landmarks(group):
    out = {}
    if "feature_landmarks_mm" not in group:
        return out
    for key, ds in group["feature_landmarks_mm"].items():
        a = np.asarray(ds, dtype=float).ravel()
        if a.size >= 2 and np.all(np.isfinite(a[:2])):
            out[key] = a[:2]
    return out


def _region_for_node(point, zone_id, subzone_id, landmarks, radius):
    nearest_key, nearest_dist = None, np.inf
    for key, xy in landmarks.items():
        d = float(np.linalg.norm(point - xy))
        if d < nearest_dist:
            nearest_key, nearest_dist = key, d
    if nearest_key is not None and nearest_dist <= radius:
        return LANDMARK_LABELS.get(nearest_key, nearest_key), nearest_key, nearest_dist
    if subzone_id is not None and int(subzone_id) in SUBZONE_NAMES:
        return SUBZONE_NAMES[int(subzone_id)], "", nearest_dist
    return ZONE_NAMES.get(int(zone_id), f"zone_{int(zone_id)}"), "", nearest_dist


def _local_stats(nodes, stress, life, center, radius):
    d = np.linalg.norm(nodes - center[None, :], axis=1)
    m = d <= radius
    if not np.any(m):
        return {"n": 0, "stress_p90": np.nan, "stress_max": np.nan, "life_median": np.nan, "life_min": np.nan}
    s, l = stress[m], life[m]
    return {
        "n": int(m.sum()),
        "stress_p90": float(np.percentile(s, 90)),
        "stress_max": float(np.max(s)),
        "life_median": float(np.median(l)),
        "life_min": float(np.min(l)),
    }


def _write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Diagnose critical-life locations and named rim-feature neighborhoods.")
    p.add_argument("--input-h5", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--landmark-radius-mm", type=float, default=0.75)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    critical_rows, neighborhood_rows = [], []

    with h5py.File(args.input_h5, "r") as h5:
        samples = _find_sample_groups(h5)
        if not samples:
            raise RuntimeError("No sample groups found containing node_coords_mm, life_raw, and stress_max_vm.")
        for i, (name, g) in enumerate(samples):
            nodes = np.asarray(g["node_coords_mm"], dtype=float)
            life = np.asarray(g["life_raw"], dtype=float).ravel()
            stress = np.asarray(g["stress_max_vm"], dtype=float).ravel()
            zone = np.asarray(g.get("zone_id", np.full(len(life), -1)), dtype=int).ravel()
            subzone = np.asarray(g["subzone_id"], dtype=int).ravel() if "subzone_id" in g else None
            if not (len(nodes) == len(life) == len(stress) == len(zone)):
                raise RuntimeError(f"Shape mismatch in {name}: nodes={len(nodes)}, life={len(life)}, stress={len(stress)}, zone={len(zone)}")
            lm = _landmarks(g)
            i_life, i_stress = int(np.nanargmin(life)), int(np.nanargmax(stress))
            life_sub = None if subzone is None else int(subzone[i_life])
            stress_sub = None if subzone is None else int(subzone[i_stress])
            life_region, life_lm, life_d = _region_for_node(nodes[i_life], zone[i_life], life_sub, lm, args.landmark_radius_mm)
            stress_region, stress_lm, stress_d = _region_for_node(nodes[i_stress], zone[i_stress], stress_sub, lm, args.landmark_radius_mm)
            critical_rows.append({
                "sample_index": i, "sample_group": name,
                "critical_region": life_region, "critical_landmark": life_lm, "critical_landmark_distance_mm": life_d,
                "critical_x_mm": nodes[i_life, 0], "critical_r_mm": nodes[i_life, 1],
                "critical_life_raw": life[i_life], "critical_stress_vm_mpa": stress[i_life],
                "critical_zone_id": int(zone[i_life]), "critical_subzone_id": life_sub,
                "max_stress_region": stress_region, "max_stress_landmark": stress_lm, "max_stress_landmark_distance_mm": stress_d,
                "max_stress_x_mm": nodes[i_stress, 0], "max_stress_r_mm": nodes[i_stress, 1],
                "max_stress_vm_mpa": stress[i_stress], "max_stress_life_raw": life[i_stress],
                "max_stress_zone_id": int(zone[i_stress]), "max_stress_subzone_id": stress_sub,
            })
            for key, xy in lm.items():
                stat = _local_stats(nodes, stress, life, xy, args.landmark_radius_mm)
                neighborhood_rows.append({"sample_index": i, "sample_group": name, "landmark": key,
                                          "feature_region": LANDMARK_LABELS.get(key, key), "x_mm": xy[0], "r_mm": xy[1],
                                          **stat})

    _write_csv(args.output_dir / "critical_locations.csv", critical_rows)
    _write_csv(args.output_dir / "feature_neighborhoods.csv", neighborhood_rows)

    counts = Counter(r["critical_region"] for r in critical_rows)
    stress_counts = Counter(r["max_stress_region"] for r in critical_rows)
    summary = {
        "n_samples": len(critical_rows), "landmark_radius_mm": args.landmark_radius_mm,
        "critical_life_region_counts": dict(counts), "max_stress_region_counts": dict(stress_counts),
        "critical_life_region_fraction": {k: v / len(critical_rows) for k, v in counts.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels, values = zip(*sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(labels)), 5))
    ax.bar(labels, values, color="#3a7ca5")
    ax.set_ylabel("Number of samples")
    ax.set_title("Minimum-life location across locality probe")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(args.output_dir / "critical_location_frequency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    unique = {k: j for j, k in enumerate(sorted(counts))}
    for region, j in unique.items():
        rows = [r for r in critical_rows if r["critical_region"] == region]
        ax.scatter([r["critical_x_mm"] for r in rows], [r["critical_r_mm"] for r in rows], s=35, alpha=0.8, label=region)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title("Minimum-life node locations")
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(args.output_dir / "critical_location_scatter.png", dpi=180)
    plt.close(fig)

    print(f"Analysed {len(critical_rows)} samples")
    print("Minimum-life region counts:")
    for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}/{len(critical_rows)} ({100*v/len(critical_rows):.1f}%)")
    print(f"Wrote: {args.output_dir / 'critical_locations.csv'}")
    print(f"Wrote: {args.output_dir / 'feature_neighborhoods.csv'}")
    print(f"Wrote: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
