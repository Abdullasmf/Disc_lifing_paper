r"""Run one fixed-geometry mesh diagnostic.

Example:
  & c:\Users\abfat\miniconda3\envs\MLEnv\python.exe Data_gen\mesh_feature_diagnostics.py --mesh medium
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MESH_PRESETS = {
    "coarse": {"LC_EDGE": 0.80, "LC_FILLET": 0.50},
    "medium": {"LC_EDGE": 0.50, "LC_FILLET": 0.30},
    "fine": {"LC_EDGE": 0.30, "LC_FILLET": 0.18},
}

LANDMARKS = [
    "front_groove_entry",
    "front_groove_floor",
    "front_groove_exit",
    "front_root",
    "rim_core_reference",
    "rear_root",
    "rear_outer_corner",
    "rear_platform_load_face_centroid",
]


def stats(stress: np.ndarray, life: np.ndarray) -> dict:
    return {
        "n_nodes": int(stress.size),
        "stress_median_mpa": float(np.median(stress)),
        "stress_p90_mpa": float(np.percentile(stress, 90)),
        "stress_p95_mpa": float(np.percentile(stress, 95)),
        "stress_max_mpa": float(np.max(stress)),
        "life_median": float(np.median(life)),
        "life_p10": float(np.percentile(life, 10)),
        "life_min": float(np.min(life)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-geometry local mesh convergence diagnostic.")
    p.add_argument("--mesh", choices=sorted(MESH_PRESETS), required=True)
    p.add_argument("--output-dir", type=Path, default=Path("Data_gen/output/mesh_feature_diagnostics"))
    p.add_argument("--radius-mm", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        from Data_gen import mesh_ops
        from Data_gen.sample_generator import generate_sample
    except ImportError:
        import sys
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from Data_gen import mesh_ops
        from Data_gen.sample_generator import generate_sample

    preset = MESH_PRESETS[args.mesh]
    mesh_ops.LC_EDGE = preset["LC_EDGE"]
    mesh_ops.LC_FILLET = preset["LC_FILLET"]

    out_dir = args.output_dir / args.mesh
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = generate_sample(
        param_offsets={},
        flange_param_offsets={},
        representation="full",
        seed=args.seed,
        include_debug_fields=True,
    )

    nodes = np.asarray(sample["node_coords_mm"], dtype=float)
    stress = np.asarray(sample["stress_max_vm"], dtype=float).ravel()
    life = np.asarray(sample["life_raw"], dtype=float).ravel()
    zone = np.asarray(sample["zone_id"], dtype=int).ravel()
    landmarks = sample.get("feature_landmarks_mm", {})

    rows = []
    for name in LANDMARKS:
        if name not in landmarks:
            continue
        xy = np.asarray(landmarks[name], dtype=float).ravel()[:2]
        d = np.linalg.norm(nodes - xy[None, :], axis=1)
        mask = d <= args.radius_mm
        if not np.any(mask):
            continue
        rows.append({
            "mesh": args.mesh,
            "feature": name,
            "x_mm": float(xy[0]),
            "r_mm": float(xy[1]),
            "radius_mm": args.radius_mm,
            **stats(stress[mask], life[mask]),
        })

    lt = zone == 1
    if np.any(lt):
        rows.append({
            "mesh": args.mesh,
            "feature": "lower_transition_zone",
            "x_mm": np.nan,
            "r_mm": np.nan,
            "radius_mm": np.nan,
            **stats(stress[lt], life[lt]),
        })

    write_csv(out_dir / "feature_statistics.csv", rows)

    i_peak = int(np.argmax(stress))
    i_life = int(np.argmin(life))
    summary = {
        "mesh": args.mesh,
        "mesh_settings_mm": {"LC_EDGE": mesh_ops.LC_EDGE, "LC_FILLET": mesh_ops.LC_FILLET},
        "seed": args.seed,
        "nominal_geometry": True,
        "landmark_radius_mm": args.radius_mm,
        "n_nodes": int(nodes.shape[0]),
        "global_peak_stress_mpa": float(stress[i_peak]),
        "global_peak_x_mm": float(nodes[i_peak, 0]),
        "global_peak_r_mm": float(nodes[i_peak, 1]),
        "global_peak_zone_id": int(zone[i_peak]),
        "global_min_life": float(life[i_life]),
        "global_min_life_x_mm": float(nodes[i_life, 0]),
        "global_min_life_r_mm": float(nodes[i_life, 1]),
        "global_min_life_zone_id": int(zone[i_life]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=stress, s=3, cmap="magma")
    fig.colorbar(sc, ax=ax, label="stress_max_vm [MPa]")
    for name in LANDMARKS:
        if name in landmarks:
            xy = np.asarray(landmarks[name], dtype=float).ravel()[:2]
            ax.plot(xy[0], xy[1], "co", ms=4)
            ax.annotate(name.replace("_", "\n"), xy, fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(f"Nominal stress, {args.mesh} mesh")
    fig.tight_layout()
    fig.savefig(out_dir / "stress_feature_landmarks.png", dpi=180)
    plt.close(fig)

    print(f"Mesh preset: {args.mesh}; LC_EDGE={mesh_ops.LC_EDGE:.3f} mm; LC_FILLET={mesh_ops.LC_FILLET:.3f} mm")
    print(f"Nodes: {nodes.shape[0]}; global peak stress: {stress[i_peak]:.2f} MPa; global min life: {life[i_life]:.3e}")
    print(f"Wrote: {out_dir / 'feature_statistics.csv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
