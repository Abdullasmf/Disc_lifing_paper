"""Generate and plot one deterministic synthetic sample.

This script demonstrates the upgraded Data_gen v2 pipeline:
- segment-aware geometry/region metadata,
- rotating-disc-inspired phase stress surrogate,
- phase-wise Miner/Basquin life calculation.

Output:
- PNG figure with 6 panels: contour by segment_id, full node cloud by region_id,
  stress_max_vm, life_raw, takeoff stress, and edge curvature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

try:
    from .config import GeneratorConfig, CYCLE_PHASES, sample_geometry_parameters
    from .features import contour_derivative_features, resample_contour_uniform_arc_length
    from .geometry import build_disc_contour
    from .mesh_ops import assign_region_and_segment_from_contour, generate_mesh
    from .physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import GeneratorConfig, CYCLE_PHASES, sample_geometry_parameters
    from Data_gen.features import contour_derivative_features, resample_contour_uniform_arc_length
    from Data_gen.geometry import build_disc_contour
    from Data_gen.mesh_ops import assign_region_and_segment_from_contour, generate_mesh
    from Data_gen.physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max


def create_example_plot(output_png: Path, seed: int = 7):
    """Generate one deterministic synthetic sample and save a multi-panel PNG."""
    rng = np.random.default_rng(seed)
    params = sample_geometry_parameters(rng)

    cfg = GeneratorConfig()
    contour = build_disc_contour(params, points_per_segment=cfg.contour_points_per_segment)
    mesh_data = generate_mesh(contour.points, cfg.mesh_grid_points_x, cfg.mesh_grid_points_r, seed=seed)

    region_id, segment_id, _, _ = assign_region_and_segment_from_contour(
        nodes=mesh_data.nodes,
        contour_points=contour.points,
        contour_region_ids=contour.region_ids,
        contour_segment_ids=contour.segment_ids,
    )

    phase_stress = compute_phase_equivalent_stresses(
        nodes=mesh_data.nodes,
        region_ids=region_id,
        geometry_params=params,
        landmarks_mm=contour.landmarks_mm,
        contour_points=contour.points,
    )
    stress_max_vm = compute_stress_max(phase_stress)
    life_raw = compute_life_raw(phase_stress, region_id)

    # Resampled contour for curvature panel
    resampled_pts, resampled_arc, _, _ = resample_contour_uniform_arc_length(
        contour.points, contour.arc_length_mm, contour.region_ids, contour.segment_ids
    )
    dfeat = contour_derivative_features(resampled_pts, resampled_arc)

    # Triangulation object for full-domain panels
    triang = mtri.Triangulation(mesh_data.nodes[:, 0], mesh_data.nodes[:, 1], mesh_data.triangles)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes.ravel()

    # Panel 0: contour colored by segment_id
    sc0 = ax[0].scatter(contour.points[:, 0], contour.points[:, 1], c=contour.segment_ids, s=8, cmap="tab10")
    ax[0].plot(contour.points[:, 0], contour.points[:, 1], "k-", lw=0.8, alpha=0.7)
    ax[0].set_title("Contour colored by segment_id")
    ax[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sc0, ax=ax[0], fraction=0.046)

    # Panel 1: triangulated region map
    tc1 = ax[1].tripcolor(triang, region_id, cmap="tab10", shading="flat")
    ax[1].set_title("Region map (triangulated)")
    ax[1].set_aspect("equal", adjustable="box")
    fig.colorbar(tc1, ax=ax[1], fraction=0.046)

    # Panel 2: triangulated stress_max_vm
    tc2 = ax[2].tripcolor(triang, stress_max_vm, cmap="inferno", shading="gouraud")
    ax[2].set_title("stress_max_vm")
    ax[2].set_aspect("equal", adjustable="box")
    fig.colorbar(tc2, ax=ax[2], fraction=0.046)

    # Panel 3: triangulated life_raw
    tc3 = ax[3].tripcolor(triang, life_raw, cmap="viridis", shading="gouraud")
    ax[3].set_title("life_raw")
    ax[3].set_aspect("equal", adjustable="box")
    fig.colorbar(tc3, ax=ax[3], fraction=0.046)

    # Panel 4: triangulated takeoff phase stress
    phase_idx = list(CYCLE_PHASES).index("takeoff")
    tc4 = ax[4].tripcolor(triang, phase_stress[:, phase_idx], cmap="magma", shading="gouraud")
    ax[4].set_title("Phase stress: takeoff")
    ax[4].set_aspect("equal", adjustable="box")
    fig.colorbar(tc4, ax=ax[4], fraction=0.046)

    # Panel 5: edge curvature using resampled contour
    sc5 = ax[5].scatter(
        resampled_pts[:, 0],
        resampled_pts[:, 1],
        c=dfeat["curvature"],
        s=10,
        cmap="cividis",
    )
    ax[5].set_title("Edge curvature feature (resampled)")
    ax[5].set_aspect("equal", adjustable="box")
    fig.colorbar(sc5, ax=ax[5], fraction=0.046)

    geom_note = (
        f"seed={seed}\n"
        f"bore_radius={params['bore_radius']:.2f} mm\n"
        f"web_length={params['web_length']:.2f} mm\n"
        f"rim_thickness={params['rim_thickness']:.2f} mm\n"
        f"contour_pts={contour.points.shape[0]}\n"
        f"full_nodes={mesh_data.nodes.shape[0]}\n"
        f"triangles={mesh_data.triangles.shape[0]}"
    )
    fig.text(0.01, 0.02, geom_note, fontsize=9, family="monospace")

    for a in ax:
        a.set_xlabel("x [mm]")
        a.set_ylabel("r [mm]")

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one deterministic example figure for Data_gen pipeline.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data_gen/output/example_sample.png"),
        help="PNG output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_example_plot(output_png=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
