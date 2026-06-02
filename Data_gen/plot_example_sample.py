"""Generate and plot one deterministic synthetic sample.

This script demonstrates the upgraded Data_gen v2 pipeline:
- segment-aware geometry/region metadata,
- rotating-disc-inspired phase stress surrogate,
- phase-wise Miner/Basquin life calculation.

Output:
- PNG figure with contour semantics, full-node regions, stress_max_vm,
  life_raw, takeoff stress, and edge curvature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import GeneratorConfig, CYCLE_PHASES, sample_geometry_parameters
from .features import contour_derivative_features
from .geometry import build_disc_contour
from .mesh_ops import assign_region_and_segment_from_contour, generate_mesh
from .physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max


def create_example_plot(output_png: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    params = sample_geometry_parameters(rng)

    cfg = GeneratorConfig()
    contour = build_disc_contour(params, points_per_segment=cfg.contour_points_per_segment)
    mesh_data = generate_mesh(contour.points, cfg.mesh_grid_points_x, cfg.mesh_grid_points_r)

    region_id, _, _, _ = assign_region_and_segment_from_contour(
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
    )
    stress_max_vm = compute_stress_max(phase_stress)
    life_raw = compute_life_raw(phase_stress, region_id)

    contour_phase_stress = compute_phase_equivalent_stresses(
        nodes=contour.points,
        region_ids=contour.region_ids,
        geometry_params=params,
        landmarks_mm=contour.landmarks_mm,
    )
    dfeat = contour_derivative_features(contour.points, contour.arc_length_mm)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes.ravel()

    sc0 = ax[0].scatter(contour.points[:, 0], contour.points[:, 1], c=contour.region_ids, s=8, cmap="tab10")
    ax[0].plot(contour.points[:, 0], contour.points[:, 1], "k-", lw=0.8, alpha=0.7)
    ax[0].set_title("Contour + region IDs")
    ax[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sc0, ax=ax[0], fraction=0.046)

    sc1 = ax[1].scatter(mesh_data.nodes[:, 0], mesh_data.nodes[:, 1], c=region_id, s=4, cmap="tab10")
    ax[1].set_title("Full node cloud region IDs")
    ax[1].set_aspect("equal", adjustable="box")
    fig.colorbar(sc1, ax=ax[1], fraction=0.046)

    sc2 = ax[2].scatter(mesh_data.nodes[:, 0], mesh_data.nodes[:, 1], c=stress_max_vm, s=4, cmap="inferno")
    ax[2].set_title("stress_max_vm")
    ax[2].set_aspect("equal", adjustable="box")
    fig.colorbar(sc2, ax=ax[2], fraction=0.046)

    sc3 = ax[3].scatter(mesh_data.nodes[:, 0], mesh_data.nodes[:, 1], c=life_raw, s=4, cmap="viridis")
    ax[3].set_title("life_raw")
    ax[3].set_aspect("equal", adjustable="box")
    fig.colorbar(sc3, ax=ax[3], fraction=0.046)

    phase_idx = list(CYCLE_PHASES).index("takeoff")
    sc4 = ax[4].scatter(mesh_data.nodes[:, 0], mesh_data.nodes[:, 1], c=phase_stress[:, phase_idx], s=4, cmap="magma")
    ax[4].set_title("phase stress: takeoff")
    ax[4].set_aspect("equal", adjustable="box")
    fig.colorbar(sc4, ax=ax[4], fraction=0.046)

    sc5 = ax[5].scatter(
        contour.points[:, 0],
        contour.points[:, 1],
        c=dfeat["curvature"],
        s=10,
        cmap="cividis",
    )
    ax[5].set_title("Edge curvature feature")
    ax[5].set_aspect("equal", adjustable="box")
    fig.colorbar(sc5, ax=ax[5], fraction=0.046)

    geom_note = (
        f"seed={seed}\n"
        f"bore_radius={params['bore_radius']:.2f} mm\n"
        f"web_length={params['web_length']:.2f} mm\n"
        f"rim_thickness={params['rim_thickness']:.2f} mm"
    )
    fig.text(0.01, 0.02, geom_note, fontsize=9, family="monospace")

    for a in ax:
        a.set_xlabel("x [mm]")
        a.set_ylabel("r [mm]")

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
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
