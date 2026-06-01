"""CLI for synthetic rotor-disc dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import GeneratorConfig, sample_geometry_parameters
from .features import extract_config_nodes
from .geometry import build_disc_contour
from .io_hdf5 import CONFIG_NAMES, close_output_files, create_output_files, write_sample
from .mesh_ops import assign_regions_from_contour, generate_mesh
from .physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max
from .plotting import save_validation_plot


def run_generation(cfg: GeneratorConfig):
    rng = np.random.default_rng(cfg.seed)
    out_files = create_output_files(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        edge_proximity_distance_mm=cfg.edge_proximity_distance_mm,
    )
    try:
        for sample_id in range(cfg.num_samples):
            params = sample_geometry_parameters(rng)
            contour = build_disc_contour(params, points_per_segment=cfg.contour_points_per_segment)
            mesh_data = generate_mesh(
                contour_points=contour.points,
                grid_x=cfg.mesh_grid_points_x,
                grid_r=cfg.mesh_grid_points_r,
            )

            region_id = assign_regions_from_contour(mesh_data.nearest_contour_index, contour.region_ids)
            phase_stress = compute_phase_equivalent_stresses(
                mesh=mesh_data.mesh,
                nodes=mesh_data.nodes,
                region_ids=region_id,
                geometry_params=params,
            )
            stress_max_vm = compute_stress_max(phase_stress)
            life_raw = compute_life_raw(phase_stress, region_id)

            for cfg_name in CONFIG_NAMES:
                payload = extract_config_nodes(
                    config_name=cfg_name,
                    nodes=mesh_data.nodes,
                    boundary_node_ids=mesh_data.boundary_node_ids,
                    distance_to_contour=mesh_data.distance_to_contour,
                    nearest_contour_index=mesh_data.nearest_contour_index,
                    contour_points=contour.points,
                    contour_region_ids=contour.region_ids,
                    stress_max_vm=stress_max_vm,
                    life_raw=life_raw,
                    phase_stress=phase_stress,
                    edge_proximity_distance_mm=cfg.edge_proximity_distance_mm,
                )
                write_sample(
                    h5f=out_files[cfg_name],
                    sample_id=sample_id,
                    payload=payload,
                    geometry_params=params,
                    seed=cfg.seed,
                )

            if cfg.save_validation_plots and sample_id < cfg.validation_plot_count:
                save_validation_plot(
                    output_dir=cfg.output_dir,
                    sample_id=sample_id,
                    contour_points=contour.points,
                    nodes=mesh_data.nodes,
                    region_id=region_id,
                    stress_max_vm=stress_max_vm,
                    life_raw=life_raw,
                )
    finally:
        close_output_files(out_files)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic rotor-disc HDF5 datasets.")
    parser.add_argument("--num-samples", type=int, default=GeneratorConfig.num_samples)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=GeneratorConfig.output_dir)
    parser.add_argument("--edge-proximity-distance-mm", type=float, default=GeneratorConfig.edge_proximity_distance_mm)
    parser.add_argument("--mesh-grid-points-x", type=int, default=GeneratorConfig.mesh_grid_points_x)
    parser.add_argument("--mesh-grid-points-r", type=int, default=GeneratorConfig.mesh_grid_points_r)
    parser.add_argument("--save-validation-plots", action="store_true")
    parser.add_argument("--validation-plot-count", type=int, default=GeneratorConfig.validation_plot_count)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = GeneratorConfig(
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        mesh_grid_points_x=args.mesh_grid_points_x,
        mesh_grid_points_r=args.mesh_grid_points_r,
        edge_proximity_distance_mm=args.edge_proximity_distance_mm,
        save_validation_plots=args.save_validation_plots,
        validation_plot_count=args.validation_plot_count,
    )
    run_generation(cfg)


if __name__ == "__main__":
    main()

