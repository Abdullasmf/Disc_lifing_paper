"""CLI for synthetic rotor-disc dataset generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from .config import GeneratorConfig, sample_geometry_parameters
    from .features import extract_config_nodes
    from .geometry import build_disc_contour
    from .io_hdf5 import CONFIG_NAMES, close_output_files, create_output_files, write_sample
    from .mesh_ops import assign_region_and_segment_from_contour, generate_mesh
    from .physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max
    from .plotting import save_validation_plot
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import GeneratorConfig, sample_geometry_parameters
    from Data_gen.features import extract_config_nodes
    from Data_gen.geometry import build_disc_contour
    from Data_gen.io_hdf5 import CONFIG_NAMES, close_output_files, create_output_files, write_sample
    from Data_gen.mesh_ops import assign_region_and_segment_from_contour, generate_mesh
    from Data_gen.physics import compute_life_raw, compute_phase_equivalent_stresses, compute_stress_max
    from Data_gen.plotting import save_validation_plot


def run_generation(cfg: GeneratorConfig):
    rng = np.random.default_rng(cfg.seed)
    out_files = create_output_files(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        edge_proximity_distance_mm=cfg.edge_proximity_distance_mm,
    )
    try:
        for sample_id in range(cfg.num_samples):
            sample_seed = int(rng.integers(0, 2**31 - 1))
            sample_rng = np.random.default_rng(sample_seed)

            params = sample_geometry_parameters(sample_rng)
            contour = build_disc_contour(params, points_per_segment=cfg.contour_points_per_segment)
            mesh_data = generate_mesh(
                contour_points=contour.points,
                grid_x=cfg.mesh_grid_points_x,
                grid_r=cfg.mesh_grid_points_r,
            )

            mesh_region_id, mesh_segment_id, mesh_nearest_contour_index, mesh_distance_to_contour = \
                assign_region_and_segment_from_contour(
                    nodes=mesh_data.nodes,
                    contour_points=contour.points,
                    contour_region_ids=contour.region_ids,
                    contour_segment_ids=contour.segment_ids,
                )

            mesh_phase_stress = compute_phase_equivalent_stresses(
                nodes=mesh_data.nodes,
                region_ids=mesh_region_id,
                geometry_params=params,
                landmarks_mm=contour.landmarks_mm,
                contour_points=contour.points,
            )
            mesh_stress_max_vm = compute_stress_max(mesh_phase_stress)
            mesh_life_raw = compute_life_raw(mesh_phase_stress, mesh_region_id)

            contour_phase_stress = compute_phase_equivalent_stresses(
                nodes=contour.points,
                region_ids=contour.region_ids,
                geometry_params=params,
                landmarks_mm=contour.landmarks_mm,
                contour_points=contour.points,
            )
            contour_stress_max_vm = compute_stress_max(contour_phase_stress)
            contour_life_raw = compute_life_raw(contour_phase_stress, contour.region_ids)

            for cfg_name in CONFIG_NAMES:
                payload = extract_config_nodes(
                    config_name=cfg_name,
                    mesh_nodes=mesh_data.nodes,
                    mesh_region_ids=mesh_region_id,
                    mesh_segment_ids=mesh_segment_id,
                    mesh_distance_to_contour=mesh_distance_to_contour,
                    mesh_stress_max_vm=mesh_stress_max_vm,
                    mesh_life_raw=mesh_life_raw,
                    mesh_phase_stress=mesh_phase_stress,
                    contour_points=contour.points,
                    contour_region_ids=contour.region_ids,
                    contour_segment_ids=contour.segment_ids,
                    contour_arc_length_mm=contour.arc_length_mm,
                    contour_stress_max_vm=contour_stress_max_vm,
                    contour_life_raw=contour_life_raw,
                    contour_phase_stress=contour_phase_stress,
                    edge_proximity_distance_mm=cfg.edge_proximity_distance_mm,
                    mesh_nearest_contour_index=mesh_nearest_contour_index,
                )
                write_sample(
                    h5f=out_files[cfg_name],
                    sample_id=sample_id,
                    payload=payload,
                    geometry_params=params,
                    segment_names=contour.segment_names,
                    segment_regions=contour.segment_regions,
                    sample_seed=sample_seed,
                )

            if cfg.save_validation_plots and sample_id < cfg.validation_plot_count:
                save_validation_plot(
                    output_dir=cfg.output_dir,
                    sample_id=sample_id,
                    contour_points=contour.points,
                    nodes=mesh_data.nodes,
                    region_id=mesh_region_id,
                    stress_max_vm=mesh_stress_max_vm,
                    life_raw=mesh_life_raw,
                    phase_stress=mesh_phase_stress,
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
