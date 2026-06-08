"""Generate one debugging plot for one deterministic sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

try:
    from .config import CYCLE_PHASES, NOMINAL_GEOMETRY_MM, radial_stations_from_params
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import CYCLE_PHASES, NOMINAL_GEOMETRY_MM, radial_stations_from_params
    from Data_gen.sample_generator import generate_sample


CURVATURE_FEATURE_INDEX = 2  # node_features order: tangent_x, tangent_r, curvature, curvature_gradient


def _load_offsets(json_path: Path | None) -> dict[str, float]:
    if json_path is None:
        return {}
    data = json.loads(json_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Offset JSON must be a dict")
    return {k: float(v) for k, v in data.items()}


def _add_radial_threshold_lines(ax, radial_breaks: np.ndarray) -> None:
    """Draw vertical dashed lines at zone radial boundaries."""
    for r_val in radial_breaks[1:5]:  # r1, r2, r3, r4 – internal thresholds
        ax.axhline(float(r_val), color="white", lw=0.7, ls="--", alpha=0.6)


def create_example_plot(
    output_png: Path,
    representation: str,
    seed: int,
    param_offsets: dict[str, float] | None = None,
    include_derivatives: bool = True,
) -> None:
    sample = generate_sample(
        param_offsets=param_offsets or {},
        representation=representation,
        seed=seed,
        include_derivatives=include_derivatives,
        include_debug_fields=True,
    )

    full = generate_sample(
        param_offsets=param_offsets or {},
        representation="full",
        seed=seed,
        include_derivatives=False,
        include_debug_fields=True,
    )

    # Radial breaks from the full sample's actual geometry parameters.
    params = full["geometry_parameters_actual"]
    rb = radial_stations_from_params(params)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes.ravel()

    contour = sample["contour_points_mm"]
    contour_zone = sample["contour_zone_id"]

    sc0 = ax[0].scatter(contour[:, 0], contour[:, 1], c=contour_zone, s=9, cmap="tab10", vmin=0, vmax=4)
    ax[0].plot(contour[:, 0], contour[:, 1], "k-", lw=0.8, alpha=0.7)
    # Mark internal zone thresholds as horizontal lines on the contour scatter.
    for r_val in rb[1:5]:
        ax[0].axhline(float(r_val), color="gray", lw=0.8, ls="--", alpha=0.7)
    ax[0].set_title("Contour colored by zone_id\n(dashed = radial thresholds)")
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("r [mm]")
    fig.colorbar(sc0, ax=ax[0], fraction=0.046)

    triang = mtri.Triangulation(full["node_coords_mm"][:, 0], full["node_coords_mm"][:, 1], full["triangles"])

    tc1 = ax[1].tripcolor(triang, full["region_id"], cmap="tab10", shading="flat", vmin=0, vmax=2)
    _add_radial_threshold_lines(ax[1], rb)
    ax[1].set_title("Region map (threshold-based)")
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("r [mm]")
    fig.colorbar(tc1, ax=ax[1], fraction=0.046)

    tc2 = ax[2].tripcolor(triang, full["stress_max_vm"], cmap="inferno", shading="gouraud")
    _add_radial_threshold_lines(ax[2], rb)
    ax[2].set_title("stress_max_vm")
    ax[2].set_aspect("equal", adjustable="box")
    ax[2].set_xlabel("x [mm]"); ax[2].set_ylabel("r [mm]")
    fig.colorbar(tc2, ax=ax[2], fraction=0.046)

    life_log10 = np.log10(np.maximum(full["life_raw"], 1e-10))
    tc3 = ax[3].tripcolor(triang, life_log10, cmap="viridis", shading="gouraud")
    _add_radial_threshold_lines(ax[3], rb)
    ax[3].set_title("log10(life_raw) [log scale]")
    ax[3].set_aspect("equal", adjustable="box")
    ax[3].set_xlabel("x [mm]"); ax[3].set_ylabel("r [mm]")
    fig.colorbar(tc3, ax=ax[3], fraction=0.046)

    phase_idx = list(CYCLE_PHASES).index("takeoff")
    tc4 = ax[4].tripcolor(triang, full["phase_stress_eq"][:, phase_idx], cmap="magma", shading="gouraud")
    _add_radial_threshold_lines(ax[4], rb)
    ax[4].set_title("Phase stress: takeoff")
    ax[4].set_aspect("equal", adjustable="box")
    ax[4].set_xlabel("x [mm]"); ax[4].set_ylabel("r [mm]")
    fig.colorbar(tc4, ax=ax[4], fraction=0.046)

    if representation == "edge" and sample["node_features"].shape[1] > CURVATURE_FEATURE_INDEX:
        curv = sample["node_features"][:, CURVATURE_FEATURE_INDEX]
        sc5 = ax[5].scatter(sample["node_coords_mm"][:, 0], sample["node_coords_mm"][:, 1], c=curv, s=10, cmap="cividis")
        ax[5].set_title("Edge curvature")
        ax[5].set_aspect("equal", adjustable="box")
        ax[5].set_xlabel("x [mm]"); ax[5].set_ylabel("r [mm]")
        fig.colorbar(sc5, ax=ax[5], fraction=0.046)
    else:
        ax[5].text(0.05, 0.5, "Edge curvature available only\nfor edge representation", fontsize=11)
        ax[5].set_title("Edge curvature")
        ax[5].set_axis_off()

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"Plot saved: {output_png}")


def _print_validation(param_offsets: dict[str, float]) -> None:
    """Print required validation checks to stdout."""
    print("\n=== Validation report ===")

    # 1. Nominal geometry ordering
    bt = NOMINAL_GEOMETRY_MM["bore_thickness"]
    rt = NOMINAL_GEOMETRY_MM["rim_thickness"]
    wt = NOMINAL_GEOMETRY_MM["web_thickness"]
    ok_order = bt > rt > wt
    print(f"[{'PASS' if ok_order else 'FAIL'}] Nominal bore_thickness({bt}) > rim_thickness({rt}) > web_thickness({wt})")

    def _validate_one(label: str, offsets: dict[str, float], seed: int) -> None:
        s_full = generate_sample(param_offsets=offsets, representation="full", seed=seed, include_debug_fields=True)
        params = s_full["geometry_parameters_actual"]
        rb_arr = radial_stations_from_params(params)
        r1, r2, r3, r4 = rb_arr[1], rb_arr[2], rb_arr[3], rb_arr[4]

        nodes = s_full["node_coords_mm"]
        zone_ids = s_full["zone_id"]
        r_nodes = nodes[:, 1]
        in_lower_band = (r_nodes > r1) & (r_nodes <= r2)
        in_upper_band = (r_nodes > r3) & (r_nodes <= r4)
        lower_correct = np.all(zone_ids[in_lower_band] == 1) if np.any(in_lower_band) else True
        upper_correct = np.all(zone_ids[in_upper_band] == 3) if np.any(in_upper_band) else True

        t_bore = params["bore_thickness"]
        t_rim = params["rim_thickness"]
        t_web = params["web_thickness"]
        order_ok = t_bore > t_rim > t_web

        lower_pts = int(np.sum(in_lower_band))
        upper_pts = int(np.sum(in_upper_band))
        total_nodes = int(nodes.shape[0])
        lower_fraction = lower_pts / max(total_nodes, 1)
        upper_fraction = upper_pts / max(total_nodes, 1)
        lower_band_frac = (r2 - r1) / max((rb_arr[5] - rb_arr[0]), 1e-12)
        upper_band_frac = (r4 - r3) / max((rb_arr[5] - rb_arr[0]), 1e-12)
        lower_denser = lower_fraction > lower_band_frac
        upper_denser = upper_fraction > upper_band_frac

        life = s_full["life_raw"]
        zone_medians = np.array([np.median(life[zone_ids == zid]) for zid in range(5)], dtype=np.float64)
        ratio_lt_web = zone_medians[1] / max(zone_medians[2], 1e-20)
        ratio_ut_web = zone_medians[3] / max(zone_medians[2], 1e-20)
        discontinuity_ok = (abs(np.log10(ratio_lt_web)) > 0.08) and (abs(np.log10(ratio_ut_web)) > 0.08)

        stress = s_full["stress_max_vm"]
        x_nodes = nodes[:, 0]
        near_x0 = np.abs(x_nodes) < 0.5
        near_x0_in_web = near_x0 & (zone_ids == 2)
        max_stress_center_web = float(np.max(stress[near_x0_in_web])) if np.any(near_x0_in_web) else np.nan
        max_stress_transition = float(np.max(stress[(zone_ids == 1) | (zone_ids == 3)]))
        no_stripe = (max_stress_center_web < max_stress_transition) if np.isfinite(max_stress_center_web) else True

        print(f"\n-- {label} sample --")
        print(f"[{'PASS' if order_ok else 'FAIL'}] Thickness order bore({t_bore:.2f}) > rim({t_rim:.2f}) > web({t_web:.2f})")
        print(f"[{'PASS' if lower_correct else 'FAIL'}] Lower transition threshold assignment zone_id==1")
        print(f"[{'PASS' if upper_correct else 'FAIL'}] Upper transition threshold assignment zone_id==3")
        print(f"[{'PASS' if lower_denser else 'FAIL'}] Lower transition mesh density {lower_fraction:.3f} > band fraction {lower_band_frac:.3f}")
        print(f"[{'PASS' if upper_denser else 'FAIL'}] Upper transition mesh density {upper_fraction:.3f} > band fraction {upper_band_frac:.3f}")
        print(f"[{'PASS' if discontinuity_ok else 'WARN'}] Life threshold discontinuity ratios: LT/web={ratio_lt_web:.3f}, UT/web={ratio_ut_web:.3f}")
        if np.isfinite(max_stress_center_web):
            print(f"[{'PASS' if no_stripe else 'WARN'}] Transition hotspot > web centerline stress: {max_stress_transition:.1f} > {max_stress_center_web:.1f}")
        else:
            print("[SKIP] Not enough web-center nodes for centerline check")

    default_offset = {
        "bore_radius_inner": -2.0,
        "bore_height": 1.0,
        "bore_thickness": -1.2,
        "lower_transition_height": 0.8,
        "web_height": 3.0,
        "web_thickness": -0.7,
        "upper_transition_height": -0.9,
        "rim_height": 1.4,
        "rim_thickness": -0.9,
        "lower_fillet_radius": -0.6,
        "upper_fillet_radius": 0.4,
    }
    _validate_one("Nominal", {}, seed=0)
    _validate_one("Offset", param_offsets if param_offsets else default_offset, seed=13)

    print("=== End validation ===\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one sample debugging plot.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--representation", type=str, default="edge", choices=("edge", "edge_proximity", "full"))
    parser.add_argument("--offsets-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("Data_gen/output/example_sample.png"))
    parser.add_argument("--no-derivatives", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = _load_offsets(args.offsets_json)
    if not args.skip_validation:
        _print_validation(offsets)
    create_example_plot(
        output_png=args.output,
        representation=args.representation,
        seed=args.seed,
        param_offsets=offsets,
        include_derivatives=not args.no_derivatives,
    )


if __name__ == "__main__":
    main()
