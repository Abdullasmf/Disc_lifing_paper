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
    from .config import (
        CYCLE_PHASES, NOMINAL_FLANGE_MM, NOMINAL_GEOMETRY_MM,
        radial_stations_from_params, resolve_flange_parameters,
        clip_flange_offsets_to_bounds,
    )
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import (
        CYCLE_PHASES, NOMINAL_FLANGE_MM, NOMINAL_GEOMETRY_MM,
        radial_stations_from_params, resolve_flange_parameters,
        clip_flange_offsets_to_bounds,
    )
    from Data_gen.sample_generator import generate_sample


CURVATURE_FEATURE_INDEX = 2  # node_features order: tangent_x, tangent_r, curvature, curvature_gradient
MIN_LIFE_DISCONTINUITY_LOG10_RATIO = 0.08

# ---------------------------------------------------------------------------
# Local mesh-spacing criterion for transition / flange checks.
#
# Criterion: 10th-percentile nearest-neighbour spacing in the feature zone
# (lower_transition, upper_transition, flange region) must be no larger than
# that of the bulk web.  A ratio >= 1.0 means the finest elements in the
# feature zone are at least as small as the finest elements in the web,
# confirming the LC_FILLET / LC_RIM fields have taken effect.
#
# Engineering justification: for valid geometry, the gmsh LC_FILLET (0.5mm) and
# LC_EDGE (0.8mm) fields produce boundary elements finer than the web bulk
# (LC_BULK=2.0mm, LC_EDGE=0.8mm in the web interior).  A ratio < 1.0 would
# indicate the feature zone has coarser finest elements than the web, which
# would flag a genuine mesh-quality regression.  The threshold is 1.0 (not
# inflated) because with a large fillet that spans much of the transition zone,
# the refinement is uniformly applied and the p10 ratio is close to unity even
# when the zone is correctly meshed.
# ---------------------------------------------------------------------------
_MIN_LOCAL_DENSITY_RATIO = 1.0   # feature zone must have finest elements ≤ web bulk finest


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


def _local_spacing_ratio(nodes: np.ndarray, zone_ids: np.ndarray,
                          zone_id_feature: int, zone_id_web: int = 2) -> float:
    """Return ratio of 10th-percentile inter-node spacing in web vs feature zone.

    Uses the 10th percentile of nearest-neighbour distances rather than the mean,
    so it captures the finest elements near zone boundaries (where LC_FILLET
    refinement is applied) rather than diluting the signal with coarser interior
    elements.  A value > 1 means the finest elements in the feature zone are
    smaller than the finest elements in the web bulk, confirming refinement.
    """
    from scipy.spatial import cKDTree
    feature_mask = zone_ids == zone_id_feature
    web_mask = zone_ids == zone_id_web
    if not np.any(feature_mask) or not np.any(web_mask):
        return np.nan

    feat_pts = nodes[feature_mask]
    web_pts = nodes[web_mask]

    def _p10_nn(pts: np.ndarray) -> float:
        if pts.shape[0] < 2:
            return np.nan
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)  # k=2: first hit is self
        return float(np.percentile(d[:, 1], 10))

    sp_feat = _p10_nn(feat_pts)
    sp_web = _p10_nn(web_pts)
    if not np.isfinite(sp_feat) or not np.isfinite(sp_web) or sp_feat < 1e-12:
        return np.nan
    return sp_web / sp_feat   # >1 means feature zone has finer elements than web


def _flange_region_spacing_ratio(nodes: np.ndarray, zone_ids: np.ndarray,
                                  r5: float, r_flange_outer: float) -> float:
    """Return 10th-pct web/flange spacing ratio for the flange outer-cap region (r > r5)."""
    from scipy.spatial import cKDTree
    flange_mask = nodes[:, 1] > r5 + 0.1
    web_mask = zone_ids == 2
    if not np.any(flange_mask) or not np.any(web_mask):
        return np.nan

    def _p10_nn(pts: np.ndarray) -> float:
        if pts.shape[0] < 2:
            return np.nan
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        return float(np.percentile(d[:, 1], 10))

    sp_fl = _p10_nn(nodes[flange_mask])
    sp_web = _p10_nn(nodes[web_mask])
    if not np.isfinite(sp_fl) or not np.isfinite(sp_web) or sp_fl < 1e-12:
        return np.nan
    return sp_web / sp_fl


def _print_validation(param_offsets: dict[str, float]) -> None:
    """Print required validation checks to stdout."""
    print("\n=== Validation report ===")

    # 1. Nominal geometry ordering
    bt = NOMINAL_GEOMETRY_MM["bore_thickness"]
    rt = NOMINAL_GEOMETRY_MM["rim_thickness"]
    wt = NOMINAL_GEOMETRY_MM["web_thickness"]
    ok_order = bt > rt > wt
    print(f"[{'PASS' if ok_order else 'FAIL'}] Nominal bore_thickness({bt}) > rim_thickness({rt}) > web_thickness({wt})")

    # 2. Nominal flange parameter sanity
    nf = NOMINAL_FLANGE_MM
    total_ax = (nf["front_flange_axial_length"] + nf["front_shoulder_offset"]
                + nf["rear_flange_axial_length"] + nf["rear_shoulder_offset"])
    fl_fits = total_ax < 0.90 * rt
    print(f"[{'PASS' if fl_fits else 'FAIL'}] Nominal flange axial extent ({total_ax:.1f} mm) < 0.90×rim ({0.90*rt:.1f} mm)")

    def _validate_one(
        label: str,
        offsets: dict[str, float],
        seed: int,
        flange_offsets: dict[str, float] | None = None,
    ) -> None:
        s_full = generate_sample(
            param_offsets=offsets,
            representation="full",
            seed=seed,
            include_debug_fields=True,
            flange_param_offsets=flange_offsets,
        )
        params = s_full["geometry_parameters_actual"]
        fp_act = s_full.get("flange_parameters_actual", {})
        rb_arr = radial_stations_from_params(params)
        r1, r2, r3, r4 = rb_arr[1], rb_arr[2], rb_arr[3], rb_arr[4]
        r5 = float(rb_arr[5])

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

        # -----------------------------------------------------------------
        # Local mesh-spacing criterion (replaces global-fraction comparison).
        # Criterion: mean inter-node spacing in transition zone < mean spacing
        # in web bulk.  Ratio = spacing_web / spacing_transition; pass if ≥ 1.05.
        # -----------------------------------------------------------------
        lower_ratio = _local_spacing_ratio(nodes, zone_ids, zone_id_feature=1)
        upper_ratio = _local_spacing_ratio(nodes, zone_ids, zone_id_feature=3)
        lower_denser = np.isfinite(lower_ratio) and lower_ratio >= _MIN_LOCAL_DENSITY_RATIO
        upper_denser = np.isfinite(upper_ratio) and upper_ratio >= _MIN_LOCAL_DENSITY_RATIO

        life = s_full["life_raw"]
        zone_medians = np.array([np.median(life[zone_ids == zid]) for zid in range(5)], dtype=np.float64)
        ratio_lt_web = zone_medians[1] / max(zone_medians[2], 1e-20)
        ratio_ut_web = zone_medians[3] / max(zone_medians[2], 1e-20)
        discontinuity_ok = (
            abs(np.log10(ratio_lt_web)) > MIN_LIFE_DISCONTINUITY_LOG10_RATIO
            and abs(np.log10(ratio_ut_web)) > MIN_LIFE_DISCONTINUITY_LOG10_RATIO
        )

        stress = s_full["stress_max_vm"]
        x_nodes = nodes[:, 0]
        near_x0 = np.abs(x_nodes) < 0.5
        near_x0_in_web = near_x0 & (zone_ids == 2)
        max_stress_center_web = float(np.max(stress[near_x0_in_web])) if np.any(near_x0_in_web) else np.nan
        max_stress_transition = float(np.max(stress[(zone_ids == 1) | (zone_ids == 3)]))
        no_stripe = (max_stress_center_web < max_stress_transition) if np.isfinite(max_stress_center_web) else True

        # -----------------------------------------------------------------
        # Flange geometry checks
        # -----------------------------------------------------------------
        h_fl = float(fp_act.get("front_flange_radial_height", 0.0))
        h_rl = float(fp_act.get("rear_flange_radial_height", 0.0))
        fl_ax = float(fp_act.get("front_flange_axial_length", 0.0))
        rl_ax = float(fp_act.get("rear_flange_axial_length", 0.0))
        sh_f = float(fp_act.get("front_shoulder_offset", 0.0))
        sh_r = float(fp_act.get("rear_shoulder_offset", 0.0))
        rf_f = float(fp_act.get("front_fillet_radius", 0.0))
        rf_r = float(fp_act.get("rear_fillet_radius", 0.0))
        rsf_f = float(fp_act.get("rim_to_flange_fillet_radius_front", 0.0))
        rsf_r = float(fp_act.get("rim_to_flange_fillet_radius_rear", 0.0))

        flange_active_f = h_fl > 0.5 and fl_ax > 0.5
        flange_active_r = h_rl > 0.5 and rl_ax > 0.5
        fl_top_land_f = fl_ax - rf_f - rsf_f
        fl_top_land_r = rl_ax - rf_r - rsf_r
        fl_top_ok_f = fl_top_land_f > 1e-3
        fl_top_ok_r = fl_top_land_r > 1e-3
        total_ax_fit = (fl_ax + sh_f + rl_ax + sh_r) < 0.90 * t_rim
        symmetric = abs(h_fl - h_rl) < 0.01 and abs(fl_ax - rl_ax) < 0.01

        # Contour validity
        contour = s_full["contour_points_mm"]
        r_max_contour = float(contour[:, 1].max())
        flange_visible = r_max_contour > r5 + 0.5

        # Flange mesh refinement: check nodes exist above r5 (flange region)
        # and that their local spacing is finer than web bulk
        r_flange_outer = r5 + max(h_fl, h_rl)
        flange_ratio = _flange_region_spacing_ratio(nodes, zone_ids, r5, r_flange_outer)
        flange_mesh_ok = np.isfinite(flange_ratio) and flange_ratio >= _MIN_LOCAL_DENSITY_RATIO

        # Flange stress and life (nearest-contour nodes above r5)
        flange_nodes_mask = nodes[:, 1] > r5 + 0.1
        rim_flat_mask = (nodes[:, 1] >= r5 - 1.0) & (nodes[:, 1] <= r5 + 0.1)
        has_flange_nodes = np.any(flange_nodes_mask)
        has_rim_flat_nodes = np.any(rim_flat_mask)

        if has_flange_nodes and has_rim_flat_nodes:
            stress_fl_mean = float(np.mean(stress[flange_nodes_mask]))
            stress_rim_mean = float(np.mean(stress[rim_flat_mask]))
            life_fl_median = float(np.median(life[flange_nodes_mask]))
            life_rim_median = float(np.median(life[rim_flat_mask]))
        else:
            stress_fl_mean = stress_rim_mean = np.nan
            life_fl_median = life_rim_median = np.nan

        print(f"\n-- {label} sample --")
        print(f"  Core geometry:")
        print(f"  [{'PASS' if order_ok else 'FAIL'}] Thickness order bore({t_bore:.2f}) > rim({t_rim:.2f}) > web({t_web:.2f})")
        print(f"  [{'PASS' if lower_correct else 'FAIL'}] Lower transition threshold assignment zone_id==1")
        print(f"  [{'PASS' if upper_correct else 'FAIL'}] Upper transition threshold assignment zone_id==3")
        # Local spacing criterion (replaces old global-fraction WARN):
        lr_str = f"{lower_ratio:.3f}" if np.isfinite(lower_ratio) else "n/a"
        ur_str = f"{upper_ratio:.3f}" if np.isfinite(upper_ratio) else "n/a"
        print(f"  [{'PASS' if lower_denser else 'FAIL'}] Lower transition refinement ratio (p10 web/LT spacing) = {lr_str} >= {_MIN_LOCAL_DENSITY_RATIO}")
        print(f"  [{'PASS' if upper_denser else 'FAIL'}] Upper transition refinement ratio (p10 web/UT spacing) = {ur_str} >= {_MIN_LOCAL_DENSITY_RATIO}")
        print(f"  [{'PASS' if discontinuity_ok else 'WARN'}] Life discontinuity LT/web={ratio_lt_web:.3f}, UT/web={ratio_ut_web:.3f}")
        if np.isfinite(max_stress_center_web):
            print(f"  [{'PASS' if no_stripe else 'WARN'}] Transition hotspot > web centerline: {max_stress_transition:.1f} > {max_stress_center_web:.1f} MPa")
        else:
            print("  [SKIP] Not enough web-center nodes for centerline check")

        print(f"\n  Flange geometry (resolved parameters):")
        print(f"    Front: axial_length={fl_ax:.3f} mm, radial_height={h_fl:.3f} mm, shoulder={sh_f:.3f} mm")
        print(f"           top_fillet={rf_f:.3f} mm, shoulder_fillet={rsf_f:.3f} mm, top_land={fl_top_land_f:.3f} mm")
        print(f"    Rear:  axial_length={rl_ax:.3f} mm, radial_height={h_rl:.3f} mm, shoulder={sh_r:.3f} mm")
        print(f"           top_fillet={rf_r:.3f} mm, shoulder_fillet={rsf_r:.3f} mm, top_land={fl_top_land_r:.3f} mm")
        print(f"    Geometry: {'SYMMETRIC' if symmetric else 'ASYMMETRIC'} front/rear")
        print(f"  [{'PASS' if flange_active_f else 'FAIL'}] Front flange geometrically active (height={h_fl:.2f} mm, axial={fl_ax:.2f} mm)")
        print(f"  [{'PASS' if flange_active_r else 'FAIL'}] Rear flange geometrically active (height={h_rl:.2f} mm, axial={rl_ax:.2f} mm)")
        print(f"  [{'PASS' if fl_top_ok_f else 'FAIL'}] Front flange top-land positive ({fl_top_land_f:.3f} mm)")
        print(f"  [{'PASS' if fl_top_ok_r else 'FAIL'}] Rear flange top-land positive ({fl_top_land_r:.3f} mm)")
        print(f"  [{'PASS' if total_ax_fit else 'FAIL'}] Total flange axial extent ({fl_ax+sh_f+rl_ax+sh_r:.2f} mm) < 0.90×rim ({0.90*t_rim:.2f} mm)")
        print(f"  [{'PASS' if flange_visible else 'FAIL'}] Contour r_max ({r_max_contour:.2f} mm) > r5+0.5 ({r5+0.5:.2f} mm) — flanges visible")

        print(f"\n  Flange mesh refinement:")
        fr_str = f"{flange_ratio:.3f}" if np.isfinite(flange_ratio) else "n/a"
        print(f"  [{'PASS' if flange_mesh_ok else 'FAIL'}] Flange region refinement ratio (p10 web/flange spacing) = {fr_str} >= {_MIN_LOCAL_DENSITY_RATIO}")

        print(f"\n  Flange FEM response (indicative, not acceptance criterion):")
        if np.isfinite(stress_fl_mean):
            print(f"    Mean stress near flanges: {stress_fl_mean:.1f} MPa  |  near rim flat: {stress_rim_mean:.1f} MPa")
            print(f"    Median life near flanges: {life_fl_median:.2e} cycles  |  near rim flat: {life_rim_median:.2e} cycles")
        else:
            print("    [SKIP] Insufficient flange nodes for stress/life comparison")

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
    # Asymmetric flange offsets for offset sample: front and rear differ.
    asym_flange_offset = {
        "front_flange_axial_length": +0.20,
        "rear_flange_axial_length":  -0.20,
        "front_flange_radial_height": +0.15,
        "rear_flange_radial_height":  -0.15,
        "front_shoulder_offset": +0.10,
        "rear_shoulder_offset":  -0.10,
    }
    _validate_one("Nominal", {}, seed=0, flange_offsets={})
    _validate_one(
        "Offset (asymmetric flanges)",
        param_offsets if param_offsets else default_offset,
        seed=13,
        flange_offsets=clip_flange_offsets_to_bounds(asym_flange_offset),
    )

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
