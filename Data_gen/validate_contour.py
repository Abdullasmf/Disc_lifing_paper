"""Validation plots: nominal and asymmetric C-groove / drive-arm outer contour.

Generates diagnostic figures saved to Data_gen/output/validation_contour/:
  1. contour_comparison.png   – legacy (no features) vs new nominal contour overlay
  2. cgroove_zoom.png         – front C-groove region zoomed
  3. arm_zoom.png             – rear drive-arm region zoomed
  4. subzone_labels.png       – subzone label colour map on new contour
  5. stress_life_nominal.png  – stress + life on full mesh (nominal geometry)

Usage
-----
  python -m Data_gen.validate_contour [--output-dir <dir>] [--skip-stress]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from .config import (
        NOMINAL_GEOMETRY_MM, NOMINAL_RIM_FEATURE_MM,
        SUBZONE_NAME_TO_ID, SUBZONE_ID_TO_NAME,
        resolve_geometry_parameters, resolve_rim_feature_parameters,
        clip_rim_feature_offsets_to_bounds, radial_stations_from_params,
    )
    from .geometry import (
        build_disc_contour, sanitize_geometry_parameters, sanitize_rim_feature_parameters,
    )
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import (
        NOMINAL_GEOMETRY_MM, NOMINAL_RIM_FEATURE_MM,
        SUBZONE_NAME_TO_ID, SUBZONE_ID_TO_NAME,
        resolve_geometry_parameters, resolve_rim_feature_parameters,
        clip_rim_feature_offsets_to_bounds, radial_stations_from_params,
    )
    from Data_gen.geometry import (
        build_disc_contour, sanitize_geometry_parameters, sanitize_rim_feature_parameters,
    )
    from Data_gen.sample_generator import generate_sample


SUBZONE_COLOURS = {
    "bore":               "#4e79a7",
    "lower_transition":   "#f28e2b",
    "web":                "#59a14f",
    "upper_transition":   "#e15759",
    "rim_main":           "#76b7b2",
    "front_face":         "#edc948",
    "front_cgroove":      "#b07aa1",
    "rear_arm_neck":      "#ff9da7",
    "rear_arm_land":      "#9c755f",
    "rear_arm_corner":    "#bab0ac",
    "rear_arm_end_face":  "#d37295",
}


def _get_params_and_rim_features(geo_offsets=None, rf_offsets=None):
    params = sanitize_geometry_parameters(resolve_geometry_parameters(geo_offsets or {}))
    rf_raw = resolve_rim_feature_parameters(rf_offsets or {})
    rf = sanitize_rim_feature_parameters(rf_raw, params["rim_thickness"], params["bore_thickness"])
    return params, rf


def plot_contour_comparison(out_dir: Path) -> None:
    """Figure 1: legacy (no rim features) vs new (C-groove + arm) contour overlay."""
    params, rf = _get_params_and_rim_features()

    contour_old = build_disc_contour(params, rim_feature_params=None)
    contour_new = build_disc_contour(params, rim_feature_params=rf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax = axes[0]
    ax.plot(contour_old.points[:, 0], contour_old.points[:, 1],
            "b-", lw=1.2, label="Legacy (flat cap)")
    ax.plot(contour_new.points[:, 0], contour_new.points[:, 1],
            "r--", lw=1.5, label="New (C-groove + drive arm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm] (axial)")
    ax.set_ylabel("r [mm] (radial)")
    ax.set_title("Full disc contour: legacy vs new\n(bore/web/rim zones unchanged)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    rb = radial_stations_from_params(params)
    r4, r5 = float(rb[4]), float(rb[5])
    mask_old = contour_old.points[:, 1] > r4 - 2.0
    mask_new = contour_new.points[:, 1] > r4 - 2.0
    ax2.plot(contour_old.points[mask_old, 0], contour_old.points[mask_old, 1],
             "b-", lw=1.5, label="Legacy")
    ax2.plot(contour_new.points[mask_new, 0], contour_new.points[mask_new, 1],
             "r--", lw=2.0, label="New")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x [mm] (axial)")
    ax2.set_ylabel("r [mm] (radial)")
    ax2.set_title("Outer rim region (zoom)\nC-groove on front side, drive arm on rear")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(r5, color="gray", ls=":", lw=0.8, label=f"r5={r5:.1f}")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "contour_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'contour_comparison.png'}")


def plot_cgroove_zoom(out_dir: Path) -> None:
    """Figure 2: Front C-groove zoom (nominal + asymmetric variant)."""
    params, rf_nom = _get_params_and_rim_features()
    asym_rf_offs = {
        "front_cgroove_axial_depth":  +0.80,
        "front_cgroove_radial_span":  +0.40,
        "front_cgroove_entry_radius": +0.10,
        "front_cgroove_floor_radius": +0.10,
    }
    _, rf_asym = _get_params_and_rim_features(rf_offsets=clip_rim_feature_offsets_to_bounds(asym_rf_offs))

    contour_nom  = build_disc_contour(params, rim_feature_params=rf_nom)
    contour_asym = build_disc_contour(params, rim_feature_params=rf_asym)

    rb = radial_stations_from_params(params)
    r5 = float(rb[5])
    t_rim = float(params["rim_thickness"])
    x_front = -0.5 * t_rim

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, (label, contour) in zip(axes, [("Nominal", contour_nom), ("Asymmetric +groove", contour_asym)]):
        pts = contour.points
        mask = (pts[:, 1] > r5 - 0.5) & (pts[:, 0] < x_front + 8.0) & (pts[:, 0] > x_front - 0.5)
        ax.plot(pts[mask, 0], pts[mask, 1], "k.-", lw=1.5, ms=4)
        for key, colour in [
            ("front_cgroove_entry", "tab:blue"),
            ("front_cgroove_floor", "tab:red"),
            ("front_cgroove_exit",  "tab:green"),
            ("ligament_reference",  "tab:orange"),
        ]:
            if key in contour.landmarks_mm:
                p = contour.landmarks_mm[key]
                ax.plot(p[0], p[1], "o", color=colour, ms=9, label=key, zorder=5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Front C-groove ({label})")
        ax.set_xlabel("x [mm]")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.axvline(x_front, color="gray", ls=":", lw=0.8)

    axes[0].set_ylabel("r [mm]")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cgroove_zoom.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'cgroove_zoom.png'}")


def plot_arm_zoom(out_dir: Path) -> None:
    """Figure 3: Rear drive-arm zoom (nominal + asymmetric variant)."""
    params, rf_nom = _get_params_and_rim_features()
    asym_rf_offs = {
        "rear_arm_axial_projection":    +0.40,
        "rear_arm_radial_height":       +0.30,
        "rear_arm_neck_thickness":      -0.20,
        "rear_arm_root_radius":         +0.10,
        "rear_arm_outer_corner_radius": +0.10,
    }
    _, rf_asym = _get_params_and_rim_features(rf_offsets=clip_rim_feature_offsets_to_bounds(asym_rf_offs))

    contour_nom  = build_disc_contour(params, rim_feature_params=rf_nom)
    contour_asym = build_disc_contour(params, rim_feature_params=rf_asym)

    rb = radial_stations_from_params(params)
    r5 = float(rb[5])
    t_rim = float(params["rim_thickness"])
    x_rear = 0.5 * t_rim

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, (label, contour) in zip(axes, [("Nominal", contour_nom), ("Asymmetric +arm", contour_asym)]):
        pts = contour.points
        mask = (pts[:, 1] > r5 - 1.0) & (pts[:, 0] > x_rear - 2.0)
        ax.plot(pts[mask, 0], pts[mask, 1], "k.-", lw=1.5, ms=4)
        for key, colour in [
            ("rear_arm_root",              "tab:blue"),
            ("rear_arm_neck",              "tab:cyan"),
            ("rear_arm_outer_corner",      "tab:red"),
            ("rear_arm_load_face_centroid","tab:purple"),
            ("rim_core_reference",         "tab:gray"),
        ]:
            if key in contour.landmarks_mm:
                p = contour.landmarks_mm[key]
                ax.plot(p[0], p[1], "o", color=colour, ms=9, label=key, zorder=5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Rear drive arm ({label})")
        ax.set_xlabel("x [mm]")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.axvline(x_rear, color="gray", ls=":", lw=0.8)

    axes[0].set_ylabel("r [mm]")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "arm_zoom.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'arm_zoom.png'}")


def plot_subzone_labels(out_dir: Path) -> None:
    """Figure 4: subzone label colour map on the new contour."""
    params, rf = _get_params_and_rim_features()
    contour = build_disc_contour(params, rim_feature_params=rf)

    fig, ax = plt.subplots(figsize=(9, 7))
    pts = contour.points
    sz  = contour.subzone_ids

    legend_handles = []
    for sz_id, sz_name in sorted(SUBZONE_ID_TO_NAME.items()):
        mask = sz == sz_id
        if not np.any(mask):
            continue
        c = SUBZONE_COLOURS.get(sz_name, "gray")
        ax.scatter(pts[mask, 0], pts[mask, 1], c=c, s=8, label=sz_name)
        legend_handles.append(mpatches.Patch(color=c, label=f"{sz_id}: {sz_name}"))

    ax.plot(pts[:, 0], pts[:, 1], "k-", lw=0.5, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm] (axial)")
    ax.set_ylabel("r [mm] (radial)")
    ax.set_title("Contour coloured by subzone_id\n(C-groove + rear arm geometry)")
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "subzone_labels.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'subzone_labels.png'}")


def plot_stress_comparison(out_dir: Path) -> None:
    """Figure 5: stress + life on full mesh (nominal geometry)."""
    import matplotlib.tri as mtri
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Generating nominal full sample for stress/life plot...")
    s = generate_sample(
        param_offsets={},
        representation="full",
        seed=0,
        include_derivatives=False,
    )
    nodes = s["node_coords_mm"]
    tris  = s["triangles"]
    stress = s["stress_max_vm"]
    life_log = np.log10(np.maximum(s["life_raw"], 1e-6))

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    tcf = axes[0].tripcolor(triang, stress, cmap="inferno", shading="gouraud")
    axes[0].set_title(f"stress_max_vm [nominal]\npeak={np.max(stress):.1f} MPa")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("r [mm]")
    fig.colorbar(tcf, ax=axes[0], fraction=0.046, label="von Mises [MPa]")

    tcf2 = axes[1].tripcolor(triang, life_log, cmap="viridis", shading="gouraud")
    axes[1].set_title("log10(life_raw) [nominal, zonal S-N]")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("r [mm]")
    fig.colorbar(tcf2, ax=axes[1], fraction=0.046, label="log10(cycles)")

    fig.tight_layout()
    fname = out_dir / "stress_life_nominal.png"
    fig.savefig(fname, dpi=180)
    plt.close(fig)
    print(f"Saved: {fname}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate C-groove + drive-arm contour.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Data_gen/output/validation_contour"))
    parser.add_argument("--skip-stress", action="store_true",
                        help="Skip the FEM stress comparison plots (faster)")
    args = parser.parse_args()

    out_dir = args.output_dir
    print(f"Output directory: {out_dir}")

    plot_contour_comparison(out_dir)
    plot_cgroove_zoom(out_dir)
    plot_arm_zoom(out_dir)
    plot_subzone_labels(out_dir)

    if not args.skip_stress:
        print("Generating FEM stress comparison plots (this takes 1-3 min each)...")
        plot_stress_comparison(out_dir)
    else:
        print("Skipped FEM stress plots (--skip-stress).")

    print("Validation complete.")


if __name__ == "__main__":
    main()
