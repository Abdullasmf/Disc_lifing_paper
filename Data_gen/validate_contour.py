"""Validation plots: old (no-step) vs new (stepd) outer contour.

Generates diagnostic figures saved to Data_gen/output/validation_contour/:
  1. contour_comparison.png  – old nominal vs new nominal contour overlay
  2. step_variants.png     – 4 deviated step variants on the outer-cap region
  3. stress_contour_old.png  – stress on old-style contour (no steps)
  4. stress_contour_new.png  – stress on new stepd contour (nominal steps)
  5. subzone_labels.png      – subzone label colour map on new contour

Usage
-----
  python -m Data_gen.validate_contour [--output-dir <dir>]
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
        NOMINAL_GEOMETRY_MM, SUBZONE_NAME_TO_ID, SUBZONE_ID_TO_NAME,
        resolve_flange_parameters, resolve_geometry_parameters,
        clip_flange_offsets_to_bounds,
    )
    from .geometry import (
        build_disc_contour, sanitize_flange_parameters, sanitize_geometry_parameters,
    )
    from .sample_generator import generate_sample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Data_gen.config import (
        NOMINAL_GEOMETRY_MM, SUBZONE_NAME_TO_ID, SUBZONE_ID_TO_NAME,
        resolve_flange_parameters, resolve_geometry_parameters,
        clip_flange_offsets_to_bounds,
    )
    from Data_gen.geometry import (
        build_disc_contour, sanitize_flange_parameters, sanitize_geometry_parameters,
    )
    from Data_gen.sample_generator import generate_sample


# Colour map for subzone labels
SUBZONE_COLOURS = {
    "bore":             "#4e79a7",
    "lower_transition": "#f28e2b",
    "web":              "#59a14f",
    "upper_transition": "#e15759",
    "rim_main":         "#76b7b2",
    "front_step":     "#edc948",
    "rear_step":      "#b07aa1",
    "front_shoulder":   "#ff9da7",
    "rear_shoulder":    "#9c755f",
}


def _get_params_and_steps(geo_offsets=None, flange_offsets=None):
    params = sanitize_geometry_parameters(resolve_geometry_parameters(geo_offsets or {}))
    fp_raw = resolve_flange_parameters(flange_offsets or {})
    fp = sanitize_flange_parameters(fp_raw, params["rim_thickness"])
    return params, fp


def plot_contour_comparison(out_dir: Path) -> None:
    """Figure 1: old (no-step) vs new (stepd) contour overlay."""
    params, fp = _get_params_and_steps()

    contour_old = build_disc_contour(params, flange_params=None)
    contour_new = build_disc_contour(params, flange_params=fp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: full disc contour
    ax = axes[0]
    ax.plot(contour_old.points[:, 0], contour_old.points[:, 1],
            "b-", lw=1.2, label="No steps (old)")
    ax.plot(contour_new.points[:, 0], contour_new.points[:, 1],
            "r--", lw=1.5, label="With steps (new)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm] (axial)")
    ax.set_ylabel("r [mm] (radial)")
    ax.set_title("Full disc contour comparison\n(bore/web/rim zones unchanged)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: zoom on outer rim region
    ax2 = axes[1]
    # Determine rim region bounds
    from Data_gen.config import radial_stations_from_params
    rb = radial_stations_from_params(params)
    r4, r5 = float(rb[4]), float(rb[5])
    r_step_outer = float(contour_new.landmarks_mm["r_step_outer"][0])

    mask_old = contour_old.points[:, 1] > r4 - 2.0
    mask_new = contour_new.points[:, 1] > r4 - 2.0

    ax2.plot(contour_old.points[mask_old, 0], contour_old.points[mask_old, 1],
             "b-", lw=1.5, label="No steps (old)")
    ax2.plot(contour_new.points[mask_new, 0], contour_new.points[mask_new, 1],
             "r--", lw=2.0, label="With steps (new)")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x [mm] (axial)")
    ax2.set_ylabel("r [mm] (radial)")
    ax2.set_title("Outer rim region (zoom)\nsteps visible at r > r5")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(r5, color="gray", ls=":", lw=0.8, label=f"r5={r5:.1f}")
    ax2.axhline(r_step_outer, color="orange", ls=":", lw=0.8, label=f"r_step={r_step_outer:.1f}")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "contour_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'contour_comparison.png'}")


def plot_step_variants(out_dir: Path) -> None:
    """Figure 2: outer-cap region for nominal + 4 deviated variants + 1 asymmetric variant."""
    params, fp_nom = _get_params_and_steps()

    # Define some deviated variants (within the nominal ± offset bounds).
    # Last variant is deliberately asymmetric (front != rear).
    variants = {
        "nominal":              {},
        "+fl_height+0.2":       {"front_flange_radial_height": +0.2, "rear_flange_radial_height": +0.2},
        "-fl_height-0.2":       {"front_flange_radial_height": -0.2, "rear_flange_radial_height": -0.2},
        "+fl_axial+0.3":        {"front_flange_axial_length": +0.3, "rear_flange_axial_length": +0.3},
        "+fillet+0.1":          {"front_fillet_radius": +0.1, "rear_fillet_radius": +0.1},
        "asymmetric":           {
            "front_flange_radial_height": +0.15,
            "rear_flange_radial_height":  -0.15,
            "front_flange_axial_length":  +0.20,
            "rear_flange_axial_length":   -0.20,
        },
    }

    from Data_gen.config import radial_stations_from_params
    rb = radial_stations_from_params(params)
    r4 = float(rb[4])
    t_rim = float(params["rim_thickness"])

    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 5), sharey=True)
    colours = ["k", "royalblue", "tomato", "seagreen", "darkorange"]

    for ax, (label, offs), colour in zip(axes, variants.items(), colours):
        fp = sanitize_flange_parameters(
            resolve_flange_parameters(clip_flange_offsets_to_bounds(offs)),
            t_rim,
        )
        contour = build_disc_contour(params, flange_params=fp)
        mask = contour.points[:, 1] > r4 - 1.0
        ax.plot(contour.points[mask, 0], contour.points[mask, 1], color=colour, lw=1.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("x [mm]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("r [mm]")
    fig.suptitle("Outer-cap step variants (rim region, r > r4)", fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "step_variants.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'step_variants.png'}")


def plot_subzone_labels(out_dir: Path) -> None:
    """Figure 3: subzone label colour map on the new contour."""
    params, fp = _get_params_and_steps()
    contour = build_disc_contour(params, flange_params=fp)

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
    ax.set_title("Contour coloured by subzone_id")
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "subzone_labels.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'subzone_labels.png'}")


def plot_stress_comparison(out_dir: Path) -> None:
    """Figures 4 & 5: stress on outer contour, no-step vs stepd."""
    import matplotlib.tri as mtri
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, kwargs in [
        ("no_step",    {"use_steps": False}),
        ("with_steps", {"use_steps": True, "flange_param_offsets": {}}),
    ]:
        print(f"  Generating {label} full sample for stress plot...")
        s = generate_sample(
            param_offsets={},
            representation="full",
            seed=0,
            include_derivatives=False,
            **kwargs,
        )
        nodes = s["node_coords_mm"]
        tris  = s["triangles"]
        stress = s["stress_max_vm"]

        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        tcf = axes[0].tripcolor(triang, stress, cmap="inferno", shading="gouraud")
        axes[0].set_title(f"stress_max_vm [{label}]\npeak={np.max(stress):.1f} MPa")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("r [mm]")
        fig.colorbar(tcf, ax=axes[0], fraction=0.046, label="von Mises [MPa]")

        life_log = np.log10(np.maximum(s["life_raw"], 1e-6))
        tcf2 = axes[1].tripcolor(triang, life_log, cmap="viridis", shading="gouraud")
        axes[1].set_title(f"log10(life_raw) [{label}]")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("r [mm]")
        fig.colorbar(tcf2, ax=axes[1], fraction=0.046, label="log10(cycles)")

        fig.tight_layout()
        fname = out_dir / f"stress_life_{label}.png"
        fig.savefig(fname, dpi=180)
        plt.close(fig)
        print(f"Saved: {fname}")


def plot_zoomed_steps(out_dir: Path) -> None:
    """Figure: zoomed views of front and rear step regions (nominal + asymmetric)."""
    params, fp_nom = _get_params_and_steps()

    # Asymmetric variant
    asym_offs = {
        "front_flange_radial_height": +0.15,
        "rear_flange_radial_height":  -0.15,
        "front_flange_axial_length":  +0.20,
        "rear_flange_axial_length":   -0.20,
    }
    t_rim = float(params["rim_thickness"])
    fp_asym = sanitize_flange_parameters(
        resolve_flange_parameters(clip_flange_offsets_to_bounds(asym_offs)),
        t_rim,
    )

    contour_nom  = build_disc_contour(params, flange_params=fp_nom)
    contour_asym = build_disc_contour(params, flange_params=fp_asym)

    from Data_gen.config import radial_stations_from_params
    rb = radial_stations_from_params(params)
    r4, r5 = float(rb[4]), float(rb[5])
    x_front = -0.5 * t_rim
    x_rear  = +0.5 * t_rim

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row_idx, (label, contour) in enumerate([("Nominal", contour_nom), ("Asymmetric", contour_asym)]):
        pts = contour.points
        rim_mask = pts[:, 1] > r4 - 1.0

        for col_idx, (side, x_center) in enumerate([("Front", x_front), ("Rear", x_rear)]):
            ax = axes[row_idx, col_idx]
            margin_x = 4.0
            margin_r = 1.0
            r_max = float(pts[:, 1].max()) + margin_r
            mask = rim_mask & (pts[:, 0] >= x_center - margin_x) & (pts[:, 0] <= x_center + margin_x)
            ax.plot(pts[mask, 0], pts[mask, 1], "k-", lw=1.5)
            ax.axhline(r5, color="blue", ls=":", lw=0.8, alpha=0.7, label=f"r5={r5:.1f}")
            ax.axhline(r_max - margin_r, color="orange", ls=":", lw=0.8, alpha=0.7)
            ax.axvline(x_center, color="red", ls="--", lw=0.8, alpha=0.6, label=f"x={x_center:.1f}")
            ax.set_xlim(x_center - margin_x, x_center + margin_x)
            ax.set_ylim(r5 - 1.0, r_max)
            ax.set_title(f"{label} / {side} step (zoomed)", fontsize=9)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("r [mm]")
            ax.legend(fontsize=7)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "step_zoom.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_dir/'step_zoom.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate old vs new disc contour with steps.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Data_gen/output/validation_contour"))
    parser.add_argument("--skip-stress", action="store_true",
                        help="Skip the FEM stress comparison plots (faster)")
    args = parser.parse_args()

    out_dir = args.output_dir
    print(f"Output directory: {out_dir}")

    plot_contour_comparison(out_dir)
    plot_step_variants(out_dir)
    plot_subzone_labels(out_dir)
    plot_zoomed_steps(out_dir)

    if not args.skip_stress:
        print("Generating FEM stress comparison plots (this takes 1-3 min each)...")
        plot_stress_comparison(out_dir)
    else:
        print("Skipped FEM stress plots (--skip-stress).")

    print("Validation complete.")


if __name__ == "__main__":
    main()
