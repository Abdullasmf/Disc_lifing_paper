"""Visualization helpers for the disc-life inference notebook.

All plotting functions accept matplotlib axes and return them so that
callers can compose multi-panel figures freely.

Public API
----------
plot_geometry(data_modified, data_nominal, ax)
    Overlay current and nominal disc contours.

plot_life_field(data, life_field, ax, title, log_scale, vmin, vmax)
    Filled triangulation coloured by predicted life.

plot_life_comparison(data, life_a, life_b, label_a, label_b, figsize)
    Side-by-side or overlay comparison of two life fields.

plot_life_diff(data, life_a, life_b, label_a, label_b, ax)
    Log-ratio difference map between two life fields.

plot_sweep_results(sweep_df, param_name, locations, ax)
    Life vs swept parameter at selected disc locations.

add_zone_legend(ax)
    Add a coloured zone overlay legend to a life-field plot.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Zone colour map (matches zone_id 0-4)
_ZONE_COLORS = {
    0: "#4e79a7",  # bore — steel blue
    1: "#f28e2b",  # lower_transition — amber
    2: "#59a14f",  # web — sage green
    3: "#e15759",  # upper_transition — red
    4: "#76b7b2",  # rim — teal
}
_ZONE_NAMES = {
    0: "Bore",
    1: "Lower transition",
    2: "Web",
    3: "Upper transition",
    4: "Rim",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_triangulation(data: Dict) -> mtri.Triangulation:
    """Create a matplotlib Triangulation from mesh nodes and triangles."""
    nodes = data["mesh_nodes"]
    tris  = data["triangles"]
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)


def _life_norm(life_field: np.ndarray, vmin=None, vmax=None):
    """Return a LogNorm over the life field for consistent colouring."""
    lo = vmin if vmin is not None else max(float(life_field.min()), 1.0)
    hi = vmax if vmax is not None else float(life_field.max())
    if lo >= hi:
        hi = lo * 10
    return mcolors.LogNorm(vmin=lo, vmax=hi)


# ---------------------------------------------------------------------------
# Geometry overlay
# ---------------------------------------------------------------------------

def plot_geometry(
    data_modified: Dict,
    data_nominal: Optional[Dict] = None,
    ax: Optional[Axes] = None,
    title: str = "Disc cross-section geometry",
) -> Axes:
    """Plot disc contour(s).

    Parameters
    ----------
    data_modified:
        Output of :func:`generate_inference_data` for the modified geometry.
    data_nominal:
        Output of :func:`generate_inference_data` for the nominal geometry.
        Pass None to skip the nominal overlay.
    ax:
        Target matplotlib axes.  Created if None.
    title:
        Plot title.

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    # Nominal (dashed)
    if data_nominal is not None:
        c_nom = data_nominal["contour_points"]
        ax.plot(
            np.append(c_nom[:, 0], c_nom[0, 0]),
            np.append(c_nom[:, 1], c_nom[0, 1]),
            "k--", lw=1.2, alpha=0.55, label="Nominal",
        )

    # Modified (solid, coloured by zone)
    c_mod = data_modified["contour_points"]
    z_mod = data_modified["contour_zone_id"]

    # Draw each zone segment with its colour
    for zid, color in _ZONE_COLORS.items():
        mask = z_mod == zid
        if not np.any(mask):
            continue
        # Draw individual segments to avoid connecting across zone boundaries
        idxs = np.where(mask)[0]
        groups = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
        first = True
        for g in groups:
            if len(g) == 0:
                continue
            ax.plot(
                c_mod[g, 0], c_mod[g, 1],
                color=color, lw=2.0,
                label=_ZONE_NAMES[zid] if first else "_nolegend_",
            )
            first = False

    ax.set_xlabel("Axial x (mm)")
    ax.set_ylabel("Radial r (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    return ax


# ---------------------------------------------------------------------------
# Life-field map
# ---------------------------------------------------------------------------

def plot_life_field(
    data: Dict,
    life_field: np.ndarray,
    ax: Optional[Axes] = None,
    title: str = "Predicted life",
    log_scale: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "Life (cycles)",
    cmap: str = "plasma",
) -> Axes:
    """Plot a life field as a filled triangulation over the disc cross-section.

    Parameters
    ----------
    data:         Output of :func:`generate_inference_data`.
    life_field:   [M] life values at mesh nodes.
    ax:           Target axes.  Created if None.
    title:        Plot title.
    log_scale:    Use logarithmic colour scale.
    vmin, vmax:   Colour scale bounds.  Auto-determined if None.
    cbar_label:   Colour bar label.
    cmap:         Matplotlib colour map name.

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    triang = _make_triangulation(data)
    values = np.asarray(life_field, dtype=np.float64)

    if log_scale:
        norm = _life_norm(values, vmin, vmax)
        tpc  = ax.tripcolor(triang, values, norm=norm, cmap=cmap, shading="gouraud")
    else:
        lo = vmin if vmin is not None else float(values.min())
        hi = vmax if vmax is not None else float(values.max())
        tpc = ax.tripcolor(
            triang, values,
            vmin=lo, vmax=hi,
            cmap=cmap, shading="gouraud",
        )

    plt.colorbar(tpc, ax=ax, label=cbar_label, fraction=0.03, pad=0.04)

    # Contour overlay
    c = data["contour_points"]
    ax.plot(
        np.append(c[:, 0], c[0, 0]),
        np.append(c[:, 1], c[0, 1]),
        "w-", lw=0.8, alpha=0.7,
    )

    ax.set_xlabel("Axial x (mm)")
    ax.set_ylabel("Radial r (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_life_comparison(
    data_a: Dict,
    life_a: np.ndarray,
    data_b: Dict,
    life_b: np.ndarray,
    label_a: str = "Modified",
    label_b: str = "Nominal",
    figsize: Tuple[float, float] = (14, 5),
    log_scale: bool = True,
    shared_scale: bool = True,
) -> Figure:
    """Side-by-side life-field comparison.

    Parameters
    ----------
    data_a, data_b:
        Geometry data dicts (may differ if geometries differ).
    life_a, life_b:
        Life fields on each respective mesh.
    label_a, label_b:
        Panel titles.
    figsize:
        Figure size (width, height) in inches.
    log_scale:
        Logarithmic colour scale.
    shared_scale:
        Enforce the same vmin/vmax across both panels.

    Returns
    -------
    fig
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if shared_scale:
        combined = np.concatenate([life_a, life_b])
        vmin = float(combined.min())
        vmax = float(combined.max())
    else:
        vmin = vmax = None

    plot_life_field(data_a, life_a, ax=axes[0], title=label_a,
                    log_scale=log_scale, vmin=vmin, vmax=vmax)
    plot_life_field(data_b, life_b, ax=axes[1], title=label_b,
                    log_scale=log_scale, vmin=vmin, vmax=vmax)

    fig.tight_layout()
    return fig


def plot_life_diff(
    data: Dict,
    life_modified: np.ndarray,
    life_nominal: np.ndarray,
    label_modified: str = "Modified",
    label_nominal: str = "Nominal",
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
) -> Axes:
    """Plot log10(life_modified / life_nominal) as a signed difference map.

    Positive (blue) → modified geometry lives longer than nominal.
    Negative (red)  → modified geometry lives shorter.

    Parameters
    ----------
    data:             Geometry data (same mesh used for both life fields).
    life_modified:    [M] life field for modified geometry.
    life_nominal:     [M] life field for nominal geometry.
    ax:               Target axes.  Created if None.
    cmap:             Diverging colour map.

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    eps = 1.0  # avoid log of zero
    ratio = np.log10(
        np.clip(life_modified, eps, None) / np.clip(life_nominal, eps, None)
    )

    triang = _make_triangulation(data)
    absmax = float(np.abs(ratio).max())
    if absmax < 1e-8:
        absmax = 1.0

    tpc = ax.tripcolor(
        triang, ratio,
        vmin=-absmax, vmax=absmax,
        cmap=cmap, shading="gouraud",
    )
    plt.colorbar(tpc, ax=ax,
                 label=f"log₁₀({label_modified} / {label_nominal})",
                 fraction=0.03, pad=0.04)

    c = data["contour_points"]
    ax.plot(
        np.append(c[:, 0], c[0, 0]),
        np.append(c[:, 1], c[0, 1]),
        "k-", lw=0.8, alpha=0.7,
    )

    ax.set_xlabel("Axial x (mm)")
    ax.set_ylabel("Radial r (mm)")
    ax.set_title(f"Life ratio: {label_modified} vs {label_nominal}")
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# Sweep plots
# ---------------------------------------------------------------------------

def plot_sweep_results(
    sweep_df: "pd.DataFrame",  # type: ignore[name-defined]
    param_name: str,
    locations: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (9, 5),
    log_y: bool = True,
) -> Figure:
    """Plot key-location life vs swept parameter value.

    Parameters
    ----------
    sweep_df:    DataFrame produced by :func:`sweep_single_parameter`.
    param_name:  Name of the swept parameter (x-axis).
    locations:   Disc locations to plot.  Defaults to all five zone midpoints.
    figsize:     Figure size.
    log_y:       Use logarithmic y-axis.

    Returns
    -------
    fig
    """
    if locations is None:
        locations = ["bore", "lower_transition", "web", "upper_transition", "rim"]

    fig, ax = plt.subplots(figsize=figsize)

    models = sweep_df["model"].unique()
    # Line styles for different models
    ls_cycle = ["-", "--", ":", "-."]
    model_ls = {m: ls_cycle[i % len(ls_cycle)] for i, m in enumerate(models)}

    colors = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    loc_colors = {loc: colors[i % len(colors)] for i, loc in enumerate(locations)}

    for loc in locations:
        if loc not in sweep_df.columns:
            continue
        for model in models:
            sub = sweep_df[sweep_df["model"] == model]
            ax.plot(
                sub[param_name],
                sub[loc],
                color=loc_colors[loc],
                linestyle=model_ls[model],
                marker="o", ms=4,
                label=f"{loc} ({model})",
            )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(f"{param_name} offset (mm)")
    ax.set_ylabel("Life (cycles)")
    ax.set_title(f"Life at key disc locations vs {param_name}")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: annotate key-point locations on a life-field axis
# ---------------------------------------------------------------------------

def annotate_key_points(
    ax: Axes,
    key_indices: Dict[str, int],
    mesh_nodes: np.ndarray,
    life_field: Optional[np.ndarray] = None,
    fontsize: int = 7,
    color: str = "white",
) -> None:
    """Overlay labelled markers for each canonical disc location.

    Parameters
    ----------
    ax:           Axes carrying a life-field plot.
    key_indices:  Output of :func:`get_key_life_points`.
    mesh_nodes:   [M, 2] node coordinates.
    life_field:   If provided, life values are included in each label.
    fontsize:     Label font size.
    color:        Marker/label colour.
    """
    short_names = {
        "bore":             "B",
        "lower_transition": "LT",
        "web":              "W",
        "upper_transition": "UT",
        "rim":              "R",
    }
    for name, idx in key_indices.items():
        x, r = mesh_nodes[idx]
        label = short_names.get(name, name)
        if life_field is not None:
            label += f"\n{life_field[idx]:.2e}"
        ax.plot(x, r, "x", color=color, ms=8, mew=2)
        ax.annotate(
            label,
            xy=(x, r), fontsize=fontsize, color=color,
            ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
        )
